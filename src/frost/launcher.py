import faulthandler
import logging
import os
import time
from uuid import uuid4

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from frost.ablation import resolve_dynamic_state_ablation
from frost.datatypes.env import Env
from frost.launch.config import (
    APAN_ENCODER_TARGET,
    TGN_ENCODER_TARGET,
    normalize_fanout,
    resolve_dynamic_state_fetch_mode,
    resolve_eval_metric,
    resolve_no_memory_queue_size,
    resolve_time_scaling_factor,
    stable_mode_enabled,
)
from frost.launch.distributed import (
    configure_torch_backends,
    create_worker_process_groups,
    gather_cpu_benchmark_pids,
    initialize_process_group,
)
from frost.launch.dispatch import dispatch_host, dispatch_worker
from frost.launch.factory import build_worker_modules
from frost.launch.storage import (
    prepare_dynamic_store,
    prepare_launch_dataset,
    prepare_ring_buffer,
    resolve_batch_layout,
)
from frost.sampling.protocols import NghSamplerProto
from frost.runtime.host._plan_router import detect_gpu_topology
from frost.neg_sampler import PrecomputedNegSampler
from frost.sampling.ngh_sampler import MultiHopNghSampler, NghSampler, NoSampler
from frost.utils._logger import init_logging
from frost.utils.precision import resolve_precision
from frost.utils.utils import broadcast_data, set_seed

faulthandler.enable()
logger = logging.getLogger(__name__)


def launch(cfg: DictConfig):
    """Initialize distributed training and launch Host or Worker processes.

    Sets up the gloo process group, loads and prepares the dataset, builds shared
    memory structures (RingBuf, MemBoxStoreManager), then forks behavior based on
    rank: the highest rank becomes the Host (orchestrates sampling and scattering),
    while all other ranks become Workers (run the training loop on GPU).

    Args:
        cfg: Hydra DictConfig containing all experiment hyperparameters including
            dataset, model, sampler, optimizer, and runtime settings.
    """
    _, _, is_host = initialize_process_group()
    configure_torch_backends(
        use_tf32=bool(cfg.get("tf32", True)),
        stable_mode=stable_mode_enabled(cfg),
        is_host=is_host,
    )

    env: Env = Env.initial()
    dynamic_state_ablation = resolve_dynamic_state_ablation(
        cfg.get("dynamic_state_ablation", "baseline")
    )
    cpu_util_bench_enabled = os.getenv("FROST_CPU_UTIL_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cpu_benchmark_pid_list = gather_cpu_benchmark_pids(
        env, enabled=cpu_util_bench_enabled
    )
    batch_layout = resolve_batch_layout(
        macro_batch_size=int(cfg.macro_batch_size),
        eval_macro_batch_size=cfg.get("eval_macro_batch_size"),
        n_gpu=env.n_gpu,
    )
    per_gpu_batch_size = batch_layout.train_per_gpu
    eval_macro_batch_size = batch_layout.eval_macro
    eval_per_gpu_batch_size = batch_layout.eval_per_gpu
    if cfg.seed is not None:
        set_seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    init_logging(env.local_rank, env.is_host, output_dir)
    # Use a per-job unique namespace for shared-memory segments so concurrent
    # launches cannot collide when they start within the same second.
    run_id = broadcast_data(
        f"run_{uuid4().hex}" if env.is_host else None, src=env.host_rank
    )
    if env.is_host:
        logger.info(
            "macro_batch_size=%d, n_gpu=%d -> per_gpu_batch_size=%d",
            cfg.macro_batch_size,
            env.n_gpu,
            per_gpu_batch_size,
        )
        logger.info(
            "eval_macro_batch_size=%d -> eval_per_gpu_batch_size=%d",
            eval_macro_batch_size,
            eval_per_gpu_batch_size,
        )
        logger.info("Using shared-memory run_id=%s", run_id)
    precision = resolve_precision(cfg)
    dataset = prepare_launch_dataset(
        cfg=cfg, env=env, run_id=run_id, precision=precision
    )
    graph = dataset.graph
    df = dataset.frame
    n_nodes = dataset.n_nodes
    ndim = dataset.ndim
    edim = dataset.edim
    nfeat = dataset.nfeat
    efeat = dataset.efeat
    with open_dict(cfg):
        cfg.ndim = ndim
        cfg.edim = edim
    if env.is_host:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        print(cfg_dict)
    gpu_topo = detect_gpu_topology(
        env.n_gpu, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if env.is_host:
        logger.info("Detected GPU topology from CUDA_VISIBLE_DEVICES: %s", gpu_topo)

    has_mem = cfg.get("mem_updater") is not None
    encoder_target = str(cfg.encoder._target_)
    if dynamic_state_ablation.requires_tgn_boxsize_one_only:
        if not has_mem:
            raise ValueError(
                "dynamic_state_ablation requires mem_updater to be configured."
            )
        if encoder_target != TGN_ENCODER_TARGET:
            raise ValueError(
                "dynamic_state_ablation is only supported for encoder=tgn."
            )
        if int(cfg.mem_updater.boxsize) != 1:
            raise ValueError(
                "dynamic_state_ablation currently only supports mem_updater.boxsize=1."
            )
    mail_recipients = int(cfg.mem_updater.get("mail_recipients", 0)) if has_mem else 0
    mail_combine = cfg.mem_updater.get("mail_combine") if has_mem else None
    if mail_recipients < 0:
        raise ValueError("mem_updater.mail_recipients must be >= 0.")
    if has_mem and mail_combine is None:
        raise ValueError("Memory models require mem_updater.mail_combine.")
    is_apan = encoder_target == APAN_ENCODER_TARGET

    if is_apan:
        if not has_mem:
            raise ValueError("APAN requires mem_updater to be configured.")
        if mail_recipients <= 0:
            raise ValueError("APAN requires mem_updater.mail_recipients > 0.")
        fanout = [mail_recipients]
        sample_4_neg = False
        ngh_sampler_strategy = cfg.ngh_sampler.strategy
    else:
        if has_mem and mail_recipients != 0:
            raise ValueError(
                "Non-APAN memory updaters must set mem_updater.mail_recipients=0."
            )
        encoder_fanout_cfg = cfg.encoder.get("fanout")
        raw_fanout = (
            OmegaConf.to_container(encoder_fanout_cfg, resolve=True)
            if OmegaConf.is_config(encoder_fanout_cfg)
            else encoder_fanout_cfg
        )
        fanout = normalize_fanout(raw_fanout)
        sample_4_neg = bool(fanout)
        ngh_sampler_strategy = cfg.ngh_sampler.strategy
    time_scaling_factor = resolve_time_scaling_factor(
        None if is_apan else cfg.ngh_sampler.get("time_scaling_factor", 0.0)
    )
    tgl_sampler_parity = bool(cfg.ngh_sampler.get("tgl_sampler_parity", False))

    postman_enabled = mail_recipients > 0
    no_memory_queue_size = resolve_no_memory_queue_size(cfg.get("no_memory_queue_size"))
    use_shm_instead = resolve_dynamic_state_fetch_mode(
        cfg,
        has_mem=has_mem,
        fanout=fanout,
    )

    if len(fanout) > 1 and has_mem:
        raise ValueError("Memory models only support 0-hop or 1-hop fanout.")

    d_mail = 0
    if has_mem:
        d_mail = 2 * cfg.mem_updater.d_mem + cfg.edim

    nccl_ddp, nccl_p2p = create_worker_process_groups(env)

    # Determine maximum n_neg between train and eval for shared memory sizing
    neg_sampler_cfg = cfg.get("neg_sampler", {})
    if "_target_" in neg_sampler_cfg:
        train_n_neg = neg_sampler_cfg.get("n_neg", cfg.n_neg)
        eval_n_neg = train_n_neg
    elif "train" in neg_sampler_cfg and "eval" in neg_sampler_cfg:
        train_n_neg = neg_sampler_cfg.train.get("n_neg", cfg.n_neg)
        eval_n_neg = neg_sampler_cfg.eval.get("n_neg", cfg.n_neg)
    else:
        train_n_neg = cfg.n_neg
        eval_n_neg = cfg.n_neg
    eval_metric = resolve_eval_metric(cfg.get("eval_metric", "ap"))
    if eval_metric == "ap" and eval_n_neg != 1:
        raise ValueError(
            "eval_metric='ap' currently requires neg_sampler.eval.n_neg=1; "
            f"got {eval_n_neg}. Use eval_metric=mrr for one-vs-many evaluation."
        )
    max_n_neg = max(train_n_neg, eval_n_neg)
    ring_buf = prepare_ring_buffer(
        env=env,
        run_id=run_id,
        precision=precision,
        fanout=fanout,
        sample_4_neg=sample_4_neg,
        train_batch_size=per_gpu_batch_size,
        eval_batch_size=eval_per_gpu_batch_size,
        train_n_neg=train_n_neg,
        eval_n_neg=eval_n_neg,
        ring_buf_size=int(cfg.ring_buf_size),
    )

    shared = prepare_dynamic_store(
        enabled=has_mem,
        env=env,
        run_id=run_id,
        precision=precision,
        n_nodes=n_nodes,
        boxsize=int(cfg.mem_updater.boxsize) if has_mem else 0,
        d_mem=int(cfg.mem_updater.d_mem) if has_mem else 0,
        edim=int(cfg.edim),
    )

    if env.is_host:
        # Launch host process
        from frost.runtime.host._engine import Host
        from frost.runtime.host.analyzer import StaticAnalyzer
        from frost.shm._core import set_as_host

        set_as_host()
        start = time.time()
        if graph is None or df is None:
            raise RuntimeError("Host rank must load graph and dataframe before launch.")

        neg_sampler_cfg = cfg.neg_sampler
        # Support both flat configs (e.g. precomputed.yaml: has _target_ at root)
        # and nested train/eval configs (e.g. random.yaml: has train/eval sub-keys).
        _shared_kwargs = dict(n_nodes=n_nodes, dataset=cfg.dataset)
        if "_target_" in neg_sampler_cfg:
            # Flat format — same sampler for train and eval
            _sampler = instantiate(neg_sampler_cfg, **_shared_kwargs)
            neg_sampler_train: PrecomputedNegSampler = _sampler
            neg_sampler_eval: PrecomputedNegSampler = _sampler
        else:
            # Nested format — separate train / eval samplers
            _common = dict(
                seed=neg_sampler_cfg.get("seed"),
            )
            # Make sure n_neg uses the specific sub-config or falls back to global
            neg_sampler_train = instantiate(
                neg_sampler_cfg.train, **_shared_kwargs, **_common, n_neg=train_n_neg
            )
            neg_sampler_eval = instantiate(
                neg_sampler_cfg.eval, **_shared_kwargs, **_common, n_neg=eval_n_neg
            )
        # Historical / inductive samplers need the full edge DataFrame to build
        # their candidate edge sets.  Call init_from_df() if the sampler supports it.
        assert df is not None
        for _ns in (neg_sampler_train, neg_sampler_eval):
            if callable(getattr(_ns, "init_from_df", None)):
                _ns.init_from_df(df)
        ngh_sampler: NghSamplerProto
        if not fanout:
            ngh_sampler = NoSampler(
                graph=graph,
                ring_buf=ring_buf,
                n_neg=max_n_neg,
                sample_4_neg=False,
            )
        elif len(fanout) == 1:
            ngh_sampler = NghSampler(
                graph=graph,
                fanout=fanout,
                ring_buf=ring_buf,
                sample_4_neg=sample_4_neg,
                n_neg=max_n_neg,
                strategy=ngh_sampler_strategy,
                seed=cfg.seed,
                time_scaling_factor=time_scaling_factor,
                tgl_sampler_parity=tgl_sampler_parity,
            )
        else:
            ngh_sampler = MultiHopNghSampler(
                graph=graph,
                fanout=fanout,
                ring_buf=ring_buf,
                n_neg=max_n_neg,
                strategy=ngh_sampler_strategy,
                seed=cfg.seed,
                time_scaling_factor=time_scaling_factor,
                sample_4_neg=True,
                tgl_sampler_parity=tgl_sampler_parity,
            )
        planner_cfg = cfg.get("access_analyzer")
        if planner_cfg is None:
            if has_mem:
                raise ValueError(
                    "Memory models must configure access_analyzer explicitly."
                )
            planner = StaticAnalyzer()
        else:
            planner = instantiate(planner_cfg)
        has_nfeat = ndim > 0
        has_efeat = edim > 0

        if_carry_over_route: bool = bool(cfg.get("if_carry_over_route", False))
        dispatch_host(
            lambda: Host(
                has_nfeat=has_nfeat,
                has_efeat=has_efeat,
                has_mem=has_mem,
                mail_same_plan_as_mem=False,
                df=df,
                env=env,
                start_time=start,
                planner=planner,
                batch_size=per_gpu_batch_size,
                neg_sampler_train=neg_sampler_train,
                neg_sampler_eval=neg_sampler_eval,
                ngh_sampler=ngh_sampler,
                time_dtype=precision.time_preprocessing_torch,
                n_epoch=cfg.epoch,
                patience=cfg.patience,
                gpu_topo=gpu_topo,
                if_carry_over_route=if_carry_over_route,
                show_tqdm=bool(cfg.get("tqdm", False)),
                no_memory_queue_size=no_memory_queue_size,
                output_dir=output_dir,
                reset_eval_neg_sampler_each_epoch=bool(
                    cfg.get("reset_eval_neg_sampler_each_epoch", False)
                ),
                share_neg_sampler_across_splits=bool(
                    cfg.get("share_neg_sampler_across_splits", False)
                ),
                normalize_event_time=bool(cfg.get("normalize_event_time", False)),
                persist_all_updates_to_cpu=bool(
                    cfg.get("persist_all_updates_to_cpu", False)
                )
                or use_shm_instead,
                dynamic_state_ablation=dynamic_state_ablation,
                eval_metric=eval_metric,
                eval_batch_size=eval_per_gpu_batch_size,
            )
        )

    else:
        # Launch worker process
        from frost.runtime.worker._engine import Worker

        assert nccl_ddp is not None, "Worker rank must be in nccl group"
        assert nccl_p2p is not None, "Worker rank must be in nccl_p2p group"

        mem_updater, encoder, predictor, postman, loss_fn = build_worker_modules(
            cfg,
            encoder_target=encoder_target,
            fanout=fanout,
            d_mail=d_mail,
            max_n_neg=max_n_neg,
            postman_enabled=postman_enabled,
        )
        if use_shm_instead and not bool(cfg.get("__debug_use_shm_instead", False)):
            logger.info(
                "Auto-enabled shared-memory dynamic-state fetch path for root-only memory model "
                "encoder=%s.",
                encoder_target,
            )
        dispatch_worker(
            lambda: Worker(
                n_neg=max_n_neg,
                mem_box_store=shared,
                ring_buf=ring_buf,
                save_to=output_dir,
                nfeat=nfeat,
                efeat=efeat,
                env=env,
                nccl_ddp=nccl_ddp,
                nccl_p2p=nccl_p2p,
                mem_updater=mem_updater,
                encoder=encoder,
                predictor=predictor,
                postman=postman,
                criterion=loss_fn,
                optimizer=cfg.optimizer,
                lr=cfg.lr,
                mail_combine=mail_combine,
                use_shm_instead=use_shm_instead,
                dynamic_state_ablation=dynamic_state_ablation,
                compile_model=cfg.get("compile", True),
                amp=bool(cfg.get("amp", False)),
                no_memory_queue_size=no_memory_queue_size,
                cpu_benchmark_pid_list=cpu_benchmark_pid_list,
                eval_metric=eval_metric,
            )
        )
