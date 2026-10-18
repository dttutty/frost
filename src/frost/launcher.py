import atexit
import faulthandler
import logging
import os
import time
from uuid import uuid4

import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from frost.datatypes.env import Env
from frost.launch.config import (
    resolve_eval_metric,
    resolve_no_memory_queue_size,
    resolve_time_scaling_factor,
    stable_mode_enabled,
)
from frost.launch.affinity import (
    resolve_numa_runtime_plan,
    resolve_shared_gpu_numa_node,
)
from frost.launch.distributed import (
    configure_torch_backends,
    create_worker_process_groups,
    gather_cpu_benchmark_pids,
    initialize_process_group,
)
from frost.launch.dispatch import dispatch_host, dispatch_worker
from frost.launch.factory import build_worker_modules
from frost.launch.gpu_topology import detect_gpu_topology
from frost.launch.plan import compile_execution_plan
from frost.launch.storage import (
    prepare_control_plane,
    prepare_dynamic_store,
    prepare_launch_dataset,
    prepare_ring_buffer,
    resolve_batch_layout,
)
from frost.shm.dynamic_state_sync import SharedDynamicStateSync
from frost.launch.single_gpu_resident import (
    SingleGpuResidentDecision,
    SingleGpuResidentMode,
    estimate_resident_storage_bytes,
    parse_single_gpu_resident_mode,
    resolve_single_gpu_resident,
)
from frost.sampling.protocols import NeighborhoodSamplerProto
from frost.runtime.experimental.dynamic_state import (
    TransportMode,
    resolve_dynamic_state_policy,
    single_gpu_resident_dynamic_state_policy,
)
from frost.neg_sampler import RandomNegLinkSampler
from frost.neg_sampler.protocols import NegSamplerProto
from frost.sampling.ngh_sampler import NeighborhoodSampler, NoSampler
from frost.models.contracts import (
    InteractionMailbox,
    MemoryState,
)
from frost.models.recipes import instantiate_model_recipe
from frost.utils._logger import init_logging
from frost.utils.dtypes import RUNTIME_DTYPES
from frost.utils.utils import broadcast_data, set_seed

faulthandler.enable()
logger = logging.getLogger(__name__)

_RANDOM_NEG_LINK_SAMPLER_TARGET = "frost.neg_sampler.RandomNegLinkSampler"


def _uses_dst_partition_random_sampler(sampler_cfg: DictConfig) -> bool:
    return (
        str(sampler_cfg.get("_target_", "")) == _RANDOM_NEG_LINK_SAMPLER_TARGET
        and str(sampler_cfg.get("candidate_mode", "all")) == "dst_partition"
    )


def _instantiate_negative_samplers(
    *,
    sampler_cfg: DictConfig,
    dataframe: pd.DataFrame,
    n_nodes: int,
    dataset: str,
    train_n_neg: int,
    eval_n_neg: int,
    train_only: bool,
) -> tuple[NegSamplerProto, NegSamplerProto]:
    """Build only the negative samplers required by the execution mode."""
    train_uses_dst_partition = _uses_dst_partition_random_sampler(sampler_cfg.train)
    eval_uses_dst_partition = not train_only and _uses_dst_partition_random_sampler(
        sampler_cfg.eval
    )
    candidate_nodes = None
    if train_uses_dst_partition or eval_uses_dst_partition:
        candidate_nodes = RandomNegLinkSampler.candidates_from_dst_nodes(
            dataframe["dst"].to_numpy(copy=False), n_nodes=n_nodes
        )
        logger.info(
            "Built %d destination-partition negative candidates from the "
            "already loaded edge frame.",
            candidate_nodes.numel(),
        )

    shared_kwargs = {"n_nodes": n_nodes, "dataset": dataset}
    common_kwargs = {"seed": sampler_cfg.get("seed")}
    train_sampler: NegSamplerProto = instantiate(
        sampler_cfg.train,
        **shared_kwargs,
        **common_kwargs,
        **({"candidate_nodes": candidate_nodes} if train_uses_dst_partition else {}),
        n_neg=train_n_neg,
    )
    train_sampler.init_from_df(dataframe)
    if train_only:
        return train_sampler, train_sampler

    eval_sampler: NegSamplerProto = instantiate(
        sampler_cfg.eval,
        **shared_kwargs,
        **common_kwargs,
        **({"candidate_nodes": candidate_nodes} if eval_uses_dst_partition else {}),
        n_neg=eval_n_neg,
    )
    eval_sampler.init_from_df(dataframe)
    return train_sampler, eval_sampler


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
        use_tf32=bool(cfg.compute.tf32),
        stable_mode=stable_mode_enabled(cfg),
        is_host=is_host,
    )

    env: Env = Env.initial()
    dynamic_state_policy = resolve_dynamic_state_policy(
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
    init_logging(env.local_rank, output_dir)
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
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    shared_gpu_numa_node = resolve_shared_gpu_numa_node(
        cfg.get("numa_policy", "auto"),
        n_gpu=env.n_gpu,
        cuda_visible_devices=cuda_visible_devices,
    )
    dataset = prepare_launch_dataset(
        cfg=cfg,
        env=env,
        run_id=run_id,
        dtypes=RUNTIME_DTYPES,
        feature_numa_node=shared_gpu_numa_node,
    )
    graph = dataset.graph
    df = dataset.frame
    n_nodes = dataset.n_nodes
    ndim = dataset.ndim
    edim = dataset.edim
    nfeat = dataset.nfeat
    efeat = dataset.efeat
    has_static_features = ndim > 0 or edim > 0
    numa_runtime = resolve_numa_runtime_plan(
        cfg.get("numa_policy", "auto"),
        n_gpu=env.n_gpu,
        cuda_visible_devices=cuda_visible_devices,
        has_static_features=has_static_features,
    )
    gpu_shm_numa_nodes = list(numa_runtime.graph_ring_nodes)
    with open_dict(cfg):
        cfg.ndim = ndim
        cfg.edim = edim
    if env.is_host:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        print(cfg_dict)
    gpu_topo = detect_gpu_topology(
        env.n_gpu, cuda_visible_devices=cuda_visible_devices
    )
    if env.is_host:
        logger.info("Detected GPU topology from CUDA_VISIBLE_DEVICES: %s", gpu_topo)
        logger.info(
            "NUMA storage placement: graph_rings=%s, shared_features/state=%s",
            gpu_shm_numa_nodes,
            {
                "features": shared_gpu_numa_node,
                "state": numa_runtime.shared_state_node,
            },
        )
        logger.info("NUMA runtime policy: %s", numa_runtime.reason)

    recipe = instantiate_model_recipe(cfg.model)
    execution_plan = compile_execution_plan(recipe)
    has_mem = execution_plan.has_dynamic_state
    state_cfg = recipe.state
    if dynamic_state_policy.transport is TransportMode.DIRECT_CPU:
        if not has_mem:
            raise ValueError(
                "dynamic_state_ablation requires model.state to be configured."
            )
        state_contract = execution_plan.dynamic_state
        if not (
            isinstance(state_contract, MemoryState)
            and isinstance(state_contract.mailbox, InteractionMailbox)
            and len(execution_plan.sampling.fanout) == 1
        ):
            raise ValueError(
                "dynamic_state_ablation requires a one-hop interaction-mail "
                "memory model."
            )
        assert state_cfg is not None
        if int(state_cfg["boxsize"]) != 1:
            raise ValueError(
                "dynamic_state_ablation currently only supports model.state.boxsize=1."
            )
    mail_combine = state_cfg.get("mail_combine") if state_cfg is not None else None
    if has_mem and mail_combine is None:
        raise ValueError("Memory models require model.state.mail_combine.")
    fanout = list(execution_plan.sampling.fanout)
    root_expansion = execution_plan.sampling.root_expansion
    ngh_sampler_strategy = cfg.ngh_sampler.strategy
    time_scaling_factor = resolve_time_scaling_factor(
        cfg.ngh_sampler.get("time_scaling_factor", 0.0)
    )
    tgl_sampler_parity = bool(cfg.ngh_sampler.get("tgl_sampler_parity", False))

    no_memory_queue_size = resolve_no_memory_queue_size(cfg.get("no_memory_queue_size"))
    if len(fanout) > 1 and has_mem:
        raise ValueError("Memory models only support 0-hop or 1-hop fanout.")

    d_mail = 0
    if has_mem:
        assert state_cfg is not None
        updater_cfg = state_cfg.get("updater")
        if not isinstance(updater_cfg, dict):
            raise TypeError("model.state.updater must resolve to a mapping")
        d_mail = 2 * int(updater_cfg["d_mem"]) + cfg.edim

    runtime_cfg = cfg.get("runtime", {})
    resident_raw_mode = runtime_cfg.get("single_gpu_resident", "off")
    resident_mode = parse_single_gpu_resident_mode(resident_raw_mode)
    resident_fraction = float(
        runtime_cfg.get("single_gpu_resident_memory_fraction", 0.7)
    )
    resident_required_bytes = estimate_resident_storage_bytes(
        node_features=nfeat,
        edge_features=efeat,
        n_nodes=n_nodes,
        boxsize=int(state_cfg["boxsize"]) if state_cfg is not None else 0,
        d_mem=(
            int(state_cfg["updater"]["d_mem"])
            if state_cfg is not None
            else 0
        ),
        d_mail=d_mail,
        state_dtype=RUNTIME_DTYPES.state_torch,
        timestamp_dtype=RUNTIME_DTYPES.timestamp_torch,
    )
    total_device_bytes = 0
    if resident_mode is not SingleGpuResidentMode.OFF:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "runtime.single_gpu_resident requires an available CUDA device."
            )
        total_device_bytes = torch.cuda.get_device_properties(0).total_memory
    resident_decision = resolve_single_gpu_resident(
        resident_mode,
        n_gpu=env.n_gpu,
        required_bytes=resident_required_bytes,
        total_device_bytes=total_device_bytes,
        memory_fraction=resident_fraction,
    )
    if resident_decision.enabled and has_mem:
        state_contract = execution_plan.dynamic_state
        supported_state = (
            isinstance(state_contract, MemoryState)
            and isinstance(state_contract.mailbox, InteractionMailbox)
            and len(execution_plan.sampling.fanout) <= 1
            and state_cfg is not None
            and int(state_cfg["boxsize"]) == 1
        )
        if not supported_state:
            reason = (
                "dynamic state currently requires an interaction mailbox, "
                "boxsize=1, and at most one sampling hop"
            )
            if resident_decision.mode is SingleGpuResidentMode.ON:
                raise ValueError(
                    "runtime.single_gpu_resident=on is unsupported: " + reason + "."
                )
            resident_decision = SingleGpuResidentDecision(
                enabled=False,
                mode=resident_decision.mode,
                required_bytes=resident_decision.required_bytes,
                budget_bytes=resident_decision.budget_bytes,
                reason=reason,
            )
    single_gpu_resident = resident_decision.enabled
    if single_gpu_resident:
        if str(cfg.get("dynamic_state_ablation", "baseline")) != "baseline":
            raise ValueError(
                "runtime.single_gpu_resident cannot be combined with "
                "dynamic_state_ablation."
            )
        if has_mem:
            dynamic_state_policy = single_gpu_resident_dynamic_state_policy()
    if env.is_host:
        logger.info(
            "Single-GPU resident backend: enabled=%s, mode=%s, "
            "persistent=%.2f GiB, budget=%.2f GiB (%s)",
            single_gpu_resident,
            resident_decision.mode.value,
            resident_decision.required_bytes / 2**30,
            resident_decision.budget_bytes / 2**30,
            resident_decision.reason,
        )

    nccl_ddp, nccl_p2p = create_worker_process_groups(env)

    # Determine maximum n_neg between train and eval for shared memory sizing
    neg_sampler_cfg = cfg.neg_sampler
    train_n_neg = neg_sampler_cfg.train.get("n_neg", cfg.n_neg)
    train_only = bool(cfg.get("benchmark", {}).get("train_only", False))
    configured_eval_n_neg = neg_sampler_cfg.eval.get("n_neg", cfg.n_neg)
    eval_n_neg = train_n_neg if train_only else configured_eval_n_neg
    eval_metric = resolve_eval_metric(cfg.get("eval_metric", "ap"))
    if not train_only and eval_metric == "ap" and eval_n_neg != 1:
        raise ValueError(
            "eval_metric='ap' currently requires neg_sampler.eval.n_neg=1; "
            f"got {eval_n_neg}. Use eval_metric=mrr for one-vs-many evaluation."
        )
    max_n_neg = max(train_n_neg, eval_n_neg)
    ring_buf = prepare_ring_buffer(
        env=env,
        run_id=run_id,
        dtypes=RUNTIME_DTYPES,
        fanout=fanout,
        root_expansion=root_expansion,
        train_batch_size=per_gpu_batch_size,
        eval_batch_size=eval_per_gpu_batch_size,
        train_n_neg=train_n_neg,
        eval_n_neg=eval_n_neg,
        ring_buf_size=int(cfg.ring_buf_size),
        gpu_numa_nodes=gpu_shm_numa_nodes,
    )
    control_plane = prepare_control_plane(
        env=env,
        run_id=run_id,
        ring_buffer=ring_buf,
        n_nodes=n_nodes,
    )

    shared = prepare_dynamic_store(
        enabled=has_mem,
        env=env,
        run_id=run_id,
        dtypes=RUNTIME_DTYPES,
        n_nodes=n_nodes,
        boxsize=int(state_cfg["boxsize"]) if state_cfg is not None else 0,
        d_mem=int(state_cfg["updater"]["d_mem"]) if state_cfg is not None else 0,
        edim=int(cfg.edim),
        numa_node=numa_runtime.shared_state_node,
        gpu_resident=single_gpu_resident,
    )

    state_sync = None
    if (
        has_mem
        and env.n_gpu > 1
        and dynamic_state_policy.transport is TransportMode.ASYNC_P2P
    ):
        state_sync = SharedDynamicStateSync(env, run_id)
        atexit.register(state_sync.close)

    if env.is_host:
        # Launch host process
        from frost.runtime.host._engine import Host
        from frost.runtime.host._config import HostOptions, HostResources
        from frost.runtime.host.analyzer import AccessPlanAnalyzer
        from frost.shm._core import set_as_host

        set_as_host()
        start = time.time()
        if graph is None or df is None:
            raise RuntimeError("Host rank must load graph and dataframe before launch.")

        assert df is not None
        neg_sampler_train, neg_sampler_eval = _instantiate_negative_samplers(
            sampler_cfg=cfg.neg_sampler,
            dataframe=df,
            n_nodes=n_nodes,
            dataset=str(cfg.dataset),
            train_n_neg=int(train_n_neg),
            eval_n_neg=int(eval_n_neg),
            train_only=train_only,
        )
        ngh_sampler: NeighborhoodSamplerProto
        if not fanout:
            ngh_sampler = NoSampler(
                ring_buf=ring_buf,
                root_expansion=root_expansion,
            )
        else:
            ngh_sampler = NeighborhoodSampler(
                graph=graph,
                fanout=fanout,
                ring_buf=ring_buf,
                strategy=ngh_sampler_strategy,
                seed=cfg.seed,
                time_scaling_factor=time_scaling_factor,
                root_expansion=root_expansion,
                tgl_sampler_parity=tgl_sampler_parity,
            )
        planner = AccessPlanAnalyzer(execution_plan)
        has_nfeat = ndim > 0 and not single_gpu_resident

        if_carry_over_route: bool = bool(cfg.get("if_carry_over_route", False))
        dispatch_host(
            lambda: Host(
                resources=HostResources(
                    dataframe=df,
                    train_negative_sampler=neg_sampler_train,
                    eval_negative_sampler=neg_sampler_eval,
                    neighborhood_sampler=ngh_sampler,
                    env=env,
                    analyzer=planner,
                    gpu_topology=gpu_topo,
                    control_plane=control_plane,
                ),
                options=HostOptions(
                    has_node_features=has_nfeat,
                    has_dynamic_state=has_mem,
                    start_time=start,
                    batch_size=per_gpu_batch_size,
                    eval_batch_size=eval_per_gpu_batch_size,
                    timestamp_dtype=RUNTIME_DTYPES.timestamp_torch,
                    epochs=cfg.epoch,
                    patience=cfg.patience,
                    early_stop_tolerance=float(
                        cfg.get("early_stop_tolerance", 0.0)
                    ),
                    carry_over_route=if_carry_over_route,
                    show_progress=bool(cfg.get("tqdm", False)),
                    no_memory_queue_size=no_memory_queue_size,
                    output_dir=output_dir,
                    reset_eval_sampler_each_epoch=bool(
                        cfg.get("reset_eval_neg_sampler_each_epoch", False)
                    ),
                    share_negative_sampler_across_splits=bool(
                        cfg.get("share_neg_sampler_across_splits", False)
                    ),
                    normalize_event_time=bool(
                        cfg.get("normalize_event_time", False)
                    ),
                    persist_all_updates_to_cpu=bool(
                        cfg.get("persist_all_updates_to_cpu", False)
                    ),
                    dynamic_state_policy=dynamic_state_policy,
                    eval_metric=eval_metric,
                    train_only=train_only,
                ),
            )
        )

    else:
        # Launch worker process
        from frost.runtime.worker._config import WorkerOptions, WorkerResources
        from frost.runtime.worker._engine import Worker

        assert nccl_ddp is not None, "Worker rank must be in nccl group"
        assert nccl_p2p is not None, "Worker rank must be in nccl_p2p group"

        mem_updater, encoder, predictor, mailbox_router = build_worker_modules(
            cfg,
            recipe=recipe,
            plan=execution_plan,
            d_mail=d_mail,
        )
        dispatch_worker(
            lambda: Worker(
                resources=WorkerResources(
                    dynamic_store=shared,
                    node_features=nfeat,
                    edge_features=efeat,
                    env=env,
                    ddp_group=nccl_ddp,
                    p2p_group=nccl_p2p,
                    memory_updater=mem_updater,
                    encoder=encoder,
                    predictor=predictor,
                    ring_buffer=ring_buf,
                    control_plane=control_plane,
                    mailbox_router=mailbox_router,
                    state_sync=state_sync,
                ),
                options=WorkerOptions(
                    max_negative_samples=max_n_neg,
                    checkpoint_dir=output_dir,
                    optimizer=cfg.optimizer,
                    learning_rate=cfg.lr,
                    mail_combine=mail_combine,
                    execution_plan=execution_plan,
                    dynamic_state_policy=dynamic_state_policy,
                    compile_model=cfg.get("compile", True),
                    compile_mode=cfg.get("compile_mode"),
                    amp=bool(cfg.compute.amp),
                    no_memory_queue_size=no_memory_queue_size,
                    cpu_benchmark_pid_list=cpu_benchmark_pid_list,
                    eval_metric=eval_metric,
                    train_only=train_only,
                    single_gpu_resident=single_gpu_resident,
                    early_stop_tolerance=float(
                        cfg.get("early_stop_tolerance", 0.0)
                    ),
                ),
            )
        )
