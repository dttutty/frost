import faulthandler
import logging
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from frost.datatypes.env import Env
from frost.protocols.mem_updater_proto import MemoryUpdaterProto
from frost.protocols.ngh_sampler_proto import NghSamplerProto
from frost.protocols.postman_proto import PostmanProto
from frost.runtime._thread_base import destroy_process_group
from frost.runtime.host._plan_router import detect_gpu_topology
from frost.runtime.worker.postman import Postman
from frost.neg_sampler import PrecomputedNegSampler
from frost.sampling.ngh_sampler import MultiHopNghSampler, NghSampler, NoSampler
from frost.shm.dataset_mgr import DatasetManager
from frost.shm.mem_mgr import MemBoxStoreManager
from frost.shm.ring_buf import RingBuf
from frost.utils._logger import init_logging
from frost.utils.precision import resolve_precision
from frost.utils.utils import broadcast_data, set_seed

faulthandler.enable()
logger = logging.getLogger(__name__)
APAN_ENCODER_TARGET = "frost.models.encoder.apan.APAN"
TGAT_ENCODER_TARGET = "frost.models.encoder.tgat.TGAT"
DYGFORMER_ENCODER_TARGET = "frost.models.encoder.dygformer.DyGFormer"
CAWN_ENCODER_TARGET = "frost.models.encoder.cawn.CAWN"
GRAPH_MIXER_ENCODER_TARGET = "frost.models.encoder.graph_mixer.GraphMixer"
TCL_ENCODER_TARGET = "frost.models.encoder.tcl.TCL"
LINK_PREDICTOR_TARGET = "frost.models.header.LinkPredictor"
PAIR_AWARE_PREDICTOR_TARGET = "frost.models.header.DyGFormerLinkPredictor"
TCL_PAIR_AWARE_PREDICTOR_TARGET = "frost.models.header.PairAwareMergeLayerPredictor"
PAIR_AWARE_ENCODER_TARGETS = frozenset(
    {DYGFORMER_ENCODER_TARGET, CAWN_ENCODER_TARGET, TCL_ENCODER_TARGET}
)
DYGLIB_ALIGNED_FEATURE_DIM = 172
DEFAULT_NO_MEMORY_QUEUE_SIZE = 4
REPO_ROOT = Path(__file__).resolve().parents[2]
DYGLIB_PROCESSED_DATA_ROOT = REPO_ROOT / "third_party" / "DyGLib" / "processed_data"


def _normalize_fanout(raw_fanout: object) -> list[int]:
    if raw_fanout is None:
        return []
    if isinstance(raw_fanout, int):
        return [raw_fanout]
    if isinstance(raw_fanout, Iterable) and not isinstance(raw_fanout, str | bytes):
        return [int(n_ngh) for n_ngh in raw_fanout]
    raise TypeError(f"Unsupported fanout config type: {type(raw_fanout)!r}")


def _set_nested_attr(root: object, path: tuple[str, ...], value: Any) -> None:
    target: Any = root
    for attr in path[:-1]:
        target = getattr(target, attr)
    setattr(target, path[-1], value)


def _resolve_dygformer_dummy_edge_dim(
    dataset: str,
    requested_dim: object,
    *,
    processed_data_root: Path | None = None,
) -> int:
    """Resolve DyGFormer's synthetic edge-feature width for DyGLib parity.

    DyGLib pads link-prediction node/edge features to a shared width of 172.
    If a processed edge-feature file is available locally, use its raw width but
    still apply the DyGLib floor so datasets with missing features (for example
    our generated LASTFM zeros with width 1) resolve to 172.
    """
    if requested_dim is not None:
        return int(requested_dim)

    dataset_key = dataset.strip().lower()
    data_root = processed_data_root or DYGLIB_PROCESSED_DATA_ROOT
    edge_feat_path = data_root / dataset_key / f"ml_{dataset_key}.npy"

    raw_edge_dim: int | None = None
    if edge_feat_path.exists():
        try:
            edge_raw_features = np.load(edge_feat_path, mmap_mode="r")
            if edge_raw_features.ndim == 2:
                raw_edge_dim = int(edge_raw_features.shape[1])
        except Exception as exc:
            logger.warning(
                "Failed to inspect DyGLib edge features at %s: %s",
                edge_feat_path,
                exc,
            )

    if raw_edge_dim is not None:
        resolved_dim = max(raw_edge_dim, DYGLIB_ALIGNED_FEATURE_DIM)
        logger.info(
            "Resolved DyGFormer dummy_edge_dim=%d for dataset=%s from DyGLib edge features "
            "(raw=%d, aligned floor=%d).",
            resolved_dim,
            dataset,
            raw_edge_dim,
            DYGLIB_ALIGNED_FEATURE_DIM,
        )
        return resolved_dim

    logger.info(
        "Resolved DyGFormer dummy_edge_dim=%d for dataset=%s using DyGLib's "
        "default aligned feature width.",
        DYGLIB_ALIGNED_FEATURE_DIM,
        dataset,
    )
    return DYGLIB_ALIGNED_FEATURE_DIM


def _resolve_predictor_cfg(
    encoder_target: str,
    predictor_cfg: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Swap in the pair-aware predictor for encoders that emit pair-specific layouts."""
    resolved = dict(predictor_cfg)
    switched = (
        encoder_target in PAIR_AWARE_ENCODER_TARGETS
        and resolved.get("_target_") == LINK_PREDICTOR_TARGET
    )
    if switched:
        if encoder_target == TCL_ENCODER_TARGET:
            resolved["_target_"] = TCL_PAIR_AWARE_PREDICTOR_TARGET
        else:
            resolved["_target_"] = PAIR_AWARE_PREDICTOR_TARGET
    return resolved, switched


def _resolve_no_memory_queue_size(configured_size: object | None) -> int:
    """Return the runtime queue depth used by stateless/no-memory models."""
    if configured_size is None:
        return DEFAULT_NO_MEMORY_QUEUE_SIZE
    try:
        resolved = int(configured_size)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"no_memory_queue_size must be an integer, got {configured_size!r}"
        ) from exc
    if resolved < 1:
        raise ValueError("no_memory_queue_size must be >= 1.")
    return resolved


def _resolve_time_scaling_factor(configured_factor: object | None) -> float:
    """Return a validated time-scaling factor for time_interval_aware sampling."""
    if configured_factor is None:
        return 0.0
    try:
        resolved = float(configured_factor)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"time_scaling_factor must be a float, got {configured_factor!r}"
        ) from exc
    if not math.isfinite(resolved):
        raise ValueError("time_scaling_factor must be finite.")
    return resolved


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
    _rank = int(os.environ.get("RANK", 0))
    _world_size = int(os.environ.get("WORLD_SIZE", 1))
    _is_host = _rank == _world_size - 1
    dist.init_process_group(backend="gloo")

    # Control TF32 globally so experiments can isolate TF32 vs compile speedups.
    # Use both the canonical set_float32_matmul_precision (sets old+new APIs
    # consistently) and the new fp32_precision attrs for cudnn paths.
    use_tf32 = bool(cfg.get("tf32", True))
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*new API settings to control TF32.*")
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    fp32_precision = "tf32" if use_tf32 else "ieee"
    _set_nested_attr(torch.backends, ("cudnn", "fp32_precision"), fp32_precision)
    _set_nested_attr(
        torch.backends, ("cudnn", "conv", "fp32_precision"), fp32_precision
    )
    _set_nested_attr(torch.backends, ("cudnn", "rnn", "fp32_precision"), fp32_precision)

    env: Env = Env.initial()
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
        logger.info("Using shared-memory run_id=%s", run_id)
    precision = resolve_precision(cfg)
    dataset_mgr = DatasetManager(
        env=env,
        run_id=run_id,
        dataset=cfg.dataset,
        index_dtype=precision.index_np,
        time_dtype=precision.time_preprocessing_np,
        feature_dtype=precision.feature_torch,
        ignore_nfeat_on_disk=bool(cfg.get("ignore_nfeat_on_disk", False)),
    )
    (
        graph,
        df,
        n_nodes,
        n_edges,
        ndim,
        edim,
        ndtype,
        edtype,
        nfeat,
        efeat,
        port,
    ) = dataset_mgr.prepare()
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
    mail_recipients = int(cfg.mem_updater.get("mail_recipients", 0)) if has_mem else 0
    mail_combine = cfg.mem_updater.get("mail_combine") if has_mem else None
    if mail_recipients < 0:
        raise ValueError("mem_updater.mail_recipients must be >= 0.")
    if has_mem and mail_combine is None:
        raise ValueError("Memory models require mem_updater.mail_combine.")
    is_apan = str(cfg.encoder._target_) == APAN_ENCODER_TARGET

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
        fanout = _normalize_fanout(raw_fanout)
        sample_4_neg = bool(fanout)
        ngh_sampler_strategy = cfg.ngh_sampler.strategy
    time_scaling_factor = _resolve_time_scaling_factor(
        None if is_apan else cfg.ngh_sampler.get("time_scaling_factor", 0.0)
    )

    postman_enabled = mail_recipients > 0
    no_memory_queue_size = _resolve_no_memory_queue_size(
        cfg.get("no_memory_queue_size")
    )

    if len(fanout) > 1 and has_mem:
        raise ValueError("Memory models only support 0-hop or 1-hop fanout.")

    d_mail = 0
    if has_mem:
        d_mail = 2 * cfg.mem_updater.d_mem + cfg.edim

    worker_ranks = list(range(0, env.world_size - 1))
    nccl_ddp = dist.new_group(ranks=worker_ranks, backend="nccl")
    nccl_p2p = dist.new_group(ranks=worker_ranks, backend="nccl")
    idx_dtype = precision.index_torch
    
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
    max_n_neg = max(train_n_neg, eval_n_neg)

    ring_buf = RingBuf(
        run_id=run_id,
        env=env,
        ndtype=idx_dtype,
        edtype=idx_dtype,
        tsdtype=precision.time_preprocessing_torch,
        fanout=fanout,
        max_batch_size=cfg.batch_size,
        n_neg=max_n_neg,
        sample_4_neg=sample_4_neg,
        max_size=cfg.ring_buf_size,
    )

    shared = None
    if has_mem:
        mem_box_store = MemBoxStoreManager(
            env=env,
            mdtype=precision.memory_torch,
            tdtype=precision.time_preprocessing_torch,
            run_id=run_id,
            n_nodes=n_nodes,
            boxsize=cfg.mem_updater.boxsize,
            d_mem=cfg.mem_updater.d_mem,
            edim=cfg.edim,
        )
        shared = mem_box_store.prepare()

    if env.is_host:
        # Launch host process
        from frost.runtime.host._engine import Host
        from frost.runtime.host.analyzer import StaticAnalyzer
        from frost.shm._core import cleanup_all_shm, set_as_host

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

        handle_stale: bool = bool(cfg.get("handle_stale", False))
        host: Host | None = None
        host_failed = True
        try:
            host = Host(
                has_nfeat=has_nfeat,
                has_efeat=has_efeat,
                has_mem=has_mem,
                mail_same_plan_as_mem=False,
                df=df,
                env=env,
                start_time=start,
                planner=planner,
                batch_size=cfg.batch_size,
                neg_sampler_train=neg_sampler_train,
                neg_sampler_eval=neg_sampler_eval,
                ngh_sampler=ngh_sampler,
                time_dtype=precision.time_preprocessing_torch,
                n_epoch=cfg.epoch,
                patience=cfg.patience,
                gpu_topo=gpu_topo,
                handle_stale=handle_stale,
                show_tqdm=bool(cfg.get("tqdm", False)),
                no_memory_queue_size=no_memory_queue_size,
                output_dir=output_dir,
            )
            host.run()
            host_failed = False
        except BaseException:
            raise
        finally:
            if host is not None:
                host.shutdown(graceful=not host_failed)
            else:
                destroy_process_group(sync=False, logger_=logger)
            cleanup_all_shm(is_host=True)

    else:
        # Launch worker process
        from frost.runtime.worker._engine import Worker

        assert nccl_ddp is not None, "Worker rank must be in nccl group"
        assert nccl_p2p is not None, "Worker rank must be in nccl_p2p group"

        mem_updater_cfg = cfg.get("mem_updater")
        if mem_updater_cfg is not None:
            mem_updater_cfg = OmegaConf.to_container(mem_updater_cfg, resolve=True)
            assert isinstance(mem_updater_cfg, dict)
            mem_updater_cfg.pop("mail_recipients", None)
            mem_updater_cfg.pop("mail_combine", None)

        mem_updater: None | MemoryUpdaterProto = instantiate(
            mem_updater_cfg, d_mail=d_mail
        )
        encoder_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
        assert isinstance(encoder_cfg, dict)
        encoder_cfg.pop("out_dim", None)
        encoder_cfg.pop("fanout", None)
        if (
            str(cfg.encoder._target_) == DYGFORMER_ENCODER_TARGET
            and cfg.edim == 0
            and bool(encoder_cfg.get("enable_dummy_edge_features", True))
        ):
            encoder_cfg["dummy_edge_dim"] = _resolve_dygformer_dummy_edge_dim(
                dataset=cfg.dataset,
                requested_dim=encoder_cfg.get("dummy_edge_dim"),
            )
        if (
            str(cfg.encoder._target_) == GRAPH_MIXER_ENCODER_TARGET
            and bool(encoder_cfg.get("align_dyglib_zero_features", False))
            and encoder_cfg.get("dyglib_feature_dim") is None
        ):
            encoder_cfg["dyglib_feature_dim"] = DYGLIB_ALIGNED_FEATURE_DIM
            logger.info(
                "Resolved GraphMixer dyglib_feature_dim=%d for encoder=%s.",
                DYGLIB_ALIGNED_FEATURE_DIM,
                GRAPH_MIXER_ENCODER_TARGET,
            )
        if str(cfg.encoder._target_) == TGAT_ENCODER_TARGET:
            encoder: nn.Module = instantiate(encoder_cfg, fanout=fanout)
        else:
            encoder = instantiate(encoder_cfg)
        predictor_cfg = OmegaConf.to_container(cfg.predictor, resolve=True)
        assert isinstance(predictor_cfg, dict)
        predictor_cfg, predictor_switched = _resolve_predictor_cfg(
            str(cfg.encoder._target_),
            predictor_cfg,
        )
        if predictor_switched:
            logger.info(
                "Auto-switched predictor to DyGFormerLinkPredictor for pair-aware encoder=%s.",
                str(cfg.encoder._target_),
            )
        predictor: nn.Module = instantiate(predictor_cfg)
        postman: None | PostmanProto = (
            Postman(n_neg=max_n_neg) if postman_enabled else None
        )
        loss_fn: torch.nn.Module = instantiate(cfg.loss_fn)
        worker: Worker | None = None
        worker_failed = True
        try:
            worker = Worker(
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
                use_shm_instead=bool(cfg.get("__debug_use_shm_instead", False)),
                compile_model=cfg.get("compile", True),
                amp=bool(cfg.get("amp", False)),
                no_memory_queue_size=no_memory_queue_size,
            )
            worker.run()
            worker_failed = False
        except BaseException:
            raise
        finally:
            from frost.shm._core import cleanup_all_shm

            if worker is not None:
                worker.cleanup(graceful=not worker_failed)
            else:
                destroy_process_group(sync=False, logger_=logger)
            cleanup_all_shm(is_host=False)
