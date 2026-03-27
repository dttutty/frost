import faulthandler
import logging
import os
import time
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from frost.datatypes.env import Env
from frost.models.header import LinkPredictor
from frost.protocols.mem_updater_proto import MemoryUpdaterProto
from frost.protocols.ngh_sampler_proto import NghSamplerProto
from frost.protocols.postman_proto import PostmanProto
from frost.runtime._thread_base import destroy_process_group
from frost.runtime.host._plan_router import detect_gpu_topology
from frost.runtime.worker.postman import Postman
from frost.sampling.neg_sampler import NegLinkSampler
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
    dist.init_process_group(backend="gloo")

    # Control TF32 globally so experiments can isolate TF32 vs compile speedups.
    # PyTorch 2.9 deprecates the old allow_tf32 toggles in favor of fp32_precision.
    fp32_precision = "tf32" if bool(cfg.get("tf32", True)) else "ieee"
    _set_nested_attr(torch.backends, ("fp32_precision",), fp32_precision)
    _set_nested_attr(
        torch.backends, ("cuda", "matmul", "fp32_precision"), fp32_precision
    )
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
        ngh_sampler_strategy = cfg.mem_updater.get("postman_delivery_strategy")
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
        ngh_sampler_strategy = cfg.encoder.get("ngh_sampler_strategy")

    postman_enabled = mail_recipients > 0

    if len(fanout) > 1 and has_mem:
        raise ValueError("Memory models only support 0-hop or 1-hop fanout.")

    d_mail = 0
    if has_mem:
        d_mail = 2 * cfg.mem_updater.d_mem + cfg.edim

    worker_ranks = list(range(0, env.world_size - 1))
    nccl_ddp = dist.new_group(ranks=worker_ranks, backend="nccl")
    nccl_p2p = dist.new_group(ranks=worker_ranks, backend="nccl")
    idx_dtype = precision.index_torch
    ring_buf = RingBuf(
        run_id=run_id,
        env=env,
        ndtype=idx_dtype,
        edtype=idx_dtype,
        tsdtype=precision.time_preprocessing_torch,
        fanout=fanout,
        max_batch_size=cfg.batch_size,
        n_neg=cfg.n_neg,
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
        from frost.runtime.host.analyzer import NoMemAnalyzer
        from frost.shm._core import cleanup_all_shm, set_as_host

        set_as_host()
        start = time.time()
        if graph is None or df is None:
            raise RuntimeError("Host rank must load graph and dataframe before launch.")

        neg_sampler: NegLinkSampler = instantiate(
            cfg.neg_sampler,
            n_nodes=n_nodes,
        )
        ngh_sampler: NghSamplerProto
        if not fanout:
            ngh_sampler = NoSampler(
                graph=graph,
                ring_buf=ring_buf,
                n_neg=cfg.n_neg,
                sample_4_neg=False,
            )
        elif len(fanout) == 1:
            ngh_sampler = NghSampler(
                graph=graph,
                fanout=fanout,
                ring_buf=ring_buf,
                sample_4_neg=sample_4_neg,
                n_neg=cfg.n_neg,
                strategy=ngh_sampler_strategy,
                seed=cfg.seed,
            )
        else:
            ngh_sampler = MultiHopNghSampler(
                graph=graph,
                fanout=fanout,
                ring_buf=ring_buf,
                n_neg=cfg.n_neg,
                strategy=ngh_sampler_strategy,
                seed=cfg.seed,
                sample_4_neg=True,
            )
        planner_cfg = cfg.get("access_analyzer")
        if planner_cfg is None:
            if has_mem:
                raise ValueError(
                    "Memory models must configure access_analyzer explicitly."
                )
            planner = NoMemAnalyzer(n_neg=cfg.neg_sampler.n_neg)
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
                neg_sampler=neg_sampler,
                ngh_sampler=ngh_sampler,
                time_dtype=precision.time_preprocessing_torch,
                n_epoch=cfg.epoch,
                patience=cfg.patience,
                gpu_topo=gpu_topo,
                handle_stale=handle_stale,
                show_tqdm=bool(cfg.get("tqdm", False)),
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
            mem_updater_cfg.pop("postman_delivery_strategy", None)

        mem_updater: None | MemoryUpdaterProto = instantiate(
            mem_updater_cfg, d_mail=d_mail
        )
        encoder_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
        assert isinstance(encoder_cfg, dict)
        encoder_cfg.pop("out_dim", None)
        encoder_cfg.pop("fanout", None)
        encoder_cfg.pop("ngh_sampler_strategy", None)
        if str(cfg.encoder._target_) == TGAT_ENCODER_TARGET:
            encoder: nn.Module = instantiate(encoder_cfg, fanout=fanout)
        else:
            encoder = instantiate(encoder_cfg)
        predictor: LinkPredictor = instantiate(cfg.predictor)
        postman: None | PostmanProto = (
            Postman(n_neg=cfg.n_neg) if postman_enabled else None
        )
        loss_fn: torch.nn.Module = instantiate(cfg.loss_fn)
        worker: Worker | None = None
        worker_failed = True
        try:
            worker = Worker(
                n_neg=cfg.neg_sampler.n_neg,
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
