import faulthandler
import time

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from frost.datatypes.env import Env
from frost.models.header import LinkPredictor
from frost.protocols.analyzer_proto import AnalyzerProto
from frost.protocols.mem_updater_proto import MemoryUpdaterProto
from frost.protocols.ngh_sampler_proto import NghSamplerProto
from frost.protocols.postman_proto import PostmanProto
from frost.sampling.neg_sampler import NegLinkSampler
from frost.shm.dataset_mgr import DatasetManager
from frost.shm.mem_mgr import MemBoxStoreManager
from frost.shm.ring_buf import RingBuf
from frost.utils._logger import init_logging
from frost.utils.utils import broadcast_data, set_seed

faulthandler.enable()


def launch(cfg: DictConfig):
    dist.init_process_group(backend="gloo")

    env: Env = Env.initial()
    init_logging(env.local_rank, env.is_host)
    run_id = broadcast_data(
        f"run_{int(time.time())}" if env.is_host else None, src=env.host_rank
    )
    run_id = ""
    dataset_mgr: DatasetManager = instantiate(cfg.dataset_mgr, env=env, run_id=run_id)
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
        save_to,
        port,
    ) = dataset_mgr.prepare()
    raw_fanout = OmegaConf.to_container(cfg.fanout, resolve=True)

    if raw_fanout is None:
        fanout = []
    elif isinstance(raw_fanout, int):
        fanout = [raw_fanout]
    else:
        fanout = list(raw_fanout)

    if env.is_host:
        with open_dict(cfg):
            cfg.ndim = ndim
            cfg.edim = edim
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        print(cfg_dict)

    d_mail = 0
    if cfg.get("mem_updater") is not None:
        d_mail = 2 * cfg.mem_updater.d_mem + cfg.edim

    worker_ranks = list(range(0, env.world_size - 1))
    nccl_ddp = dist.new_group(ranks=worker_ranks, backend="nccl")
    nccl_p2p = dist.new_group(ranks=worker_ranks, backend="nccl")
    idx_dtype = torch.int32
    ring_buf: RingBuf = instantiate(
        cfg.ring_buf,
        run_id=run_id,
        env=env,
        ndtype=idx_dtype,
        edtype=idx_dtype,
        fanout=fanout,
    )
    if cfg.seed is not None:
        set_seed(cfg.seed)

    shared = None
    if cfg.get("mem_box_store") is not None:
        mem_box_store: MemBoxStoreManager = instantiate(
            cfg.mem_box_store,
            env=env,
            mdtype=torch.float32,
            run_id=run_id,
            n_nodes=n_nodes,
        )
        shared = mem_box_store.prepare()

    if env.is_host:
        # Launch host process
        from frost.runtime.host._engine import Host
        from frost.shm._core import cleanup_all_shm, set_as_host

        set_as_host()
        start = time.time()

        neg_sampler: NegLinkSampler = instantiate(
            cfg.neg_sampler,
            n_nodes=n_nodes,
        )
        ngh_sampler: NghSamplerProto = instantiate(
            cfg.get("ngh_sampler"),
            graph=graph,
            ring_buf=ring_buf,
            fanout=fanout,
        )
        analyzer: AnalyzerProto = instantiate(cfg.get("analyzer"))
        assert isinstance(df, pd.DataFrame)
        has_nfeat = ndim > 0
        has_efeat = edim > 0

        handle_stale: bool = bool(cfg.get("handle_stale", False))
        host = Host(
            has_nfeat=has_nfeat,
            has_efeat=has_efeat,
            has_mem=cfg.get("mem_updater") is not None,
            mail_same_plan_as_mem=False,
            df=df,
            env=env,
            start_time=start,
            analyzer=analyzer,
            batch_size=cfg.batch_size,
            neg_sampler=neg_sampler,
            ngh_sampler=ngh_sampler,
            n_epoch=cfg.epoch,
            patience=cfg.patience,
            gpu_topo=OmegaConf.to_container(cfg.gpu_topo, resolve=True),
            handle_stale=handle_stale,
        )
        try:
            host.run()
        finally:
            cleanup_all_shm(is_host=True)

    else:
        # Launch worker process
        from frost.runtime.worker._engine import Worker

        assert nccl_ddp is not None, "Worker rank must be in nccl group"
        assert nccl_p2p is not None, "Worker rank must be in nccl_p2p group"

        mem_updater: None | MemoryUpdaterProto = instantiate(
            cfg.get("mem_updater"), d_mail=d_mail
        )
        encoder: nn.Module = instantiate(cfg.encoder)
        predictor: LinkPredictor = instantiate(cfg.predictor)
        postman: None | PostmanProto = instantiate(cfg.get("postman"))
        loss_fn: torch.nn.Module = instantiate(cfg.loss_fn)
        worker = Worker(
            n_neg=cfg.neg_sampler.n_neg,
            mem_box_store=shared,
            ring_buf=ring_buf,
            save_to=save_to,
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
            mail_combine=cfg.get("mail_combine"),
        )
        try:
            worker.run()
        finally:
            from frost.shm._core import cleanup_all_shm

            cleanup_all_shm(is_host=False)
