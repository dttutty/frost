import argparse
import datetime
import logging
import os
import random
import socket
import time
from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from dgl.utils.shared_mem import (
    create_shared_mem_array as csm,
    get_shared_mem_array as gsm,
)

from gnn import APAN, JODIE, TGAT, TGN, DyGFormer, DySAT_TGL
from gnn.components import EdgePredictor, TimeEncode
from gnn.with_memory.memory_updater import MfgMemoryUpdater, TransformerMemoryUpdater
from schemas.datatypes.cfg import GNNCFG, MemoryCFG, SampleCFG, TrainCFG
from schemas.datatypes.env import Env
from schemas.datatypes.misc import ModelComponents, Shared
from utils import load_feat, load_graph, parse_config


# 定义一个泛型变量 T
T = TypeVar("T")

# 建议做法 通过环境变量控制
# 在终端运行: export APP_ENV=dev && python main.py  -> 会显示所有日志
# 在终端运行: python main.py                       -> 只显示 Info 及以上

# 获取环境变量 默认为 'production'
env = os.getenv("APP_ENV", "production")

# 开发模式 什么都看 / 生产模式 只看 Info, Warning, Error
current_level = logging.DEBUG if env == "dev" else logging.INFO


logging.basicConfig(
    level=current_level,  # <--- 关键在这里
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

logger.debug("这是调试信息 - 只有 dev 模式能看到")
logger.info("这是普通信息 - 谁都能看到")
logger.error("这是错误 - 谁都能看到")

start = time.time()


# ---------- Logging ----------
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
log = logging.getLogger(__name__)


def get_free_port() -> int:
    """在 master 节点上获取一个空闲端口"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def broadcast_data(data: Any = None, src: int = 0) -> Any:  # noqa: ANN401
    payload = [data]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument(
        "--seed", action="store_const", const=42, default=None, help="set random seed"
    )
    parser.add_argument(
        "--rnd_edim", type=int, default=128
    )  # if your dataset has no edge features, set rnd_edim > 0 to use random edge features
    parser.add_argument(
        "--rnd_ndim", type=int, default=128
    )  # if your dataset has no node features, set rnd_ndim > 0 to use random node features
    # parser.add_argument(
    #     "--history", type=int, default=1
    # )  # number of history(snapshot) windows, used for sampling and gnn training
    # s_cfg = parser.add_argument_group("Sampler Settings")
    # s_cfg.add_argument(
    #     "--sample_neighbors", nargs="+", type=int
    # )  # a list of integers indicating how many neighbors are sampled in each layer, sample_layers can be calculted from sample_neighbors
    # s_cfg.add_argument(
    #     "--sample_strategy", type=str, default="recent", choices=["recent", "uniform"]
    # )  # 'recent' that samples most recent neighbors or 'uniform' that uniformly samples neighbors form the past
    # s_cfg.add_argument("--sample_prop_time", type=bool, default=False)
    # s_cfg.add_argument("--sample_duration", type=int, default=0)
    # s_cfg.add_argument("--sample_thread", type=int, default=32)
    # s_cfg.add_argument("--sample_neg_ratio", type=int, default=1)

    parser.add_argument("--omp_num_threads", type=int, default=8)
    parser.add_argument(
        "--model", type=str, choices=["apan", "tgn", "dysat", "jodie", "tgat"]
    )
    args = parser.parse_args()

    # match args.model:
    #     case "apan":
    # if args.sample_neighbors is not None:
    #     args.sample_leyer = len(args.sample_neighbors)
    #     assert args.sample_neg_ratio is not None
    #     assert args.sample_prop_time is not None
    # if args.history > 1:
    #     assert args.sample_duration > 0

    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.omp_num_threads)

    # initialize distributed environment
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=3600))
    env = Env.initial()
    logger.debug(f"env: {env}")
    worker_ranks = list(range(0, env.world_size - 1))
    all_ranks = list(range(0, env.world_size))
    logger.debug(f"Rank {env.local_rank}: Creating worker NCCL groups...")
    gloo = dist.new_group(ranks=all_ranks, backend="gloo")
    assert gloo is not None, "Failed to create gloo process group"
    nccl = dist.new_group(ranks=worker_ranks, backend="nccl")
    assert nccl is not None, "Failed to create nccl process group"
    if args.seed is not None:
        set_seed(args.seed)

    # host load graph and features, then broadcast to workers
    my_data = None
    if env.is_host:  # only host hold the graph and dataframe(edges)
        graph, df, n_nodes, n_edges = load_graph(args.dataset)
        edge_np, edim, edtype = load_feat(
            args.dataset,
            "edge",
            n_edges,
            args.rnd_edim,
        )
        node_np, ndim, ndtype = load_feat(
            args.dataset,
            "node",
            n_nodes,
            args.rnd_ndim,
        )
        if not os.path.isdir("models"):
            os.mkdir("models")
        config_base_name = os.path.basename(args.config)
        save_to = f"models/{args.dataset}_{config_base_name}_{time.time()}.pkl"
        if node_np is not None:
            nfeat = csm("nfeat", [n_nodes, ndim], dtype=node_np.dtype)
            nfeat.copy_(node_np)
        if edge_np is not None:
            efeat = csm("efeat", [n_edges, edim], dtype=edge_np.dtype)
            efeat.copy_(edge_np)
        port = get_free_port()
        my_data = (n_nodes, n_edges, ndim, edim, ndtype, edtype, save_to, port)
    dist.barrier()
    n_nodes, n_edges, ndim, edim, ndtype, edtype, save_to, port = broadcast_data(
        my_data, src=env.host_rank
    )

    if not env.is_host:
        nfeat = None
        efeat = None
        if ndim > 0:
            nfeat = gsm("nfeat", [n_nodes, ndim], dtype=ndtype)
        if edim > 0:
            efeat = gsm("efeat", [n_edges, edim], dtype=edtype)

    cfg = parse_config(args.config)
    cfg.t.ndim = ndim
    cfg.t.edim = edim
    cfg.g.d_in = cfg.m.d_mem if cfg.m is not None else ndim
    if cfg.g.d_out is None and cfg.m is not None:
        cfg.g.d_out = cfg.m.d_mem
    if cfg.s.sample_support_node:
        assert cfg.s.layer == len(cfg.s.neighbor)

    # initialize shared memory for memory and mailbox
    shared = None
    if cfg.m is not None:
        mail_shape = [n_nodes, cfg.m.boxsize, 2 * cfg.m.d_mem + edim]
    F32 = torch.float32
    # shared.box_next_pos should be int

    if env.local_rank == 0 and cfg.m is not None:
        mem = csm("mem", [n_nodes, cfg.m.d_mem], F32)
        mem_ts = csm("mem_ts", [n_nodes], F32)
        box = csm(
            "mail",
            mail_shape,
            F32,
        )
        box_ts = csm("mail_ts", [n_nodes, cfg.m.boxsize], F32)
    dist.barrier()

    if env.local_rank != 0 and cfg.m is not None:
        mem = gsm("mem", [n_nodes, cfg.m.d_mem], F32)
        mem_ts = gsm("mem_ts", [n_nodes], F32)
        box = gsm(
            "mail",
            mail_shape,
            F32,
        )
        box_ts = gsm("mail_ts", [n_nodes, cfg.m.boxsize], F32)
    shared = None
    if cfg.m is not None:
        shared = Shared(
            mem=mem,
            mem_ts=mem_ts,
            box=box,
            box_ts=box_ts,
        )

    dist.barrier()
    model_components = _init_gnn(cfg.t, cfg.s, cfg.g, cfg.m)

    if env.is_host:
        from run_host import Host

        logger.debug("Host: Initializing host process...")

        host = Host(cfg, graph, df, env, start, gloo, model_components.gnn)
        host.run()
    else:
        from run_worker import Worker

        logger.debug("Worker: Initializing worker process...")

        worker = Worker(
            s_cfg=cfg.s,
            t_cfg=cfg.t,
            shared=shared,
            save_to=save_to,
            nfeat=nfeat,
            efeat=efeat,
            env=env,
            nccl=nccl,
            gloo=gloo,
            components=model_components,
            m_cfg=cfg.m,
        )
        worker.run()


def _init_gnn(
    t_cfg: TrainCFG,
    s_cfg: SampleCFG,
    g_cfg: GNNCFG,
    m_cfg: MemoryCFG | None = None,
) -> ModelComponents:
    memory_updater = None
    gnn = None
    header = None
    d_mail = 0
    if m_cfg is not None:
        d_mail = 2 * m_cfg.d_mem + t_cfg.edim
    match t_cfg.model.lower():
        case "tgn":
            assert m_cfg is not None, "TGN requires memory config"
            memory_updater = MfgMemoryUpdater(
                rnn=torch.nn.GRUCell(d_mail + m_cfg.d_time, m_cfg.d_mem),
                time_encoder=TimeEncode(m_cfg.d_time),
            )
            assert g_cfg.d_out is not None, "TGN requires gnn.d_out to be set"
            gnn = TGN(
                dim_node=t_cfg.ndim,
                dim_edge=t_cfg.edim,
                dim_memory=m_cfg.d_mem,
                dim_time=m_cfg.d_time,
                dim_out=g_cfg.d_out,
            )
            header = EdgePredictor(g_cfg.d_out, neg_samples=1)

        case "apan":
            assert m_cfg is not None, "APAN requires memory config"
            assert m_cfg.att_h is not None, "APAN requires gnn.d_out to be set"

            memory_updater = TransformerMemoryUpdater(
                d_mail=d_mail,
                d_mem=t_cfg.ndim,
                d_time=m_cfg.d_time,
                boxsize=m_cfg.boxsize,
                heads=m_cfg.att_h,
                dropout=t_cfg.dropout,
                att_dropout=t_cfg.dropout,
            )
            gnn = APAN(
                dim_memory=m_cfg.d_mem,
                neg_samples=1,
            )
            header = EdgePredictor(m_cfg.d_mem, neg_samples=1)

        case "jodie":
            assert m_cfg is not None, "JODIE requires memory config"
            memory_updater = MfgMemoryUpdater(
                rnn=torch.nn.RNNCell(d_mail + m_cfg.d_time, m_cfg.d_mem),
                time_encoder=TimeEncode(m_cfg.d_time),
            )
            gnn = JODIE(
                dim_node=t_cfg.ndim,
                dim_memory=m_cfg.d_mem,
                dim_time=m_cfg.d_time,
            )
            header = EdgePredictor(m_cfg.d_mem, neg_samples=1)

        case "tgat":
            assert g_cfg.d_out is not None, "TGAT requires gnn.d_out to be set"
            assert g_cfg.att_h is not None, "TGAT requires gnn.att_h to be set"
            gnn = TGAT(
                dim_node=t_cfg.ndim,
                dim_edge=t_cfg.edim,
                dim_time=g_cfg.d_time,
                dim_hid=g_cfg.d_out,
                num_layers=s_cfg.layer,
                heads=g_cfg.att_h,
                dropout=t_cfg.dropout,
            )
            header = EdgePredictor(g_cfg.d_out, neg_samples=1)

        case "dysat":
            assert g_cfg.d_out is not None, "DySAT requires gnn.d_out to be set"
            assert g_cfg.att_h is not None, "DySAT requires gnn.att_h to be set"

            gnn = DySAT_TGL(
                dim_node=t_cfg.ndim,
                dim_edge=t_cfg.edim,
                dim_time=g_cfg.d_time,
                dim_hid=g_cfg.d_out,
                num_layers=s_cfg.layer,
                heads=g_cfg.att_h,
                dropout=t_cfg.dropout,
                history_len=s_cfg.history,
            )
            header = EdgePredictor(g_cfg.d_out, neg_samples=1)

        case "dygformer":
            gnn = DyGFormer(
                dim_node=t_cfg.ndim,
                dim_edge=t_cfg.edim,
            )

        case _:
            raise ValueError(f"Unknown model: {t_cfg.model}")

    return ModelComponents(
        memory_updater=memory_updater,
        gnn=gnn,
        header=header,
    )


if __name__ == "__main__":
    main()
