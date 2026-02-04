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
from schemas.shm_layout import get_shm_layout_config
from shm_manager import DoubleBufferedSHM
from utils import load_feat, load_graph, parse_config


start = time.time()


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
            logger.error(f"Unknown model: {t_cfg.model}")
            raise ValueError(f"Unknown model: {t_cfg.model}")

    return ModelComponents(
        memory_updater=memory_updater,
        gnn=gnn,
        header=header,
    )


# 定义一个泛型变量 T
T = TypeVar("T")

# 建议做法 通过环境变量控制
# 在终端运行: export APP_ENV=dev && python main.py  -> 会显示所有日志
# 在终端运行: python main.py                       -> 只显示 Info 及以上

# 获取环境变量 默认为 'production'
env_str = os.getenv("APP_ENV", "production")

# 开发模式 什么都看 / 生产模式 只看 Info, Warning, Error
current_level = logging.DEBUG if env_str == "dev" else logging.INFO


def __init_logging(local_rank: int) -> logging.Logger:
    # 1. 只有第一次调用有效，所以确保这里包含所有参数
    root_logger = logging.getLogger()

    # 1. 强制重置所有配置 (Python 3.8+)
    logging.basicConfig(
        level=current_level,
        filename=f"rank_{local_rank}.log",
        filemode="w",
        format="%(asctime)s - %(name)s - [%(threadName)s] - %(levelname)s - %(message)s",
        force=True,  # <--- 关键：强制覆盖之前的任何配置
    )

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            # 关键：每次写入都强制刷新到磁盘
            handler.flush = (
                lambda h=handler, os=os: (
                    h.stream.flush(),
                    os.fsync(h.stream.fileno()),
                )[0]
                if hasattr(h.stream, "fileno")
                else h.stream.flush()
            )
            # 也可以通过设置缓冲为 0 来强制实时写入（需要重新创建 Handler）

    # 2. 只有 Rank 0 的信息才输出到终端，避免 4 个进程同时刷屏
    if local_rank == 0 or local_rank == 2:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # 终端格式可以简单点，方便阅读
        formatter = logging.Formatter("\033[32m%(levelname)s\033[0m - %(message)s")
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    # 3. 屏蔽掉第三方库（如 torch, dgl）过杂的 DEBUG 信息
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("dgl").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


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
local_rank = int(os.environ.get("LOCAL_RANK", 0))
logger = __init_logging(local_rank)
dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=3600))
env = Env.initial()
worker_ranks = list(range(0, env.world_size - 1))
all_ranks = list(range(0, env.world_size))
gloo = dist.new_group(ranks=all_ranks, backend="gloo")
assert gloo is not None, "Failed to create gloo process group"
nccl = dist.new_group(ranks=worker_ranks, backend="nccl")
nccl_p2p = dist.new_group(ranks=worker_ranks, backend="nccl")
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
if env.local_rank == 0:
    print(cfg)
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
    box = csm("mail", mail_shape, F32)
    box_ts = csm("mail_ts", [n_nodes, cfg.m.boxsize], F32)

dist.barrier()

if env.local_rank != 0 and cfg.m is not None:
    mem = gsm("mem", [n_nodes, cfg.m.d_mem], F32)
    mem_ts = gsm("mem_ts", [n_nodes], F32)
    box = gsm("mail", mail_shape, F32)
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


layout_config = get_shm_layout_config(
    batch_size=cfg.t.batch_size,
    fanouts=cfg.s.neighbor if cfg.s is not None else [10],
    history=cfg.s.history if cfg.s is not None else 1,
)
limits = {"nodes": 600000, "edges": 1200000}  # 根据 batch size 调整
shm_managers: list[DoubleBufferedSHM] = []
shm_manager: DoubleBufferedSHM | None = None
if env.is_host:
    for i in range(env.n_gpu):
        # Host 负责创建 (create=True)
        # 最后一个rank是host
        buffer_for_rank_i = DoubleBufferedSHM(
            worker_rank=i, create=True, layout=layout_config
        )
        shm_managers.append(buffer_for_rank_i)
    dist.barrier()
else:
    dist.barrier()
    shm_manager = DoubleBufferedSHM(
        worker_rank=env.local_rank, create=False, layout=layout_config
    )


if env.is_host:
    from run_host import Host

    assert shm_managers is not None
    host = Host(
        cfg=cfg,
        graph=graph,
        df=df,
        env=env,
        start_time=start,
        gloo=gloo,
        gnn=model_components.gnn,
        shm_managers=shm_managers,
    )
    host.run()
else:
    from run_worker import Worker

    assert shm_manager is not None

    worker = Worker(
        s_cfg=cfg.s,
        t_cfg=cfg.t,
        shared=shared,
        save_to=save_to,
        nfeat=nfeat,
        efeat=efeat,
        env=env,
        nccl=nccl,
        nccl_p2p=nccl_p2p,
        gloo=gloo,
        components=model_components,
        m_cfg=cfg.m,
        shm_manager=shm_manager,
    )
    worker.run()
