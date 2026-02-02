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
    if __debug__:
        logger.debug(f"Initializing GNN model: {t_cfg.model}")
    memory_updater = None
    gnn = None
    header = None
    d_mail = 0
    if m_cfg is not None:
        d_mail = 2 * m_cfg.d_mem + t_cfg.edim
        if __debug__:
            logger.debug(f"Memory config: d_mem={m_cfg.d_mem}, d_mail={d_mail}")

    match t_cfg.model.lower():
        case "tgn":
            if __debug__:
                logger.debug("Initializing TGN model...")
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
            if __debug__:
                logger.debug("TGN model initialized successfully")

        case "apan":
            if __debug__:
                logger.debug("Initializing APAN model...")
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
            if __debug__:
                logger.debug("APAN model initialized successfully")

        case "jodie":
            if __debug__:
                logger.debug("Initializing JODIE model...")
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
            if __debug__:
                logger.debug("JODIE model initialized successfully")

        case "tgat":
            if __debug__:
                logger.debug("Initializing TGAT model...")
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
            if __debug__:
                logger.debug("TGAT model initialized successfully")

        case "dysat":
            if __debug__:
                logger.debug("Initializing DySAT model...")
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
            if __debug__:
                logger.debug("DySAT model initialized successfully")

        case "dygformer":
            if __debug__:
                logger.debug("Initializing DyGFormer model...")
            gnn = DyGFormer(
                dim_node=t_cfg.ndim,
                dim_edge=t_cfg.edim,
            )
            if __debug__:
                logger.debug("DyGFormer model initialized successfully")

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
    if __debug__:
        logger.debug(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if __debug__:
        logger.debug("Random seed set successfully")


def broadcast_data(data: Any = None, src: int = 0) -> Any:  # noqa: ANN401
    if __debug__:
        logger.debug(f"Broadcasting data from rank {src}")
    payload = [data]
    dist.broadcast_object_list(payload, src=src)
    if __debug__:
        logger.debug(f"Data broadcast complete, received: {type(payload[0])}")
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
if __debug__:
    logger.debug(f"env: {env}")
worker_ranks = list(range(0, env.world_size - 1))
all_ranks = list(range(0, env.world_size))
if __debug__:
    logger.debug(f"Rank {env.local_rank}: Creating worker NCCL groups...")
gloo = dist.new_group(ranks=all_ranks, backend="gloo")
assert gloo is not None, "Failed to create gloo process group"
if __debug__:
    logger.debug(f"Rank {env.local_rank}: GLOO group created successfully")
nccl = dist.new_group(ranks=worker_ranks, backend="nccl")
nccl_p2p = dist.new_group(ranks=worker_ranks, backend="nccl")

assert nccl is not None, "Failed to create nccl process group"
if __debug__:
    logger.debug(f"Rank {env.local_rank}: NCCL group created successfully")

if args.seed is not None:
    set_seed(args.seed)

# host load graph and features, then broadcast to workers
my_data = None
if env.is_host:  # only host hold the graph and dataframe(edges)
    if __debug__:
        logger.debug("Host: Loading graph and features...")
    graph, df, n_nodes, n_edges = load_graph(args.dataset)
    if __debug__:
        logger.debug(f"Host: Loaded graph with {n_nodes} nodes and {n_edges} edges")

    edge_np, edim, edtype = load_feat(
        args.dataset,
        "edge",
        n_edges,
        args.rnd_edim,
    )
    if __debug__:
        logger.debug(f"Host: Loaded edge features, dim={edim}, dtype={edtype}")

    node_np, ndim, ndtype = load_feat(
        args.dataset,
        "node",
        n_nodes,
        args.rnd_ndim,
    )
    if __debug__:
        logger.debug(f"Host: Loaded node features, dim={ndim}, dtype={ndtype}")

    if not os.path.isdir("models"):
        os.mkdir("models")
        if __debug__:
            logger.debug("Created models directory")
    config_base_name = os.path.basename(args.config)
    save_to = f"models/{args.dataset}_{config_base_name}_{time.time()}.pkl"
    if __debug__:
        logger.debug(f"Model will be saved to: {save_to}")

    if node_np is not None:
        if __debug__:
            logger.debug("Host: Creating shared memory for node features...")
        nfeat = csm("nfeat", [n_nodes, ndim], dtype=node_np.dtype)
        nfeat.copy_(node_np)
        if __debug__:
            logger.debug("Host: Node features copied to shared memory")

    if edge_np is not None:
        if __debug__:
            logger.debug("Host: Creating shared memory for edge features...")
        efeat = csm("efeat", [n_edges, edim], dtype=edge_np.dtype)
        efeat.copy_(edge_np)
        if __debug__:
            logger.debug("Host: Edge features copied to shared memory")

    port = get_free_port()
    if __debug__:
        logger.debug(f"Host: Using port {port}")
    my_data = (n_nodes, n_edges, ndim, edim, ndtype, edtype, save_to, port)

if __debug__:
    logger.debug(f"Rank {env.local_rank}: Waiting at barrier before broadcast...")
dist.barrier()

n_nodes, n_edges, ndim, edim, ndtype, edtype, save_to, port = broadcast_data(
    my_data, src=env.host_rank
)
if __debug__:
    logger.debug(
        f"Rank {env.local_rank}: Received broadcast data - nodes={n_nodes}, edges={n_edges}, ndim={ndim}, edim={edim}"
    )

if not env.is_host:
    if __debug__:
        logger.debug("Worker: Accessing shared memory for features...")
    nfeat = None
    efeat = None
    if ndim > 0:
        nfeat = gsm("nfeat", [n_nodes, ndim], dtype=ndtype)
        if __debug__:
            logger.debug(
                f"Worker: Retrieved node features from shared memory, shape={nfeat.shape}"
            )
    if edim > 0:
        efeat = gsm("efeat", [n_edges, edim], dtype=edtype)
        if __debug__:
            logger.debug(
                f"Worker: Retrieved edge features from shared memory, shape={efeat.shape}"
            )

if __debug__:
    logger.debug(f"Parsing config from {args.config}")
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
if __debug__:
    logger.debug(
        f"Config parsed: model={cfg.t.model}, sample_layers={cfg.s.layer}, memory_enabled={cfg.m is not None}"
    )

# initialize shared memory for memory and mailbox
shared = None
if cfg.m is not None:
    mail_shape = [n_nodes, cfg.m.boxsize, 2 * cfg.m.d_mem + edim]
    if __debug__:
        logger.debug(
            f"Memory enabled: d_mem={cfg.m.d_mem}, boxsize={cfg.m.boxsize}, mail_shape={mail_shape}"
        )
F32 = torch.float32
# shared.box_next_pos should be int

if env.local_rank == 0 and cfg.m is not None:
    if __debug__:
        logger.debug("Rank 0: Creating shared memory for memory and mailbox...")
    mem = csm("mem", [n_nodes, cfg.m.d_mem], F32)
    mem_ts = csm("mem_ts", [n_nodes], F32)
    box = csm("mail", mail_shape, F32)
    box_ts = csm("mail_ts", [n_nodes, cfg.m.boxsize], F32)
    if __debug__:
        logger.debug("Rank 0: Shared memory created successfully")

if __debug__:
    logger.debug(
        f"Rank {env.local_rank}: Waiting at barrier after rank 0 creates shared memory..."
    )
dist.barrier()

if env.local_rank != 0 and cfg.m is not None:
    if __debug__:
        logger.debug(
            f"Rank {env.local_rank}: Accessing shared memory for memory and mailbox..."
        )
    mem = gsm("mem", [n_nodes, cfg.m.d_mem], F32)
    mem_ts = gsm("mem_ts", [n_nodes], F32)
    box = gsm("mail", mail_shape, F32)
    box_ts = gsm("mail_ts", [n_nodes, cfg.m.boxsize], F32)
    if __debug__:
        logger.debug(f"Rank {env.local_rank}: Shared memory accessed successfully")

shared = None
if cfg.m is not None:
    shared = Shared(
        mem=mem,
        mem_ts=mem_ts,
        box=box,
        box_ts=box_ts,
    )
    if __debug__:
        logger.debug(f"Rank {env.local_rank}: Shared memory wrapper created")

if __debug__:
    logger.debug(
        f"Rank {env.local_rank}: Waiting at barrier before model initialization..."
    )
dist.barrier()

if __debug__:
    logger.debug("Initializing model components...")
model_components = _init_gnn(cfg.t, cfg.s, cfg.g, cfg.m)
if __debug__:
    logger.debug(
        f"Model components initialized: gnn={type(model_components.gnn).__name__}, memory_updater={type(model_components.memory_updater).__name__ if model_components.memory_updater else None}"
    )


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

    if __debug__:
        logger.debug("Host: Initializing host process...")

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
    if __debug__:
        logger.debug("Host: Starting host process...")
    host.run()
else:
    from run_worker import Worker

    if __debug__:
        logger.debug("Worker: Initializing worker process...")

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
    if __debug__:
        logger.debug("Worker: Starting worker process...")
    worker.run()
