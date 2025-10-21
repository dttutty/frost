import argparse

# 允许 CUDA_VISIBLE_DEVICES 为空（CPU-only 或者容器里没设置）
# _raw_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
# cuda_visible_devices = [x for x in _raw_cvd.split(",") if x != ""] if _raw_cvd else []
# device_count = len(cuda_visible_devices)
# ---------- Torch / Dist ----------
import datetime
import logging
import os
import random
import socket
from functools import wraps
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array

from memorys import MemoryMailbox

# 把重依赖模块的导入尽量往后放，避免潜在循环导入风险
from modules import Config
from schemas.datatypes import DistEnv, SharedMailboxArrays
from utils import (
    get_model_save_path,
    get_node_count_from_graph,
    get_node_edge_dimensions,
    get_shape_and_dtype,
    load_feat,
    load_graph,
    parse_config,
)


# ---------- Logging ----------
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
log = logging.getLogger(__name__)


def get_free_port():
    """在 master 节点上获取一个空闲端口"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_broadcast(env, func=None, *, src_rank=0):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            payload = [None]
            if env.local_rank == src_rank:
                payload[0] = fn(*args, **kwargs)
            dist.broadcast_object_list(payload, src=src_rank)
            return payload[0]

        return wrapper

    return decorator if func is None else decorator(func)


def init_process_group(timeout_s: int = 3600, backend: str = "gloo"):
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend, timeout=datetime.timedelta(seconds=timeout_s)
        )


def create_memory_mailbox(cfg: Config, n, edim, local_rank) -> Optional[MemoryMailbox]:
    """
    根据配置初始化并返回一个共享的 MemoryMailbox。
    如果配置为 'none'，则返回 None。
    """
    if cfg.m.type == "none":
        return None

    # (init_shared 函数定义在这里或外面) ...
    def init_shared(name, shape, dtype):
        # ... (和之前一样) ...
        if local_rank == 0:
            arr = create_shared_mem_array(name, torch.Size(shape), dtype=dtype)
        dist.barrier()
        if local_rank != 0:
            arr = get_shared_mem_array(name, torch.Size(shape), dtype=dtype)
        return arr

    # 1. 定义配置 (和之前一样)
    shared_cfg = {
        "node_memory": ([n, cfg.m.dim_out], torch.float32),
        "node_memory_ts": ([n], torch.float32),
        "mails": (
            [n, cfg.m.mailbox_size, 2 * cfg.m.dim_out + edim],
            torch.float32,
        ),
        "mail_ts": ([n, cfg.m.mailbox_size], torch.float32),
        "next_mail_pos": ([n], torch.long),
        "mail_pos_update_pending_flag": ([n], torch.int8),
    }

    # 2. 【关键改动】
    #    不再创建字典，而是调用 dataclass 的工厂方法
    #    它内部完成了 create, barrier, zero, 和打包
    shared_arrays_obj = SharedMailboxArrays.create_from_config(
        shared_cfg, init_shared, local_rank
    )

    # 3. 传入强类型的 dataclass 实例，而不是字典
    return MemoryMailbox(cfg.m, shared_cfg, edim, shared_arrays_obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="dataset name")
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    parser.add_argument("--rand_edge_features", type=int, default=128)
    parser.add_argument("--rand_node_features", type=int, default=128)
    parser.add_argument("--omp_num_threads", type=int, default=8)
    parser.add_argument("--dedup", action="store_true", help="enable dedup")
    args = parser.parse_args()
    # if torch.cuda.device_count() > 0:
    #     # worker 进程此时只看到 1 张 ⇒ 设 0 就好；host 进程看到 0 张，会跳过
    #     torch.cuda.set_device(0)

    # 覆盖线程数（如有需要）
    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.omp_num_threads)

    # 进程组初始化（CPU后端，兼容 host）
    init_process_group(timeout_s=3600)

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # worker 会执行 (device_count=1)，host 会跳过 (device_count=0)
        torch.cuda.set_device(0)

    env = DistEnv.initialize()

    # world_size = GPU 工人数 + 1(host)
    # if device_count > 0:
    #     expected = device_count + 1
    #     if env.world_size != expected:
    #         raise RuntimeError(
    #             f"world_size({env.world_size}) != num_gpus({device_count}) + 1; "
    #             f"请用 torchrun --nproc_per_node={expected} 启动（单机假设）。"
    #         )
    # else:
    #     # CPU-only 调试模式：允许 world_size == 1（只有 host）
    #     if env.world_size != 1:
    #         log.warning(
    #             "未检测到 CUDA_VISIBLE_DEVICES，建议仅以 1 个进程运行作为 CPU 调试。"
    #         )

    # 对 GPU 工人绑定 device=0（因上面把可见设备约束为单张卡）
    # is_gpu_worker = device_count > 0 and (env.local_rank < device_count)
    # if is_gpu_worker and torch.cuda.is_available():
    #     torch.cuda.set_device(0)

    # 仅 GPU 工人参与 NCCL 通信；host 不加入
    # nccl_group = None
    # if device_count > 0:
    #     worker_ranks = list(range(max(0, env.world_size - 1)))
    #     if len(worker_ranks) > 0:
    #         nccl_group = dist.new_group(ranks=worker_ranks, backend="nccl")

    nccl_group = None
    # [修改] 使用 env.global_ngpus (在所有进程中值相同)
    if env.ngpus > 0:
        worker_ranks = list(range(env.ngpus))  # ngpus = global_ngpus
        # [注意] 所有进程 (包括 host) 都必须调用 new_group
        nccl_group = dist.new_group(ranks=worker_ranks, backend="nccl")

    # 包装需要在特定 rank 执行后广播结果的函数
    bget_shape_and_dtype = rank_broadcast(env, get_shape_and_dtype, src_rank=0)
    bget_node_edge_dimensions = rank_broadcast(
        env, get_node_edge_dimensions, src_rank=0
    )
    bget_model_save_path = rank_broadcast(env, get_model_save_path, src_rank=0)
    # 统计点数由 host(rank = world_size-1) 负责
    host_rank = env.world_size - 1
    bget_node_count_from_graph = rank_broadcast(
        env, get_node_count_from_graph, src_rank=host_rank
    )

    def init_shared_array(name, tensor_on_rank0: torch.Tensor | None):
        """
        Rank 0: create + copy; others: 等待 meta 后 get。
        返回当前进程可用的共享张量（或 None）。
        """
        arr_for_current_rank: torch.Tensor
        if env.local_rank == 0:
            if tensor_on_rank0 is not None:
                current_shape = tuple(tensor_on_rank0.shape)
                current_dtype = tensor_on_rank0.dtype
                arr_for_current_rank = create_shared_mem_array(
                    name, current_shape, dtype=current_dtype
                )
                arr_for_current_rank.copy_(tensor_on_rank0)
                meta_payload = {"shape": current_shape, "dtype": current_dtype}
            else:
                meta_payload = {"shape": None, "dtype": None}
        else:
            meta_payload = None

        # 广播 meta
        final_shape, final_dtype = bget_shape_and_dtype(meta_payload)

        if final_shape is None or final_dtype is None:
            return None

        # 稳一点：确保 rank0 已创建好
        dist.barrier()
        if env.local_rank != 0:
            arr_for_current_rank = get_shared_mem_array(
                name, final_shape, dtype=final_dtype
            )

        return arr_for_current_rank

    set_seed(args.seed)

    # 只有 rank0 读入特征，然后通过共享内存分发
    if env.local_rank == 0:
        node_np, edge_np = load_feat(
            args.data, args.rand_edge_features, args.rand_node_features
        )
    else:
        node_np = edge_np = None

    # 创建/获取共享特征
    efeat = init_shared_array("edge_feats", tensor_on_rank0=edge_np)
    nfeat = init_shared_array("node_feats", tensor_on_rank0=node_np)

    # 维度信息（由 rank0 计算并广播）
    _, ndim, _, _, edim, _ = bget_node_edge_dimensions(node_np, edge_np)

    config_base_name = os.path.basename(args.config)
    path_saver = bget_model_save_path(args.data, config_base_name)

    # host 进程加载图并持有
    if env.is_host:
        g, df = load_graph(args.data)
    else:
        g = df = None

    # 由 host 统计 num_nodes 并广播
    n = bget_node_count_from_graph(g)

    # 解析配置（每个进程各自做，避免传大对象）
    cfg = parse_config(args.config)
    cfg.t.dedup = bool(args.dedup)
    cfg.t.ndim = ndim
    cfg.t.edim = edim

    mm = create_memory_mailbox(cfg, n, edim, env.local_rank)

    from worker.gate import GlobalMicrobatchGate

    port_payload = [0]  # 用列表作为可变容器
    if env.rank == 0:
        port_payload[0] = get_free_port()
        log.info(f"Rank 0 found free port for gate: {port_payload[0]}")

    dist.broadcast_object_list(port_payload, src=0)
    gate_port = port_payload[0]
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    log.info(
        f"Rank {env.rank}: Initializing TCPStore for Gate at {master_addr}:{gate_port}"
    )

    try:
        store = dist.TCPStore(
            host_name=master_addr,
            port=gate_port,
            world_size=env.world_size,  # 确保 world_size 是总进程数
            is_master=(env.rank == 0),
            timeout=datetime.timedelta(seconds=300),  # 设置超时
        )
    except RuntimeError as e:
        log.error(f"Rank {env.rank}: Failed to init TCPStore for gate: {e}")
        raise e

    gate = GlobalMicrobatchGate(store=store, m_per_b=env.ngpus, gap=4)
    from run_host import HostTrainer
    from run_worker import run_worker

    # 分工：最后一个 rank 作为 host（采样+分发），其余为 GPU worker
    if env.is_host:
        trainer = HostTrainer(cfg, g, df, mm, env)
        trainer.train()
    else:
        run_worker(cfg, nccl_group, mm, path_saver, nfeat, efeat, env, gate)
