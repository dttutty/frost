from __future__ import annotations

import logging
import os
import warnings

import torch
import torch.distributed as dist

from frost.datatypes.env import Env
from frost.launch.config import set_nested_attr


logger = logging.getLogger(__name__)


def initialize_process_group() -> tuple[int, int, bool]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="gloo")
    return rank, world_size, rank == world_size - 1


def configure_torch_backends(
    *, use_tf32: bool, stable_mode: bool, is_host: bool
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*new API settings to control TF32.*"
        )
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    fp32_precision = "tf32" if use_tf32 else "ieee"
    set_nested_attr(torch.backends, ("cudnn", "fp32_precision"), fp32_precision)
    set_nested_attr(torch.backends, ("cudnn", "conv", "fp32_precision"), fp32_precision)
    set_nested_attr(torch.backends, ("cudnn", "rnn", "fp32_precision"), fp32_precision)
    if stable_mode:
        torch.backends.cudnn.benchmark = False
        if is_host:
            logger.info(
                "Enabled stable mode. CUDA_DEVICE_MAX_CONNECTIONS=%s, "
                "CUBLAS_WORKSPACE_CONFIG=%s",
                os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
                os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            )


def gather_cpu_benchmark_pids(env: Env, *, enabled: bool) -> list[int] | None:
    if not enabled:
        return None
    gathered_pids: list[int | None] = [None] * env.world_size
    dist.all_gather_object(gathered_pids, os.getpid())
    return [int(pid) for pid in gathered_pids if pid is not None]


def create_worker_process_groups(
    env: Env,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    worker_ranks = list(range(env.host_rank))
    ddp = dist.new_group(ranks=worker_ranks, backend="nccl")
    p2p = dist.new_group(ranks=worker_ranks, backend="nccl")
    return ddp, p2p
