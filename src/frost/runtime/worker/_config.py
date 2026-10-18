"""Grouped construction inputs for the worker runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn

from frost.datatypes.env import Env
from frost.launch.plan import ExecutionPlan
from frost.models.protocols import MailboxRouterProto, MemoryUpdaterProto
from frost.runtime.experimental.dynamic_state import DynamicStatePolicy
from frost.shm.mem_mgr import MemBoxStore
from frost.shm.control_plane import SharedControlPlane
from frost.shm.dynamic_state_sync import SharedDynamicStateSync
from frost.shm.ring_buf import RingBuf


@dataclass(frozen=True)
class WorkerResources:
    dynamic_store: MemBoxStore | None
    node_features: torch.Tensor | None
    edge_features: torch.Tensor | None
    env: Env
    ddp_group: dist.ProcessGroup
    p2p_group: dist.ProcessGroup
    memory_updater: MemoryUpdaterProto | None
    encoder: nn.Module
    predictor: nn.Module
    ring_buffer: RingBuf
    control_plane: SharedControlPlane
    mailbox_router: MailboxRouterProto | None
    state_sync: SharedDynamicStateSync | None = None


@dataclass(frozen=True)
class WorkerOptions:
    max_negative_samples: int
    checkpoint_dir: str
    optimizer: Literal["adam", "sgd"]
    learning_rate: float
    mail_combine: Literal["last", "mean"] | None
    execution_plan: ExecutionPlan
    dynamic_state_policy: DynamicStatePolicy | None = None
    compile_model: bool = True
    compile_mode: str | None = None
    amp: bool = False
    no_memory_queue_size: int = 4
    cpu_benchmark_pid_list: list[int] | None = None
    eval_metric: Literal["ap", "mrr"] = "ap"
    train_only: bool = False
    single_gpu_resident: bool = False
    early_stop_tolerance: float = 0.0
