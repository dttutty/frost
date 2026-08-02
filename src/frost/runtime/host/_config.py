"""Grouped construction inputs for the host runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import torch

from frost.datatypes.env import Env
from frost.neg_sampler.protocols import NegSamplerProto
from frost.runtime.experimental.dynamic_state import DynamicStatePolicy
from frost.runtime.host.protocols import AnalyzerProto
from frost.sampling.protocols import NeighborhoodSamplerProto
from frost.shm.control_plane import SharedControlPlane


@dataclass(frozen=True)
class HostResources:
    dataframe: pd.DataFrame
    train_negative_sampler: NegSamplerProto
    eval_negative_sampler: NegSamplerProto
    neighborhood_sampler: NeighborhoodSamplerProto
    env: Env
    analyzer: AnalyzerProto
    gpu_topology: list
    control_plane: SharedControlPlane


@dataclass(frozen=True)
class HostOptions:
    has_node_features: bool
    has_dynamic_state: bool
    start_time: float
    batch_size: int
    eval_batch_size: int
    timestamp_dtype: torch.dtype
    epochs: int
    patience: int
    early_stop_tolerance: float = 0.0
    carry_over_route: bool = False
    show_progress: bool = False
    no_memory_queue_size: int = 4
    output_dir: str | None = None
    reset_eval_sampler_each_epoch: bool = False
    share_negative_sampler_across_splits: bool = False
    normalize_event_time: bool = False
    persist_all_updates_to_cpu: bool = False
    dynamic_state_policy: DynamicStatePolicy | None = None
    eval_metric: Literal["ap", "mrr"] = "ap"
    train_only: bool = False
