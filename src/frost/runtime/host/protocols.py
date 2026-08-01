"""Interfaces owned by the host-side planning runtime."""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import numpy as np

from frost.runtime.host.types import AccessPlanInputs, DynamicPlanInQ, DynamicPlanOutQ
from frost.datatypes.sampled_layout import SampledLayoutView
from frost.runtime._thread_base import ThreadBase
from frost.utils.tensor_set import TensorSet


@runtime_checkable
class AnalyzerProto(Protocol):
    """Access-analysis and planner factory contract consumed by ``Host``."""

    def analyze_batch(
        self,
        layouts: list[SampledLayoutView],
        *,
        include_node_features: bool,
    ) -> AccessPlanInputs: ...

    def can_share_memory_mailbox_planner(self) -> bool: ...

    def get_mem_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]: ...

    def get_mem_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]: ...

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        if_carry_over_route: bool,
        output_dir: str | None,
        persist_all_updates_to_cpu: bool = False,
        current_batch_only: bool = False,
    ) -> ThreadBase | None: ...

    def get_mail_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]: ...

    def get_mail_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]: ...

    def get_feat_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> list[TensorSet]: ...
