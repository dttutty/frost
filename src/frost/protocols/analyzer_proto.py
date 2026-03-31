from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import numpy as np

from frost.aliases.host import DynamicPlanInQ, DynamicPlanOutQ
from frost.datatypes.sampled_layout import SampledLayoutView
from frost.runtime._thread_base import ThreadBase
from frost.utils.tensor_set import TensorSet


@runtime_checkable
class AnalyzerProto(Protocol):
    def get_mem_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]: ...

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
        output_dir: str | None,
    ) -> ThreadBase | None: ...

    def get_mail_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]: ...

    def get_feat_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> list[TensorSet]: ...
