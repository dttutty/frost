from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from frost.aliases.host import DynamicPlanInQ, DynamicPlanOutQ
from frost.datatypes.access import Access
from frost.runtime._thread_base import ThreadBase
from frost.utils.tensor_set import TensorSet


@runtime_checkable
class AnalyzerProto(Protocol):
    n_neg: int

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ) -> Access: ...

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
    ) -> ThreadBase | None: ...

    def get_mail_plan_inputs(
        self, access_list: list[Access]
    ) -> tuple[list[TensorSet], list[TensorSet]]: ...
