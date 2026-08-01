from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Literal, TypeAlias

from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, StaticPlan
from frost.datatypes.sampled_layout import SampledLayoutView
from frost.utils.tensor_set import TensorSet


DynamicPlanOutQ: TypeAlias = Queue[list[DynamicPlan] | None]
StaticPlanOutQ: TypeAlias = Queue[list[StaticPlan] | None]

MetaBundle: TypeAlias = list[Meta]
LayoutBundle: TypeAlias = list[SampledLayoutView]
MetasDyNeed: TypeAlias = tuple[list[TensorSet], list[TensorSet], list[int], MetaBundle]
MetasStNeed: TypeAlias = list[TensorSet]
DynamicPlanInQ: TypeAlias = Queue[MetasDyNeed | None | Literal["end_of_epoch"]]
StaticPlanInQ: TypeAlias = Queue[MetasStNeed | None | Literal["end_of_epoch"]]


@dataclass(frozen=True)
class DynamicAccessInputs:
    """Compact per-rank node sets consumed by one dynamic-state planner."""

    needs: list[TensorSet]
    updates: list[TensorSet]
    raw_need_counts: list[int]


@dataclass(frozen=True)
class AccessPlanInputs:
    """All access-plan inputs materialized from one sampled macro-batch."""

    memory: DynamicAccessInputs | None
    mailbox: DynamicAccessInputs | None
    node_features: list[TensorSet] | None
