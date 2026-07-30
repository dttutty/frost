from __future__ import annotations

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
