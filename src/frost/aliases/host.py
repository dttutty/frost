from __future__ import annotations

import logging
from queue import Queue
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, PlanSet, StaticPlan
from frost.datatypes.sampled_layout import SampledLayoutView
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)

# SampledLayoutView travels as a sidecar alongside Meta in the host pipeline only.
# Workers never see it; the planner consumes it before Scatter sends Meta out.
MetaWithLayout: TypeAlias = tuple[Meta, SampledLayoutView]
MetaBatcherInQ: TypeAlias = Queue[MetaWithLayout | None | Literal["end_of_epoch"]]
DynamicPlanOutQ: TypeAlias = Queue[list[DynamicPlan] | None]
StaticPlanOutQ: TypeAlias = Queue[list[StaticPlan] | None]

MetaBundle: TypeAlias = list[Meta]
LayoutBundle: TypeAlias = list[SampledLayoutView]
MetaBundleQ: TypeAlias = Queue[tuple[MetaBundle, LayoutBundle] | None | Literal["end_of_epoch"]]
MetaBundleWithPlan: TypeAlias = tuple[MetaBundle, list[PlanSet]]
MetaBundleWithoutPlan: TypeAlias = tuple[MetaBundle, None]

MetasDyNeed: TypeAlias = tuple[list[TensorSet], list[TensorSet], MetaBundle]
MetasStNeed: TypeAlias = list[TensorSet]


DynamicPlanInQ: TypeAlias = Queue[MetasDyNeed | None | Literal["end_of_epoch"]]
StaticPlanInQ: TypeAlias = Queue[MetasStNeed | None | Literal["end_of_epoch"]]

PlanBundle: TypeAlias = list[DynamicPlan]
IntArr = NDArray[np.int64]
FloatArr = NDArray[np.floating]
