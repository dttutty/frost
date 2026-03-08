from __future__ import annotations

import logging
from queue import Queue
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from frost.datatypes.access import Access
from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, PlanSet, StaticPlan
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)

# Access travels as a sidecar alongside Meta in the Host pipeline only.
# Workers never see Access; it is consumed by the Planner before Scatter sends Meta out.
MetaWithAccess: TypeAlias = tuple[Meta, Access | None]
MetaBatcherInQ: TypeAlias = Queue[MetaWithAccess | None | Literal["end_of_epoch"]]
DynamicPlanOutQ: TypeAlias = Queue[list[DynamicPlan] | None]
StaticPlanOutQ: TypeAlias = Queue[list[StaticPlan] | None]

MetaBundle: TypeAlias = list[Meta]
AccessBundle: TypeAlias = list[Access | None]
MetaBundleQ: TypeAlias = Queue[tuple[MetaBundle, AccessBundle] | None | Literal["end_of_epoch"]]
MetaBundleWithPlan: TypeAlias = tuple[MetaBundle, list[PlanSet]]
MetaBundleWithoutPlan: TypeAlias = tuple[MetaBundle, None]

MetasDyNeed: TypeAlias = tuple[list[TensorSet], list[TensorSet], MetaBundle]
MetasStNeed: TypeAlias = list[TensorSet]


DynamicPlanInQ: TypeAlias = Queue[MetasDyNeed | None | Literal["end_of_epoch"]]
StaticPlanInQ: TypeAlias = Queue[MetasStNeed | None | Literal["end_of_epoch"]]

PlanBundle: TypeAlias = list[DynamicPlan]
IntArr = NDArray[np.int64]
FloatArr = NDArray[np.floating]
