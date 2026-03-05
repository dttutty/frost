from __future__ import annotations

import logging
from queue import Queue
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from frost.datatypes.meta import Meta
from frost.datatypes.plan import DyPlan, PlanSet, StPlan
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


MataBatcherInQ: TypeAlias = Queue[Meta | None | Literal["end_of_epoch"]]
DyPlanOutQ: TypeAlias = Queue[list[DyPlan] | None]
StPlanOutQ: TypeAlias = Queue[list[StPlan] | None]

MetaBundle: TypeAlias = list[Meta]
MetaBundleQ: TypeAlias = Queue[MetaBundle | None | Literal["end_of_epoch"]]
MetaBundleWithPlan: TypeAlias = tuple[MetaBundle, list[PlanSet]]
MetaBundleWithoutPlan: TypeAlias = tuple[MetaBundle, None]

MetasDyNeed: TypeAlias = tuple[list[TensorSet], list[TensorSet], MetaBundle]
MetasStNeed: TypeAlias = list[TensorSet]


DyPlanInQ: TypeAlias = Queue[MetasDyNeed | None | Literal["end_of_epoch"]]
StPlanInQ: TypeAlias = Queue[MetasStNeed | None | Literal["end_of_epoch"]]

PlanBundle: TypeAlias = list[DyPlan]
IntArr = NDArray[np.int64]
FloatArr = NDArray[np.floating]
