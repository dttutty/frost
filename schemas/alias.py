from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from tensor_set import TensorSet

from .datatypes.misc import Meta
from .datatypes.plan import DyPlan, PlanSet


MetaBundleWithPlan: TypeAlias = tuple[list[Meta], list[PlanSet]]
MetaBundleWithoutPlan: TypeAlias = tuple[list[Meta], None]
MetaBundle: TypeAlias = list[Meta]


PlanBundle: TypeAlias = list[DyPlan]
IntArr = NDArray[np.int64]
FloatArr = NDArray[np.floating]

MetaWEvent: TypeAlias = tuple[Meta, torch.cuda.Event]

MetasDyNeed: TypeAlias = tuple[list[TensorSet], list[TensorSet]]
MetasStNeed: TypeAlias = list[TensorSet]
