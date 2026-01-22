from __future__ import annotations

from typing import List, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .datatypes.misc import Meta
from .datatypes.plan import DyPlan, PlanSet


MetaBundleWithPlan: TypeAlias = Tuple[List[Meta], List[PlanSet]]
MetaBundleWithoutPlan: TypeAlias = Tuple[List[Meta], None]
MetaBundle: TypeAlias = List[Meta]


PlanBundle: TypeAlias = List[DyPlan]
IntArr = NDArray[np.int64]
FloatArr = NDArray[np.floating]
