from typing import List, Tuple, TypeAlias

from schemas.datatypes import Meta
from schemas.plan import PlanSet


MetaBundleWithPlan: TypeAlias = Tuple[List[Meta], List[PlanSet]]


MetaBundleWithoutPlan: TypeAlias = Tuple[List[Meta], None]
