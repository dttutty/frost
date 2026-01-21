from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from numpy.typing import NDArray
from torch import Tensor

from .data_to_sync import BoxTsNid, MemTsNid
from .misc import X
from .plan import TensorDyPlan, TensorPlanSet


@dataclass(slots=True)
class MemTask:
    root_new_mem: MemTsNid
    old_mem_ts_nid: MemTsNid
    plan: TensorDyPlan
    is_last: bool


@dataclass(slots=True)
class BoxTask:
    root_new_mem_orig: MemTsNid
    pos_efeat: Tensor
    macro_bid: int
    old_box: BoxTsNid
    plan: TensorDyPlan
    is_last: bool
    mail_delivery_tensor: Tensor


@dataclass(slots=True)
class P2PTask:
    nid: GatherElement
    data: GatherElement
    ts: GatherElement
    type: Literal["mem", "box", "mail"]


@dataclass(slots=True)
class GatherElement:
    recv: Tensor
    send: Tensor
    out_splits: NDArray
    in_splits: NDArray


@dataclass(slots=True, kw_only=True)
class SyncTask:
    macro_bid: int
    plan_set: TensorPlanSet
    root_new_mem: MemTsNid
    x: X
    mail_delivery_tensor: torch.Tensor | None
    is_last: bool
    ready_event: torch.cuda.Event
