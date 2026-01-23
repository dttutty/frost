from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from numpy.typing import NDArray
from torch import Tensor

from .data_to_sync import BoxTsNid, MemTsNid
from .plan import TensorDyPlan


@dataclass(slots=True)
class MemTask:
    macro_bid: int
    root_new_mem: MemTsNid
    old_mem_ts_nid: MemTsNid
    mem_plan: TensorDyPlan
    is_last: bool


@dataclass(slots=True)
class BoxTask:
    root_new_mem_orig: MemTsNid
    pos_efeat: Tensor | None
    macro_bid: int
    old_box: BoxTsNid
    box_plan: TensorDyPlan
    is_last: bool
    mail_delivery_tensor: Tensor | None


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
    mem_plan: TensorDyPlan
    box_plan: TensorDyPlan
    root_new_mem: MemTsNid
    pos_efeat: Tensor
    old_x: X
    mail_delivery_tensor: torch.Tensor | None
    is_last: bool
    ready_event: torch.cuda.Event
