from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
from torch import Tensor

from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid

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
    r2n_mdt: Tensor | None


@dataclass(slots=True)
class P2PTask:
    nid2send: Tensor
    nid2recv: Tensor
    data2send: Tensor
    data2recv: Tensor
    ts2send: Tensor
    ts2recv: Tensor
    rec_cts: NDArray
    send_cts: NDArray
