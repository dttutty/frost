from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
from torch import Tensor

from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.misc import Meta

from .plan import DyPlan


@dataclass(slots=True)
class MemTask:
    macro_bid: int
    root_new_mem: MemTsNid
    old_mem_ts_nid: MemTsNid
    mem_plan: DyPlan
    is_last: bool
    meta: Meta | None = None


@dataclass(slots=True)
class BoxTask:
    root_nid: Tensor
    root_new_mem_orig: MemTsNid
    pos_efeat: Tensor | None
    macro_bid: int
    old_box: BoxTsNid
    box_plan: DyPlan
    is_last: bool
    r2n_mdt: Tensor | None
    meta: Meta | None = None

    def __repr__(self) -> str:
        # if data is tensor, show its shape
        return (
            f"BoxTask(macro_bid={self.macro_bid}, "
            f"root_nid_shape={self.root_nid.shape}, "
            f"root_new_mem_orig={self.root_new_mem_orig}, "
            f"pos_efeat_shape={self.pos_efeat.shape if self.pos_efeat is not None else None}, "
            f"old_box={self.old_box}, "
            f"box_plan={self.box_plan}, "
            f"is_last={self.is_last})"
        )


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

    def __repr__(self) -> str:
        # if data is tensor, show its shape
        return (
            f"P2PTask(nid2send_shape={self.nid2send.shape}, "
            f"nid2recv_shape={self.nid2recv.shape}, "
            f"data2send_shape={self.data2send.shape}, "
            f"data2recv_shape={self.data2recv.shape}, "
            f"ts2send_shape={self.ts2send.shape}, "
            f"ts2recv_shape={self.ts2recv.shape}, "
            f"rec_cts={self.rec_cts}, "
            f"send_cts={self.send_cts})"
        )
