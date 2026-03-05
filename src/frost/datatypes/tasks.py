from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
from torch import Tensor

from frost.datatypes.graph import Graph
from frost.datatypes.mem_mail_box.box import BoxTsNid
from frost.datatypes.mem_mail_box.mem import MemTsNid
from frost.datatypes.plan import DyPlan


@dataclass(slots=True)
class MemTask:
    batch_idx: int
    root_new_mem: MemTsNid
    mem_plan: DyPlan
    is_last: bool


@dataclass(slots=True)
class BoxTask:
    pos_efeat: Tensor | None
    root_new_mem: MemTsNid
    batch_idx: int
    box_plan: DyPlan
    is_last: bool
    old_box: BoxTsNid
    graph: Graph | None

    def __repr__(self) -> str:
        # if data is tensor, show its shape
        return (
            f"BoxTask(batch_idx={self.batch_idx}, "
            f"box_plan={self.box_plan}, "
            f"is_last={self.is_last}, "
            f"graph={self.graph})"
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
    batch_idx: int

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
            f"send_cts={self.send_cts}, "
            f"batch_idx={self.batch_idx})"
        )

    def clone(self) -> P2PTask:
        return P2PTask(
            nid2send=self.nid2send.clone(),
            nid2recv=self.nid2recv.clone(),
            data2send=self.data2send.clone(),
            data2recv=self.data2recv.clone(),
            ts2send=self.ts2send.clone(),
            ts2recv=self.ts2recv.clone(),
            rec_cts=self.rec_cts.copy(),
            send_cts=self.send_cts.copy(),
            batch_idx=self.batch_idx,
        )
