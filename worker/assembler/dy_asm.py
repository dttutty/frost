import logging
import queue
import threading
from typing import List, Tuple

import torch
from dgl.heterograph import DGLBlock

from schemas.datatypes.data_to_sync import BoxTsNid, MemTsNid
from schemas.datatypes.misc import Meta, X
from schemas.datatypes.plan import TensorPlanSet
from thread_base_with_stream import StreamThreadBase


logger = logging.getLogger(__name__)


def prep_before_dy_asm(shared, plan_set: TensorPlanSet) -> tuple[MemTsNid, BoxTsNid]:
    # this is compatible with no mfg and mfg case
    # pre move part data to gpu, avoid h2d in assembler
    assert shared is not None
    assert plan_set.mem is not None
    mem_fetch_nids = plan_set.mem.fetch
    mem = MemTsNid(
        nid=mem_fetch_nids,
        data=shared.mem[mem_fetch_nids],
        ts=shared.mem_ts[mem_fetch_nids],
    ).cuda()
    if plan_set.box is not None:
        mail_fetch_nids = plan_set.box.fetch
    else:
        mail_fetch_nids = mem_fetch_nids
    box = BoxTsNid(
        nid=mail_fetch_nids,
        data=shared.box[mail_fetch_nids],
        ts=shared.box_ts[mail_fetch_nids],
    ).cuda()

    return mem, box


class DyAssembler(StreamThreadBase):
    """
    动态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    def __init__(
        self,
        xpart_q: queue.Queue[Meta],
        delta_q: queue.Queue[X],
        xfull_q: queue.Queue[Tuple[Meta, torch.cuda.Stream]],
        stop_ev: threading.Event,
        has_mfg=True,
        has_reverse_mdt=False,
    ):
        super().__init__(name="dy_assembler")
        self.xpart_q = xpart_q
        self.delta_q = delta_q
        self.xfull_q = xfull_q
        self.stop_ev = stop_ev
        self.stream = torch.cuda.Stream()
        self.has_mfg = has_mfg
        self.has_reverse_mdt = has_reverse_mdt
        self.stop_ev = stop_ev

    def fill_block_with_mem_mail(
        self, gnn_mfgs_0: List[DGLBlock], mem: MemTsNid, box: BoxTsNid
    ):
        for block in gnn_mfgs_0:
            assert mem.is_sorted  # make sure mem.id is sorted
            assert box.is_sorted  # make sure box.id is sorted
            assert torch.equal(mem.nid, box.nid), (
                "MemTsNid and BoxTsNid must have the same NIDs."
            )
            global_nid: torch.Tensor = block.srcdata["ID"]
            permute_nidx = torch.searchsorted(mem.nid, global_nid)

            if mem is not None:
                block.srcdata["mem"] = mem.data[permute_nidx]  # 这里放memory
                block.srcdata["mem_ts"] = mem.ts[permute_nidx]
            if box is not None:
                block.srcdata["box"] = box.data[permute_nidx]
                block.srcdata["box_ts"] = box.ts[permute_nidx]

    def _merge(
        self,
        meta: Meta,
        x_part: X,
        mem_delta: MemTsNid,
        box_delta: BoxTsNid,
    ):
        assert x_part.mem is not None
        assert x_part.box is not None

        x_part.mem = x_part.mem.merge_mem(mem_delta)
        x_part.box = x_part.box.merge_box(box_delta)

        x_full = x_part

        x_full.mem = x_full.mem.sort_by_nid()  # 你的 sort
        x_full.box = x_full.box.sort_by_nid()  # 你的 sort

        return meta, x_full

    def logic(self):
        while not self.stop_ev.is_set():
            meta, ready_event = self.xpart_q.get(block=True)
            x_part, plan_set = meta.x, meta.plan_set
            delta = self.delta_q.get(block=True)
            assert delta.mem is not None
            assert delta.box is not None

            torch.cuda.current_stream().wait_event(ready_event)

            meta, x_full = self._merge(meta, x_part, delta.mem, delta.box)

            assert x_full.mem is not None
            assert x_full.box is not None
            if meta.gnn_mfgs is not None:
                self.fill_block_with_mem_mail(meta.gnn_mfgs[0], x_full.mem, x_full.box)
            else:
                # 这种是为了APAN哪种种没有mfg的情况
                meta.x = x_full.unzip_and_align(meta.root_nid)
            # 3) 把结果交给主线程（主线程取时再 wait_stream）

            self.xfull_q.put((meta), block=True)
