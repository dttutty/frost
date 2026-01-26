import logging
import threading
from queue import Queue as Q
from typing import Literal

import torch
from dgl.heterograph import DGLBlock

from schemas.alias import MetaWEvent
from schemas.datatypes.data_to_sync import BoxTsNid, MemTsNid
from schemas.datatypes.misc import MemNBox, Shared
from schemas.datatypes.plan import TensorDyPlan
from thread_base_with_stream import StreamThreadBase


logger = logging.getLogger(__name__)


def prep_part_mem_n_box(
    shared: Shared, mem_plan: TensorDyPlan, box_plan: TensorDyPlan
) -> MemNBox:
    # this is compatible with no mfg and mfg case
    # pre move part data to gpu, avoid h2d in assembler
    assert shared is not None
    assert mem_plan is not None
    mem_fetch_nids = mem_plan.fetch
    mem = MemTsNid(
        nid=mem_fetch_nids,
        data=shared.mem[mem_fetch_nids],
        ts=shared.mem_ts[mem_fetch_nids],
    ).cuda()
    mail_fetch_nids = box_plan.fetch
    box = BoxTsNid(
        nid=mail_fetch_nids,
        data=shared.box[mail_fetch_nids],
        ts=shared.box_ts[mail_fetch_nids],
    ).cuda()
    mem_n_box = MemNBox(mem=mem, box=box)
    return mem_n_box


class DyAssembler(StreamThreadBase):
    """
    动态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    def __init__(
        self,
        in_q: Q[MetaWEvent | None | Literal["end_of_epoch"]],
        delta_q: Q[tuple[MemNBox | None, torch.cuda.Event]],
        out_q: Q[MetaWEvent | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
        has_reverse_mdt: bool = False,
    ) -> None:
        super().__init__(name="dy_assembler")
        self.in_q = in_q
        self.delta_q = delta_q
        self.out_q = out_q
        self.has_reverse_mdt = has_reverse_mdt
        self.stop_ev = stop_ev

    def fill_block_with_mem_mail(
        self, gnn_mfgs_0: list[DGLBlock], mem: MemTsNid, box: BoxTsNid
    ) -> None:
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
        mem_n_box: MemNBox,
        delta: MemNBox,
    ) -> MemNBox:
        mem = mem_n_box.mem.merge_mem(delta.mem)
        box = mem_n_box.box.merge_box(delta.box)

        mem = mem.sort_by_nid()
        box = box.sort_by_nid()

        return MemNBox(mem=mem, box=box)

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            in_data = self.in_q.get(block=True)
            if in_data is None:
                self.out_q.put(None, block=True)
                break
            elif in_data == "end_of_epoch":
                delta, delta_ready_event = self.delta_q.get(block=True)
                self.out_q.put("end_of_epoch", block=True)
                continue
            meta, meta_ready_event = in_data
            mem_n_box = meta.mem_n_box
            assert mem_n_box is not None
            torch.cuda.current_stream().wait_event(meta_ready_event)
            delta, delta_ready_event = self.delta_q.get(block=True)
            torch.cuda.current_stream().wait_event(delta_ready_event)
            if delta is None:
                mem_n_box_full = mem_n_box
            else:
                mem_n_box_full = self._merge(mem_n_box, delta)

            if meta.gnn_mfgs is not None:
                self.fill_block_with_mem_mail(
                    meta.gnn_mfgs[0], mem_n_box_full.mem, mem_n_box_full.box
                )
            else:
                # 这种是为了APAN哪种种没有mfg的情况
                meta.mem_n_box = mem_n_box_full.unzip_and_align(meta.root_nid)
            # 3) 把结果交给主线程（主线程取时再 wait_stream）
            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            self.out_q.put((meta, ready_event), block=True)
