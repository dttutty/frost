import logging
import threading
from queue import Queue as Q
from typing import Literal

import torch
from dgl.heterograph import DGLBlock

from buffer import DyBuf
from schemas.alias import MetaWEvent
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.misc import MemNBox, Shared
from schemas.datatypes.plan import DyPlan
from thread_base_with_stream import StreamThreadBase


logger = logging.getLogger(__name__)


def _try_gather_to_pinned(
    src_data: torch.Tensor,
    src_ts: torch.Tensor,
    indices: torch.Tensor,
    pinned_buf: DyBuf,
    target_cls: type[MemTsNid] | type[BoxTsNid],
) -> MemTsNid | BoxTsNid | None:  # 返回 None 表示 Buffer 不够大，需要 fallback
    n = indices.numel()
    # [关键修复] 检查 Buffer 容量是否足够
    if n > pinned_buf.data.shape[0]:
        # Log a warning if needed, or just silently fallback
        return None

    # 1. 确定 Pinned Buffer 的有效区域视图
    buf_data_view = pinned_buf.data[:n]
    buf_ts_view = pinned_buf.ts[:n]
    buf_nid_view = pinned_buf.nid[:n]

    # 2. Gather Data (CPU -> Pinned CPU)
    torch.index_select(src_data, 0, indices, out=buf_data_view)

    # Handle Timestamp dimensions
    if src_ts.dim() == 1 and buf_ts_view.dim() == 2:
        torch.index_select(src_ts.unsqueeze(1), 0, indices, out=buf_ts_view)
    else:
        torch.index_select(src_ts, 0, indices, out=buf_ts_view)

    # Fill NID
    if indices.dim() == 1 and buf_nid_view.dim() == 2:
        buf_nid_view.copy_(indices.view(-1, 1))
    else:
        buf_nid_view.copy_(indices)

    # 3. Async Copy (Pinned CPU -> GPU)
    return target_cls(
        nid=indices.to("cuda", non_blocking=True),
        data=buf_data_view.to("cuda", non_blocking=True),
        ts=buf_ts_view.to("cuda", non_blocking=True),
    )


def prep_part_mem_n_box(
    shared: Shared,
    mem_plan: DyPlan,
    box_plan: DyPlan,
    pin_mem_buf: DyBuf | None = None,
    pin_box_buf: DyBuf | None = None,
) -> MemNBox:
    assert shared is not None
    assert mem_plan is not None

    # --- 处理 Memory ---
    mem_fetch_nids = mem_plan.fetch.data
    mem = None

    # 尝试走 Pin Memory 路径
    if pin_mem_buf is not None:
        mem = _try_gather_to_pinned(
            src_data=shared.mem,
            src_ts=shared.mem_ts,
            indices=mem_fetch_nids,
            pinned_buf=pin_mem_buf,
            target_cls=MemTsNid,
        )

    # 如果没开启 Buffer 或者 Buffer 不够大 (返回 None)，走普通路径
    if mem is None:
        mem = MemTsNid(
            nid=mem_fetch_nids,
            data=shared.mem[mem_fetch_nids],
            ts=shared.mem_ts[mem_fetch_nids],
        ).cuda()  # 普通阻塞传输

    # --- 处理 Box (Mail) ---
    mail_fetch_nids = box_plan.fetch.data
    box = None

    if pin_box_buf is not None:
        box = _try_gather_to_pinned(
            src_data=shared.box,
            src_ts=shared.box_ts,
            indices=mail_fetch_nids,
            pinned_buf=pin_box_buf,
            target_cls=BoxTsNid,
        )

    if box is None:
        box = BoxTsNid(
            nid=mail_fetch_nids,
            data=shared.box[mail_fetch_nids],
            ts=shared.box_ts[mail_fetch_nids],
        ).cuda()

    assert isinstance(mem, MemTsNid)
    assert isinstance(box, BoxTsNid)

    return MemNBox(mem=mem, box=box)


class DyAssembler(StreamThreadBase):
    """
    动态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    def __init__(
        self,
        in_q: Q[MetaWEvent | None | Literal["end_of_epoch"]],
        delta_q: Q[tuple[MemNBox, torch.cuda.Event] | Literal["nothing"]],
        full_q: Q[MetaWEvent | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
        has_reverse_mdt: bool = False,
    ) -> None:
        super().__init__(name="dy_assembler", stop_ev=stop_ev)
        self.in_q = in_q
        self.delta_q = delta_q
        self.full_q = full_q
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
                self.full_q.put(None, block=True)
                break
            elif in_data == "end_of_epoch":
                self.full_q.put("end_of_epoch", block=True)
                continue
            meta, meta_ready_event = in_data
            mem_n_box = meta.mem_n_box
            assert mem_n_box is not None
            torch.cuda.current_stream().wait_event(meta_ready_event)
            delta_tuple = self.delta_q.get(block=True)
            if delta_tuple == "nothing":
                mem_n_box_full = mem_n_box
            else:
                delta, delta_ready_event = delta_tuple
                torch.cuda.current_stream().wait_event(delta_ready_event)
                mem_n_box_full = self._merge(mem_n_box, delta)

            if meta.n2r_mfgs is not None:
                self.fill_block_with_mem_mail(
                    meta.n2r_mfgs[0], mem_n_box_full.mem, mem_n_box_full.box
                )
            else:
                # 这种是为了APAN哪种种没有mfg的情况
                meta.mem_n_box = mem_n_box_full.retain_and_align_to_root(meta.root_nid)
            # 3) 把结果交给主线程（主线程取时再 wait_stream）
            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            self.full_q.put((meta, ready_event), block=True)
