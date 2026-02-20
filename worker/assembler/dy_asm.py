import logging
import threading
import time
from queue import Queue as Q
from typing import Any, Literal

import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock

from buffer import DyBuf
from schemas.alias import MetaWEvent
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.misc import MemNBox, Shared
from schemas.datatypes.plan import DyPlan
from schemas.datatypes.task import P2PTask
from thread_base_with_stream import StreamThreadBase


logger = logging.getLogger(__name__)


def _try_gather_to_pinned(
    src_data: torch.Tensor,
    src_ts: torch.Tensor,
    indices: torch.Tensor,
    pinned_buf: DyBuf,
    target_cls: type[MemTsNid] | type[BoxTsNid],
    device: torch.device,
) -> MemTsNid | BoxTsNid | None:  # 返回 None 表示 Buffer 不够大，需要 fallback
    n = indices.numel()

    if n == 0:
        # 利用 pinned_buf 切出一个 0 大小的 view，这样能保留正确的 feature 维度和 dtype
        return target_cls(
            nid=indices.to(device, non_blocking=False),
            data=pinned_buf.data[:0].to(device, non_blocking=False),
            ts=pinned_buf.ts[:0].to(device, non_blocking=False),
        )

    # [关键修复] 检查 Buffer 容量是否足够
    if n > pinned_buf.data.shape[0]:
        return None

    # 1. 确定 Pinned Buffer 的有效区域视图
    buf_data_view = pinned_buf.data[:n]
    buf_ts_view = pinned_buf.ts[:n]
    buf_nid_view = pinned_buf.nid[:n]
    # if __debug__:
    # 边界检查：确保 indices 都在数据源的索引范围内
    if n > 0:  # 只有非空时才检查 max/min，防止 tensor 为空时调用 min() 报错
        max_idx = indices.max().item()
        min_idx = indices.min().item()
        assert max_idx < src_data.size(0), (
            f"Index {max_idx} exceeds src_data size {src_data.size(0)}"
        )
        assert min_idx >= 0, f"Negative index {min_idx} detected"

    # 维度匹配检查
    assert buf_data_view.size(0) == indices.numel(), (
        f"Buffer size mismatch: buf_data={buf_data_view.size(0)}, indices={indices.numel()}"
    )
    # 2. Gather Data (CPU -> Pinned CPU)
    torch.index_select(src_data, 0, indices, out=buf_data_view)

    # Handle Timestamp dimensions
    assert src_ts.dim() == buf_ts_view.dim()
    assert src_ts.dim() == 1 or src_ts.dim() == 2

    torch.index_select(src_ts, 0, indices, out=buf_ts_view)

    # Fill NID
    assert indices.dim() == buf_nid_view.dim()
    assert indices.dim() == 1
    buf_nid_view.copy_(indices)

    # 3. Async Copy (Pinned CPU -> GPU)

    return target_cls(
        nid=indices.to(device, non_blocking=False),
        data=buf_data_view.to(device, non_blocking=False),
        ts=buf_ts_view.to(device, non_blocking=False),
        is_sorted=True,
    )


def prep_part_mem_n_box(
    shared: Shared,
    mem_plan: DyPlan,
    box_plan: DyPlan,
    device: torch.device,
    pin_mem_buf: DyBuf | None = None,
    pin_box_buf: DyBuf | None = None,
) -> MemNBox:
    assert shared is not None
    assert mem_plan is not None

    # --- 处理 Memory ---
    mem_fetch_nids = mem_plan.fetch.data
    mem = None
    pin_mem_buf = None
    pin_box_buf = None
    # 尝试走 Pin Memory 路径
    if pin_mem_buf is not None:
        mem = _try_gather_to_pinned(
            src_data=shared.mem,
            src_ts=shared.mem_ts,
            indices=mem_fetch_nids,
            pinned_buf=pin_mem_buf,
            target_cls=MemTsNid,
            device=device,
        )
        if mem is not None:
            pass

    # 如果没开启 Buffer 或者 Buffer 不够大 (返回 None)，走普通路径
    if mem is None:
        mem = MemTsNid(
            nid=mem_fetch_nids,
            data=shared.mem[mem_fetch_nids],
            ts=shared.mem_ts[mem_fetch_nids],
        ).to(device)  # 普通阻塞传输

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
            device=device,
        )
        if box is not None:
            pass

    if box is None:
        box = BoxTsNid(
            nid=mail_fetch_nids,
            data=shared.box[mail_fetch_nids],
            ts=shared.box_ts[mail_fetch_nids],
            is_sorted=True,
        ).to(device)  # 普通阻塞传输

    assert isinstance(mem, MemTsNid)
    assert isinstance(box, BoxTsNid)

    assert torch.equal(mem.nid, box.nid), "Fetched Mem and Box NIDs do not match"

    return MemNBox(mem=mem, box=box)


class DyAssembler(StreamThreadBase):
    """
    动态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    def __init__(
        self,
        in_q: Q[MetaWEvent | None | Literal["end_of_epoch"]],
        delta_q: Q[tuple[P2PTask, P2PTask, list[Any]] | Literal["nothing"]],
        full_q: Q[MetaWEvent | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
        device: torch.device,
        has_reverse_mdt: bool = False,
        boxsize: int = 1,
    ) -> None:
        super().__init__(name="dy_assembler", stop_ev=stop_ev)
        self.in_q = in_q
        self.delta_q = delta_q
        self.full_q = full_q
        self.has_reverse_mdt = has_reverse_mdt
        self.device = device
        self.stop_ev = stop_ev
        self.boxsize = boxsize
        self._ready_event = torch.cuda.Event(
            enable_timing=False, blocking=False, interprocess=False
        )

    def fill_block_with_mem_box(
        self,
        gnn_mfgs_0: list[DGLBlock],
        sorted_mem: MemTsNid,
        sorted_box: BoxTsNid,
        is_box_and_mem_same: bool = True,
    ) -> None:
        for idx, block in enumerate(gnn_mfgs_0):
            global_nid = block.srcdata["ID"]

            if sorted_mem is None:
                raise RuntimeError("mem is None but block needs mem")  # 或者 continue

            if sorted_mem.nid.numel() == 0:
                if global_nid.numel() > 0:
                    raise RuntimeError(f"Block {idx} requires nodes but mem is empty")
                continue

            pos = torch.searchsorted(sorted_mem.nid, global_nid)
            if __debug__ and dist.get_rank() == 0:
                print("--- Debug Info ---")
                print("sorted_mem", sorted_mem.nid.cpu())
                print("global_nid", torch.unique(global_nid).cpu())

            # Always validate indices to avoid CUDA-side out-of-bounds.
            out_of_range = pos >= sorted_mem.nid.numel()
            if out_of_range.any():
                missing_ids = global_nid[out_of_range]
                examples = missing_ids[:10].detach().cpu().tolist()
                raise RuntimeError(
                    f"DyAssembler: Block {idx} missing {missing_ids.numel()} nodes. examples={examples}"
                )

            found = sorted_mem.nid[pos] == global_nid
            if not found.all():
                uniq_global_nid = torch.unique(global_nid)
                sorted_uniq_global_nid, uniq_indices = torch.unique(
                    uniq_global_nid, return_inverse=True
                )
                pos_uniq = torch.searchsorted(sorted_mem.nid, sorted_uniq_global_nid)
                found1 = sorted_mem.nid[pos_uniq] == sorted_uniq_global_nid
                if not found1.all():
                    missing_ids = sorted_uniq_global_nid[~found1]
                    examples = missing_ids[:10].detach().cpu().tolist()
                    raise RuntimeError(
                        f"DyAssembler: Block {idx} missing {missing_ids.numel()} unique nodes. examples={examples}"
                    )
                missing_ids = global_nid[~found]
                examples = missing_ids[:10].detach().cpu().tolist()
                raise RuntimeError(
                    f"DyAssembler: Block {idx} missing {missing_ids.numel()} nodes. examples={examples}"
                )

            block.srcdata["mem"] = sorted_mem.data[pos]
            block.srcdata["mem_ts"] = sorted_mem.ts[pos]

            if is_box_and_mem_same:
                # 若已 assert 一致，可复用 pos；否则就对 box 单独 searchsorted + found
                block.srcdata["box"] = sorted_box.data[pos]
                block.srcdata["box_ts"] = sorted_box.ts[pos]
            else:
                box_pos = torch.searchsorted(sorted_box.nid, global_nid)
                out_of_range = box_pos >= sorted_box.nid.numel()
                if out_of_range.any():
                    missing_ids = global_nid[out_of_range]
                    examples = missing_ids[:10].detach().cpu().tolist()
                    raise RuntimeError(
                        f"DyAssembler: Block {idx} missing {missing_ids.numel()} nodes in box. examples={examples}"
                    )

                box_found = sorted_box.nid[box_pos] == global_nid
                if not box_found.all():
                    missing_ids = global_nid[~box_found]
                    examples = missing_ids[:10].detach().cpu().tolist()
                    raise RuntimeError(
                        f"DyAssembler: Block {idx} missing {missing_ids.numel()} nodes in box. examples={examples}"
                    )
                block.srcdata["box"] = sorted_box.data[box_pos]
                block.srcdata["box_ts"] = sorted_box.ts[box_pos]

    def _merge(
        self,
        mem_n_box: MemNBox,
        delta: MemNBox,
    ) -> MemNBox:
        def _assert_no_duplicate_nid(nid: torch.Tensor, label: str) -> None:
            if nid.numel() == 0:
                return
            uniq, counts = torch.unique(nid, return_counts=True)
            if (counts > 1).any():
                dup_nid = uniq[(counts > 1).nonzero(as_tuple=True)[0]]
                examples = dup_nid[:10].cpu().tolist()
                raise RuntimeError(
                    f"DyAssembler: duplicate NIDs detected in {label}; "
                    f"total={nid.numel()} unique={uniq.numel()} examples={examples}"
                )

        if __debug__:
            # Guard against duplicate NIDs before/after merge.
            _assert_no_duplicate_nid(mem_n_box.mem.nid, "mem_n_box.mem")
            _assert_no_duplicate_nid(mem_n_box.box.nid, "mem_n_box.box")
            _assert_no_duplicate_nid(delta.mem.nid, "delta.mem")
            _assert_no_duplicate_nid(delta.box.nid, "delta.box")

        sorted_mem, sort_idx = mem_n_box.mem.merge_mem(delta.mem, is_overlap=True)
        sorted_box = mem_n_box.box.merge_box(
            delta.box, is_overlap=True, sort_idx=sort_idx
        )

        if __debug__:
            _assert_no_duplicate_nid(sorted_mem.nid, "merged.mem")
            _assert_no_duplicate_nid(sorted_box.nid, "merged.box")

        return MemNBox(mem=sorted_mem, box=sorted_box)

    def logic(self) -> None:
        batch_count = 0
        epoch_count = 0
        total_wait_ms = 0.0
        total_merge_ms = 0.0
        total_fill_ms = 0.0

        while not self.stop_ev.is_set():
            in_data = self.in_q.get(block=True)

            if in_data is None:
                self.full_q.put(None, block=True)
                break
            elif in_data == "end_of_epoch":
                epoch_count += 1
                self.full_q.put("end_of_epoch", block=True)
                continue

            batch_count += 1
            meta, meta_ready_event = in_data
            torch.cuda.current_stream().wait_event(meta_ready_event)
            mem_n_box = meta.mem_n_box
            assert mem_n_box is not None, (
                "Meta must contain mem_n_box before entering DyAssembler"
            )

            delta_tuple = self.delta_q.get(block=True)
            logger.info(
                f"DyAssembler received delta for batch {meta.batch_idx}, epoch {epoch_count}"
            )

            if delta_tuple == "nothing":
                t_merge_start = time.perf_counter()
                mem_n_box = mem_n_box
                total_merge_ms += (time.perf_counter() - t_merge_start) * 1000.0
            else:
                mem, box, works = delta_tuple
                t_wait_start = time.perf_counter()
                for work in works:
                    work.wait()
                total_wait_ms += (time.perf_counter() - t_wait_start) * 1000.0

                t_merge_start = time.perf_counter()
                delta_mem = MemTsNid(
                    nid=mem.nid2recv, data=mem.data2recv, ts=mem.ts2recv
                )
                delta_box = BoxTsNid(
                    nid=box.nid2recv, data=box.data2recv, ts=box.ts2recv
                )
                delta = MemNBox(mem=delta_mem, box=delta_box)
                mem_n_box = self._merge(mem_n_box, delta)

                total_merge_ms += (time.perf_counter() - t_merge_start) * 1000.0

            if self.boxsize == 1:
                mem_n_box.box.data = mem_n_box.box.data.squeeze(1)

            logger.info(
                f"DyAssembler after merge for batch {meta.batch_idx}, epoch {epoch_count}: "
            )
            if meta.n2r_mfgs is not None:
                t_fill_start = time.perf_counter()

                self.fill_block_with_mem_box(
                    meta.n2r_mfgs[0], mem_n_box.mem, mem_n_box.box
                )
                total_fill_ms += (time.perf_counter() - t_fill_start) * 1000.0
                meta.mem_n_box = mem_n_box
            else:
                # 这种是为了APAN哪种种没有mfg的情况
                meta.mem_n_box = mem_n_box.retain_and_align_to_root(meta.root_nid)

            # 3) 把结果交给主线程（主线程取时再 wait_stream）
            self._ready_event.record(torch.cuda.current_stream())
            # torch.cuda.current_stream().synchronize()
            self.full_q.put((meta, self._ready_event), block=True)
            logger.info(
                f"DyAssembler put full meta for batch {meta.batch_idx}, epoch {epoch_count} into full_q"
            )

            if batch_count % 100 == 0:
                avg_wait = total_wait_ms / max(1, batch_count)
                avg_merge = total_merge_ms / max(1, batch_count)
                avg_fill = total_fill_ms / max(1, batch_count)
                logger.info(
                    "DyAsm avg(ms) over %d batches: wait=%.2f merge=%.2f fill=%.2f",
                    batch_count,
                    avg_wait,
                    avg_merge,
                    avg_fill,
                )
