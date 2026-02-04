import logging
import threading
import time
from queue import Queue as Q
from typing import Any, Literal

import torch
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
            nid=indices.to(device, non_blocking=True),
            data=pinned_buf.data[:0].to(device, non_blocking=True),
            ts=pinned_buf.ts[:0].to(device, non_blocking=True),
        )

    # [关键修复] 检查 Buffer 容量是否足够
    if n > pinned_buf.data.shape[0]:
        return None

    # 1. 确定 Pinned Buffer 的有效区域视图
    buf_data_view = pinned_buf.data[:n]
    buf_ts_view = pinned_buf.ts[:n]
    buf_nid_view = pinned_buf.nid[:n]
    if __debug__:
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
        nid=indices.to(device, non_blocking=True),
        data=buf_data_view.to(device, non_blocking=True),
        ts=buf_ts_view.to(device, non_blocking=True),
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
        ).to(device)  # 普通阻塞传输

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
        self, gnn_mfgs_0: list[DGLBlock], mem: MemTsNid, box: BoxTsNid
    ) -> None:
        # assert torch.equal(mem.nid, box.nid)

        for idx, block in enumerate(gnn_mfgs_0):
            # 1. 确保有序
            if not getattr(mem, "is_sorted", False):
                mem = mem.sort_by_nid()
            if not getattr(box, "is_sorted", False):
                box = box.sort_by_nid()

            global_nid: torch.Tensor = block.srcdata["ID"]

            # --- 核心查找逻辑 ---

            # 查找插入位置
            permute_nidx = torch.searchsorted(mem.nid, global_nid)

            # 防止 searchsorted 返回 len(mem.nid) 导致的越界
            max_idx = mem.nid.numel() - 1

            # 异常情况 1: mem 为空但图需要节点
            if max_idx < 0:
                if global_nid.numel() > 0:
                    err_msg = f"DyAsm CRITICAL: Block {idx} requires {global_nid.numel()} nodes, but mem_n_box is EMPTY!"
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)
                continue

            # Clamp 是为了让下一步的“验证”能够执行，而不是为了混过去
            permute_nidx_clamped = permute_nidx.clamp(max=max_idx)

            # 验证：找到的位置上的 ID 是否真的等于我们要找的 ID
            # mask_found[i] == True 表示 global_nid[i] 存在于 mem.nid 中
            mask_found = mem.nid[permute_nidx_clamped] == global_nid

            # --- [修改点] 显式暴露错误 ---
            if not mask_found.all():
                # 提取缺失的节点 ID 用于调试
                missing_mask = ~mask_found
                missing_ids = global_nid[missing_mask]
                missing_count = missing_ids.numel()

                # 取前 10 个缺失的 ID 作为样本
                examples = missing_ids[:10].cpu().tolist()

                error_msg = (
                    f"\n[DyAsm Mismatch Error] Block {idx}:\n"
                    f"  > Requesting {global_nid.numel()} nodes for graph calculation.\n"
                    f"  > Found {missing_count} MISSING nodes in mem_n_box (which has {mem.nid.numel()} nodes).\n"
                    f"  > Examples of missing NIDs: {examples}\n"
                    f"  > Mem NID range: min={mem.nid.min()}, max={mem.nid.max()}\n"
                    f"  > Req NID range: min={global_nid.min()}, max={global_nid.max()}\n"
                    f"  This implies DyAssembler failed to fetch or merge these nodes."
                )

                logger.error(error_msg)
                # 打印到标准输出以防日志被截断
                print(error_msg)

                # 抛出 Python 异常，这将优雅地停止程序，而不是触发 CUDA Error
                raise RuntimeError(
                    f"DyAssembler: Missing {missing_count} nodes for Block {idx}"
                )

            # 如果能走到这里，说明所有节点都找到了，安全赋值
            if mem is not None:
                block.srcdata["mem"] = mem.data[permute_nidx_clamped]
                block.srcdata["mem_ts"] = mem.ts[permute_nidx_clamped]
            if box is not None:
                block.srcdata["box"] = box.data[permute_nidx_clamped]
                block.srcdata["box_ts"] = box.ts[permute_nidx_clamped]

    def _merge(
        self,
        mem_n_box: MemNBox,
        delta: MemNBox,
    ) -> MemNBox:
        mem = mem_n_box.mem.merge_mem(delta.mem, is_overlap=True)
        box = mem_n_box.box.merge_box(delta.box, is_overlap=True)

        mem = mem.sort_by_nid()
        box = box.sort_by_nid()

        return MemNBox(mem=mem, box=box)

    def logic(self) -> None:
        batch_count = 0
        epoch_count = 0

        while not self.stop_ev.is_set():
            batch_start = time.time()
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
            mem_n_box = meta.mem_n_box
            torch.cuda.current_stream().wait_event(meta_ready_event)
            # mem = mem_n_box.mem
            # box = mem_n_box.box
            # if torch.equal(mem.nid, box.nid) == False:
            #     print("Meta Error: MemTsNid and BoxTsNid NIDs do not match.")
            #     print("Meta Mem NIDs:", mem.nid)
            #     print("Meta Box NIDs:", box.nid)
            #     raise ValueError("Meta NID mismatch between MemTsNid and BoxTsNid.")

            assert mem_n_box is not None

            delta_tuple = self.delta_q.get(block=True)

            if delta_tuple == "nothing":
                mem_n_box.mem = mem_n_box.mem.sort_by_nid()
                mem_n_box.box = mem_n_box.box.sort_by_nid()
                mem_n_box_full = mem_n_box
            else:
                mem_p2p, box_p2p, works = delta_tuple
                for work in works:
                    work.wait()

                delta_mem = MemTsNid(
                    nid=mem_p2p.nid2recv, data=mem_p2p.data2recv, ts=mem_p2p.ts2recv
                )
                delta_mem = delta_mem.sort_by_nid()
                delta_box = BoxTsNid(
                    nid=box_p2p.nid2recv, data=box_p2p.data2recv, ts=box_p2p.ts2recv
                )
                delta_box = delta_box.sort_by_nid()

                delta = MemNBox(mem=delta_mem, box=delta_box)
                # delta, delta_ready_event = delta_tuple

                # torch.cuda.current_stream().wait_event(delta_ready_event)

                # if torch.equal(delta.mem.nid, delta.box.nid) == False:
                #     print("Delta  Error: MemTsNid and BoxTsNid NIDs do not match.")
                #     print(delta.mem.nid)
                #     print(delta.box.nid)
                #     raise ValueError(
                #         "Delta NID mismatch between MemTsNid and BoxTsNid."
                #     )
                mem_n_box_full = self._merge(mem_n_box, delta)

            # if self.boxsize == 1:
            # mem_n_box_full.box.data = mem_n_box_full.box.data.squeeze(1)

            if meta.n2r_mfgs is not None:
                self.fill_block_with_mem_box(
                    meta.n2r_mfgs[0], mem_n_box_full.mem, mem_n_box_full.box
                )
                meta.mem_n_box = mem_n_box_full
            else:
                # 这种是为了APAN哪种种没有mfg的情况
                meta.mem_n_box = mem_n_box_full.retain_and_align_to_root(meta.root_nid)

            # 3) 把结果交给主线程（主线程取时再 wait_stream）
            self._ready_event.record(torch.cuda.current_stream())
            torch.cuda.current_stream().synchronize()
            self.full_q.put((meta, self._ready_event), block=True)
