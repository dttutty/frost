import logging
import threading
from queue import Queue as Q
from typing import Literal, TypeVar

import torch

from buffer import Buffers, DyBuf
from schemas.datatypes.mem_mail_box._data_ops import merge_with_priority
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.task import MemTask, P2PTask
from thread_base_with_stream import StreamThreadBase

from .utils.data_ops import make_p2p_task


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


@torch.no_grad()
class MemHandler(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers,
        shared_mem: torch.Tensor,
        mem_task_q: Q[MemTask],
        mem_p2p_q: Q[P2PTask | Literal["nothing_to_p2p"]],
        shared_mem_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="mem_handler", stop_ev=stop_ev)
        self.buf = buf
        self.shared_mem = shared_mem
        self.shared_mem_ts = shared_mem_ts
        self.h2d_sem = h2d_sem
        self.mem_task_q = mem_task_q
        self.mem_p2p_q = mem_p2p_q
        self.stop_ev = stop_ev

    def _write_back_via_pinned(
        self,
        valid_nid_gpu: torch.Tensor,
        valid_data_gpu: torch.Tensor,
        valid_ts_gpu: torch.Tensor,
        pinned_buf: DyBuf,
    ) -> None:
        """
        利用 Pinned Memory 进行 D2H 回写优化
        """
        n = valid_nid_gpu.numel()

        # 1. 检查 Buffer 容量
        if n > pinned_buf.data.shape[0]:
            # Fallback 到普通模式
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_mem[valid_nid] = valid_data_gpu.cpu()
            self.shared_mem_ts[valid_nid] = valid_ts_gpu.cpu()
            return

        # 2. 获取 Buffer 视图
        buf_data = pinned_buf.data[:n]
        buf_nid = pinned_buf.nid[:n]
        buf_ts = pinned_buf.ts[:n]

        # 3. Async Copy: GPU -> Pinned CPU
        valid_data_gpu.to(buf_data, non_blocking=True)
        # TS 和 NID 在 GPU 上通常是 1D，Buffer 是 2D (N,1)
        valid_ts_gpu.to(buf_ts.squeeze(1), non_blocking=True)
        # NID 需要转成 float/long 对应类型，通常 Buffer 初始化时类型定了，这里简单 copy
        # 注意：shared_mem 索引需要 Long，但 buffer 可能是 Float，这里仅做传输中介
        valid_nid_gpu.to(dtype=buf_nid.dtype).to(buf_nid.squeeze(1), non_blocking=True)

        # 4. Synchronize: 等待传输完成
        # 因为下一步马上要写入 shared_mem (Pageable)，必须确保数据已在 Pinned 中
        torch.cuda.current_stream().synchronize()

        # 5. Copy: Pinned CPU -> Shared Memory (CPU)
        # 从 buffer 读取回来转成 long 作为索引
        idx = buf_nid.view(-1).long()
        self.shared_mem[idx] = buf_data
        self.shared_mem_ts[idx] = buf_ts.view(-1)

    def write_back_mem(
        self, writeback: torch.Tensor | None, mem_ts_nid: MemTsNid
    ) -> None:
        if mem_ts_nid is None or writeback is None:
            self.h2d_sem.release()
            return

        # 优化: 确保 writeback 在 GPU 上以加速 mask 计算
        if writeback.device.type != "cuda":
            writeback = writeback.cuda()

        # 在 GPU 上做 mask
        mask = torch.isin(mem_ts_nid.nid, writeback)

        # 获取需要写回的数据 (仍在 GPU)
        valid_nid_gpu = mem_ts_nid.nid[mask]
        valid_data_gpu = mem_ts_nid.data[mask]
        valid_ts_gpu = mem_ts_nid.ts[mask]

        if self.buf.mem_d2h is not None:
            self._write_back_via_pinned(
                valid_nid_gpu, valid_data_gpu, valid_ts_gpu, self.buf.mem_d2h
            )
        else:
            # 原始逻辑 Fallback
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_mem[valid_nid] = valid_data_gpu.cpu()
            self.shared_mem_ts[valid_nid] = valid_ts_gpu.cpu()

        self.h2d_sem.release()

    def logic(self) -> None:
        while self.stop_ev.is_set() is False:
            mem_task = self.mem_task_q.get(block=True)

            mem_plan = mem_task.mem_plan
            root_new_mem = mem_task.root_new_mem.keep_last()
            mem_ts_nid = merge_with_priority(root_new_mem, mem_task.old_mem_ts_nid)
            if mem_task.is_last is not None:
                p2p_task = make_p2p_task(
                    data_td_nid=mem_ts_nid,
                    plan=mem_plan,
                    strategy="last",
                )
                self.mem_p2p_q.put(p2p_task)
            else:
                self.mem_p2p_q.put("nothing_to_p2p")
                # skip p2p(), directly put None to p2p_q as sentinel

            self.write_back_mem(mem_plan.writeback, mem_ts_nid)
