import logging
import threading
from queue import Queue as Q
from typing import TypeVar

import torch

from buffer import Buffers, DyBuf
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from thread_base_with_stream import StreamThreadBase


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


class MemWriter(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers,
        shared_mem: torch.Tensor,
        shared_mem_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        mem_writer_q: Q[tuple[torch.Tensor | None, MemTsNid, torch.cuda.Event]],
        device: torch.device,
    ) -> None:
        super().__init__(name="mem_writer", stop_ev=stop_ev)
        if __debug__:
            logger.debug("MemWriter: Initializing")
        self.buf = buf
        self.device = device
        self.shared_mem = shared_mem
        self.shared_mem_ts = shared_mem_ts
        self.h2d_sem = h2d_sem
        self.mem_writer_q = mem_writer_q
        self.stop_ev = stop_ev
        self.writer_stream = torch.cuda.Stream(device=self.device)
        if __debug__:
            logger.debug(
                f"MemWriter: shared_mem shape={shared_mem.shape}, shared_mem_ts shape={shared_mem_ts.shape}"
            )
        if __debug__:
            logger.debug(
                f"MemWriter: mem_d2h buffer={'enabled' if buf.mem_d2h is not None else 'disabled'}"
            )
        if __debug__:
            logger.debug("MemWriter: Initialized")

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
        if n == 0:
            if __debug__:
                logger.debug(
                    "MemWriter: No valid entries to write back, returning early"
                )
            return
        if __debug__:
            logger.debug(
                f"MemWriter: _write_back_via_pinned called, n={n}, buffer capacity={pinned_buf.data.shape[0]}"
            )

        # 1. 检查 Buffer 容量
        if n > pinned_buf.data.shape[0]:
            logger.warning(
                f"MemWriter: Buffer insufficient (need {n}, have {pinned_buf.data.shape[0]}), using fallback path"
            )
            # Fallback 到普通模式
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_mem[valid_nid] = valid_data_gpu.cpu()
            self.shared_mem_ts[valid_nid] = valid_ts_gpu.cpu()
            if __debug__:
                logger.debug("MemWriter: Fallback write-back complete")
            return

        if __debug__:
            logger.debug("MemWriter: Using pinned buffer for D2H transfer")
        # 2. 获取 Buffer 视图
        buf_data = pinned_buf.data[:n]
        buf_nid = pinned_buf.nid[:n]
        buf_ts = pinned_buf.ts[:n]

        # 3. Async Copy: GPU -> Pinned CPU
        if __debug__:
            logger.debug("MemWriter: Starting async D2H transfer")

        # _write_back_via_pinned 内
        with torch.cuda.stream(self.writer_stream):
            buf_data.copy_(valid_data_gpu, non_blocking=True)
            # TS 和 NID 在 GPU 上通常是 1D，Buffer 是 2D (N,1)
            buf_ts.copy_(valid_ts_gpu, non_blocking=True)
            # NID 需要转成 float/long 对应类型，通常 Buffer 初始化时类型定了，这里简单 copy
            # 注意：shared_mem 索引需要 Long，但 buffer 可能是 Float，这里仅做传输中介
            buf_nid.copy_(valid_nid_gpu, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.writer_stream)

        done.synchronize()  # 只同步 writer 自己

        # 4. Synchronize: 等待传输完成
        # 因为下一步马上要写入 shared_mem (Pageable)，必须确保数据已在 Pinned 中
        if __debug__:
            logger.debug("MemWriter: Synchronizing CUDA stream")

        # 5. Copy: Pinned CPU -> Shared Memory (CPU)
        # 从 buffer 读取回来转成 long 作为索引
        if __debug__:
            logger.debug("MemWriter: Copying to shared memory")
        idx = buf_nid.view(-1).long()
        self.shared_mem[idx] = buf_data
        self.shared_mem_ts[idx] = buf_ts.view(-1)
        if __debug__:
            logger.debug(
                f"MemWriter: Write-back complete via pinned buffer, wrote {n} entries"
            )

    def write_back_mem(
        self, writeback: torch.Tensor | None, mem_ts_nid: MemTsNid
    ) -> None:
        if __debug__:
            logger.debug(
                f"MemWriter: write_back_mem called, writeback={'provided' if writeback is not None else 'none'}, mem_ts_nid={'provided' if mem_ts_nid is not None else 'none'}"
            )

        if mem_ts_nid is None or writeback is None:
            if __debug__:
                logger.debug(
                    "MemWriter: No write-back needed (mem_ts_nid or writeback is None), releasing h2d_sem"
                )

            return

        if __debug__:
            logger.debug(
                f"MemWriter: writeback size={writeback.numel()}, mem_ts_nid size={mem_ts_nid.nid.numel()}"
            )

        # 优化: 确保 writeback 在 GPU 上以加速 mask 计算
        if writeback.device.type != "cuda":
            if __debug__:
                logger.debug("MemWriter: Moving writeback tensor to GPU")
            writeback = writeback.cuda()

        # 在 GPU 上做 mask
        if __debug__:
            logger.debug("MemWriter: Filtering valid entries on GPU")
        mask = torch.isin(mem_ts_nid.nid, writeback)

        # 获取需要写回的数据 (仍在 GPU)
        valid_nid_gpu = mem_ts_nid.nid[mask]
        valid_data_gpu = mem_ts_nid.data[mask]
        valid_ts_gpu = mem_ts_nid.ts[mask]

        if self.buf.mem_d2h is not None:
            if __debug__:
                logger.debug("MemWriter: Using pinned memory path for write-back")
            self._write_back_via_pinned(
                valid_nid_gpu, valid_data_gpu, valid_ts_gpu, self.buf.mem_d2h
            )
        else:
            if __debug__:
                logger.debug("MemWriter: Using regular (blocking) path for write-back")
            # 原始逻辑 Fallback
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_mem[valid_nid] = valid_data_gpu.cpu()
            self.shared_mem_ts[valid_nid] = valid_ts_gpu.cpu()

        if __debug__:
            logger.debug("MemWriter: Releasing h2d_sem")

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            mem_writer_q_task = self.mem_writer_q.get(block=True)
            if mem_writer_q_task is None:
                if __debug__:
                    logger.debug("MemWriter: received sentinel, exiting logic loop")
                break
            data, mem_ts_nid, ready_event = mem_writer_q_task
            if data is not None:
                torch.cuda.current_stream().wait_event(ready_event)

            self.write_back_mem(data, mem_ts_nid)
            self.h2d_sem.release()
