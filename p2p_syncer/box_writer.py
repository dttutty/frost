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


class BoxWriter(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers,
        shared_box: torch.Tensor,
        shared_box_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        box_writer_q: Q[tuple[torch.Tensor | None, BoxTsNid, torch.cuda.Event]],
        device: torch.device,
    ) -> None:
        super().__init__(name="box_writer", stop_ev=stop_ev)

        self.buf = buf
        self.device = device
        self.shared_box = shared_box
        self.shared_box_ts = shared_box_ts
        self.h2d_sem = h2d_sem
        self.stop_ev = stop_ev
        self.box_writer_q = box_writer_q
        self.writer_stream = torch.cuda.Stream(device=self.device)

    @torch.no_grad()
    def _write_back_via_pinned(
        self,
        valid_nid_gpu: torch.Tensor,
        valid_data_gpu: torch.Tensor,
        valid_ts_gpu: torch.Tensor,
        pinned_buf: DyBuf,
    ) -> None:
        """D2H Pinned Memory Optimization"""
        n = valid_nid_gpu.numel()
        if n == 0:
            if __debug__:
                logger.debug("BoxWriter: No valid entries to write back, returning")
            return
        if __debug__:
            logger.debug(
                f"BoxWriter: _write_back_via_pinned called, n={n}, buffer capacity={pinned_buf.data.shape[0]}"
            )

        if n > pinned_buf.data.shape[0]:
            logger.warning(
                f"BoxWriter: Buffer insufficient (need {n}, have {pinned_buf.data.shape[0]}), using fallback path"
            )
            # Fallback
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_box[valid_nid] = valid_data_gpu.cpu()
            self.shared_box_ts[valid_nid] = valid_ts_gpu.cpu()
            if __debug__:
                logger.debug("BoxWriter: Fallback write-back complete")
            return

        if __debug__:
            logger.debug("BoxWriter: Using pinned buffer for D2H transfer")
        buf_data = pinned_buf.data[:n]
        buf_nid = pinned_buf.nid[:n]
        buf_ts = pinned_buf.ts[:n]

        # Async D2H
        if __debug__:
            logger.debug("BoxWriter: Starting async D2H transfer")

        # _write_back_via_pinned 内
        with torch.cuda.stream(self.writer_stream):
            buf_data.copy_(valid_data_gpu, non_blocking=True)
            buf_ts.copy_(valid_ts_gpu, non_blocking=True)
            buf_nid.copy_(valid_nid_gpu, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.writer_stream)

        done.synchronize()  # 只同步 writer 自己

        # Sync
        if __debug__:
            logger.debug("BoxWriter: Synchronizing CUDA stream")

        # Copy to Shared
        if __debug__:
            logger.debug("BoxWriter: Copying to shared memory")
        idx = buf_nid.long()
        self.shared_box[idx] = buf_data
        target = self.shared_box_ts[idx]
        self.shared_box_ts[idx] = buf_ts
        if __debug__:
            logger.debug(
                f"BoxWriter: Write-back complete via pinned buffer, wrote {n} entries"
            )

    def write_back_box(
        self,
        writeback: torch.Tensor | None,
        box_ts_nid: BoxTsNid,
    ) -> None:
        if __debug__:
            logger.debug(
                f"BoxWriter: write_back_box called, writeback={'provided' if writeback is not None else 'none'}, box_ts_nid={'provided' if box_ts_nid is not None else 'none'}"
            )

        # [Bugfix] 应该是 or None 才 return, 之前是 not None 就 return
        if box_ts_nid is None or writeback is None:
            if __debug__:
                logger.debug(
                    "BoxWriter: No write-back needed (box_ts_nid or writeback is None)"
                )
            return

        if __debug__:
            logger.debug(
                f"BoxWriter: writeback size={writeback.numel()}, box_ts_nid size={box_ts_nid.nid.numel()}"
            )

        # 1. Move writeback to GPU for mask calculation
        if writeback.device.type != "cuda":
            if __debug__:
                logger.debug("BoxWriter: Moving writeback tensor to GPU")
            writeback = writeback.cuda()

        # 2. Filter on GPU
        if __debug__:
            logger.debug("BoxWriter: Filtering valid entries on GPU")
        mask = torch.isin(box_ts_nid.nid, writeback)

        valid_nid_gpu = box_ts_nid.nid[mask]
        valid_data_gpu = box_ts_nid.data[mask]
        valid_ts_gpu = box_ts_nid.ts[mask]

        # 3. Use Pinned Memory D2H
        if self.buf.box_d2h is not None:
            if __debug__:
                logger.debug("BoxWriter: Using pinned memory path for write-back")
            self._write_back_via_pinned(
                valid_nid_gpu, valid_data_gpu, valid_ts_gpu, self.buf.box_d2h
            )
        else:
            if __debug__:
                logger.debug("BoxWriter: Using regular (blocking) path for write-back")
            # Fallback
            id = valid_nid_gpu.cpu().long()
            self.shared_box[id] = valid_data_gpu.cpu()
            self.shared_box_ts[id] = valid_ts_gpu.cpu()

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            box_writer_q_task = self.box_writer_q.get(block=True)
            if box_writer_q_task is None:
                if __debug__:
                    logger.debug("BoxWriter: received sentinel, exiting logic loop")
                break

            data, box_ts_nid, ready_event = box_writer_q_task
            if data is not None:
                torch.cuda.current_stream().wait_event(ready_event)

            self.write_back_box(data, box_ts_nid)
            self.h2d_sem.release()
