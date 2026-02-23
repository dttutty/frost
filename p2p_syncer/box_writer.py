import logging
import threading
from queue import Queue as Q
from typing import TypeVar, cast

import torch

from buffer import Buffers, DyBuf
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.protocol.misc import CudaEventLike
from thread_base_with_stream import StreamThreadBase


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")


logger = logging.getLogger(__name__)


class BoxWriter(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers | None,
        shared_box: torch.Tensor,
        shared_box_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        box_writer_q: Q[tuple[torch.Tensor | None, BoxTsNid, CudaEventLike]],
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
        self.writer_stream = cast(
            torch.cuda.Stream, torch.cuda.Stream(device=self.device)
        )
        self.disable_pin = False

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
            return

        if n > pinned_buf.data.shape[0]:
            logger.warning(
                f"BoxWriter: Buffer insufficient (need {n}, have {pinned_buf.data.shape[0]}), using fallback path"
            )
            # Fallback
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_box[valid_nid] = valid_data_gpu.cpu()
            self.shared_box_ts[valid_nid] = valid_ts_gpu.cpu()
            return

        buf_data = pinned_buf.data[:n]
        buf_nid = pinned_buf.nid[:n]
        buf_ts = pinned_buf.ts[:n]

        # Async D2H
        # _write_back_via_pinned 内
        with torch.cuda.stream(self.writer_stream):
            buf_data.copy_(valid_data_gpu, non_blocking=True)
            buf_ts.copy_(valid_ts_gpu, non_blocking=True)
            buf_nid.copy_(valid_nid_gpu, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.writer_stream)

        done.synchronize()  # 只同步 writer 自己

        # Sync
        # Copy to Shared
        idx = buf_nid.long()
        self.shared_box[idx] = buf_data
        target = self.shared_box_ts[idx]
        self.shared_box_ts[idx] = buf_ts

    def write_back_box(
        self,
        writeback: torch.Tensor | None,
        box_ts_nid: BoxTsNid,
    ) -> None:
        # [Bugfix] 应该是 or None 才 return, 之前是 not None 就 return
        if box_ts_nid is None or writeback is None:
            return

        # 1. Move writeback to GPU for mask calculation
        if writeback.device.type != "cuda":
            writeback = writeback.cuda()

        # 2. Filter on GPU
        mask = torch.isin(box_ts_nid.nid, writeback)

        # # print datatype of shared_box_ts
        # logger.debug(
        #     f"shared_box_ts dtype: {self.shared_box_ts.dtype}, device: {self.shared_box_ts.device}"
        # )
        # # print datatype of valid_ts_gpu
        # logger.debug(
        #     f"valid_ts_gpu dtype: {box_ts_nid.ts.dtype}, device: {box_ts_nid.ts.device}"
        # )
        valid_nid_gpu = box_ts_nid.nid[mask]
        valid_data_gpu = box_ts_nid.data[mask]
        valid_ts_gpu = box_ts_nid.ts[mask]

        # 3. Use Pinned Memory D2H
        if (
            self.buf is not None
            and self.buf.box_d2h is not None
            and not self.disable_pin
        ):
            self._write_back_via_pinned(
                valid_nid_gpu, valid_data_gpu, valid_ts_gpu, self.buf.box_d2h
            )
        else:
            # Fallback
            id = valid_nid_gpu.cpu().long()
            self.shared_box[id] = valid_data_gpu.cpu()
            self.shared_box_ts[id] = valid_ts_gpu.cpu()

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            box_writer_q_task = self.box_writer_q.get(block=True)
            if box_writer_q_task is None:
                break

            data, box_ts_nid, ready_event = box_writer_q_task
            if data is not None and data.shape[0] > 0:
                torch.cuda.current_stream().wait_event(ready_event)

            self.write_back_box(data, box_ts_nid)
            # print box_ts_nid.nid
            # if data.shape[0] > 0:
            #     logger.debug("BoxWriter: Written back box for NIDs %s", data.shape)
            self.h2d_sem.release()
