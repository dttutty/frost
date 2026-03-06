import logging
import threading
from queue import Queue as Q
from typing import TypeVar, cast

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.protocols.cuda_event_proto import CudaEvent
from frost.runtime._thread_base import StreamThreadBase

AllType = TypeVar("AllType", MemEntry, MailEntry, BoxEntry)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")


logger = logging.getLogger(__name__)


class BoxWriter(StreamThreadBase):
    def __init__(
        self,
        shared_box: torch.Tensor,
        shared_box_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        box_wr_q: Q[tuple[torch.Tensor | None, int, BoxEntry, CudaEvent] | None],
        device: torch.device,
    ) -> None:
        super().__init__(name="box_wr", stop_ev=stop_ev)
        self.device = device
        self.shared_box = shared_box
        self.shared_box_ts = shared_box_ts
        self.h2d_sem = h2d_sem
        self.stop_ev = stop_ev
        self.box_wr_q = box_wr_q
        self.writer_stream = cast(
            torch.cuda.Stream, torch.cuda.Stream(device=self.device)
        )

    def write_back_box(
        self,
        writeback: torch.Tensor | None,
        box_ts_nid: BoxEntry,
    ) -> None:
        if box_ts_nid is None or writeback is None:
            return

        # Move writeback to GPU for mask calculation
        if writeback.device.type != "cuda":
            writeback = writeback.cuda()

        # Filter on GPU
        mask = torch.isin(box_ts_nid.nid, writeback)

        valid_nid_gpu = box_ts_nid.nid[mask]
        valid_data_gpu = box_ts_nid.data[mask]
        valid_ts_gpu = box_ts_nid.ts[mask]

        # Direct Device-to-Host (D2H) copy
        idx = valid_nid_gpu.cpu().long()
        self.shared_box[idx] = valid_data_gpu.cpu()
        self.shared_box_ts[idx] = valid_ts_gpu.cpu()

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            box_wr_q_task = self.box_wr_q.get(block=True)
            if box_wr_q_task is None:
                break

            data, batch_idx, box_ts_nid, ready_event = box_wr_q_task
            if data is not None and data.shape[0] > 0:
                torch.cuda.current_stream().wait_event(ready_event)

            self.write_back_box(data, box_ts_nid)
            self.h2d_sem.release()
