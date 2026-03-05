import logging
import threading
from typing import TypeVar, cast

import torch

from frost.aliases.syncer import MemWrQ
from frost.datatypes.mem_mail_box.box import BoxTsNid
from frost.datatypes.mem_mail_box.mail import MailTsNid
from frost.datatypes.mem_mail_box.mem import MemTsNid
from frost.runtime._thread_base import StreamThreadBase

AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


class MemWriter(StreamThreadBase):
    def __init__(
        self,
        shared_mem: torch.Tensor,
        shared_mem_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        mem_wr_q: MemWrQ,
        device: torch.device,
    ) -> None:
        super().__init__(name="mem_wr", stop_ev=stop_ev)
        self.device = device
        self.shared_mem = shared_mem
        self.shared_mem_ts = shared_mem_ts
        self.h2d_sem = h2d_sem
        self.mem_wr_q = mem_wr_q
        self.stop_ev = stop_ev
        self.writer_stream = cast(
            torch.cuda.Stream, torch.cuda.Stream(device=self.device)
        )

    def write_back_mem(
        self, writeback: torch.Tensor | None, mem_ts_nid: MemTsNid
    ) -> None:
        if mem_ts_nid is None or writeback is None:
            return

        # Ensure writeback tensor is on GPU for fast mask calculation
        if writeback.device.type != "cuda":
            writeback = writeback.cuda()

        # Compute mask on GPU
        mask = torch.isin(mem_ts_nid.nid, writeback)

        # Extract valid data to write back (still on GPU)
        valid_nid_gpu = mem_ts_nid.nid[mask]
        valid_data_gpu = mem_ts_nid.data[mask]
        valid_ts_gpu = mem_ts_nid.ts[mask]

        # Direct Device-to-Host (D2H) copy
        valid_nid = valid_nid_gpu.cpu().long()
        self.shared_mem[valid_nid] = valid_data_gpu.cpu()
        self.shared_mem_ts[valid_nid] = valid_ts_gpu.cpu()

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            mem_wr_q_task = self.mem_wr_q.get(block=True)
            if mem_wr_q_task is None:
                break

            data, batch_idx, mem_ts_nid, ready_event = mem_wr_q_task
            if data is not None:
                torch.cuda.current_stream().wait_event(ready_event)

            self.write_back_mem(data, mem_ts_nid)
            self.h2d_sem.release()
