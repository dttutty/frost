import logging
import threading
from queue import Queue
from typing import Union, cast

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.protocols.cuda_event_proto import CudaEvent
from frost.runtime._thread_base import ThreadBase
from frost.shm.mem_mgr import MemBoxStore

logger = logging.getLogger(__name__)


class SharedMemWriter(ThreadBase):
    """Generic write-back thread for shared-memory tensors (node memory or mailbox)."""

    def __init__(
        self,
        shared_data: torch.Tensor,
        shared_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        wr_q: Queue,
        device: torch.device,
        name: str,
    ) -> None:
        super().__init__(name=name, stop_ev=stop_ev)
        self.device = device
        self.shared_data = shared_data
        self.shared_ts = shared_ts
        self.h2d_sem = h2d_sem
        self.wr_q = wr_q
        self.stop_ev = stop_ev
        self.writer_stream = cast(
            torch.cuda.Stream, torch.cuda.Stream(device=self.device)
        )

    def write_back(
        self,
        writeback: torch.Tensor | None,
        entry: Union[MemEntry, BoxEntry],
    ) -> None:
        if entry is None or writeback is None:
            return

        if writeback.device.type != "cuda":
            writeback = writeback.cuda()

        mask = torch.isin(entry.nid, writeback)
        valid_nid_gpu = entry.nid[mask]
        valid_data_gpu = entry.data[mask]
        valid_ts_gpu = entry.ts[mask]

        idx = valid_nid_gpu.cpu().long()
        self.shared_data[idx] = valid_data_gpu.cpu()
        self.shared_ts[idx] = valid_ts_gpu.cpu()

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            task = self.wr_q.get(block=True)
            if task is None:
                break

            data, batch_idx, entry, ready_event = task
            if data is not None and data.shape[0] > 0:
                torch.cuda.current_stream().wait_event(ready_event)

            self.write_back(data, entry)
            self.h2d_sem.release()


class BoxMailWriter(ThreadBase):
    """Write-back thread that inserts new mail into the per-node ring buffer.

    Unlike ``SharedMemWriter`` which overwrites entire rows, this writer
    calls ``MemBoxStore.write_mail()`` to insert a single mail per node
    into the circular buffer at ``box_pos``, then advances the pointer.

    Expects entries as ``BoxEntry`` with ``boxsize=1`` (i.e. the new mail
    wrapped via ``mail_to_box``).
    """

    def __init__(
        self,
        mem_box_store: MemBoxStore,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        wr_q: Queue,
        device: torch.device,
        name: str,
    ) -> None:
        super().__init__(name=name, stop_ev=stop_ev)
        self.device = device
        self.mem_box_store = mem_box_store
        self.h2d_sem = h2d_sem
        self.wr_q = wr_q
        self.stop_ev = stop_ev

    def write_back(
        self,
        writeback: torch.Tensor | None,
        entry: BoxEntry,
    ) -> None:
        if entry is None or writeback is None:
            return

        if writeback.device.type != "cuda":
            writeback = writeback.cuda()

        mask = torch.isin(entry.nid, writeback)
        valid_nid = entry.nid[mask]

        if valid_nid.numel() == 0:
            return

        # entry is BoxEntry with boxsize=1: data=[N,1,D], ts=[N,1]
        # Squeeze back to MailEntry format: data=[N,D], ts=[N]
        valid_data = entry.data[mask].squeeze(1)
        valid_ts = entry.ts[mask].squeeze(1)

        mail = MailEntry(
            nid=valid_nid,
            data=valid_data,
            ts=valid_ts,
            is_sorted=False,
            is_unique=True,
        )
        self.mem_box_store.write_mail(mail)

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            task = self.wr_q.get(block=True)
            if task is None:
                break

            data, batch_idx, entry, ready_event = task
            if data is not None and data.shape[0] > 0:
                torch.cuda.current_stream().wait_event(ready_event)

            self.write_back(data, entry)
            self.h2d_sem.release()
