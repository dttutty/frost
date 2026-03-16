import logging
import threading
from queue import Queue
from typing import Union, cast

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
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
    wrapped via ``mail_to_box``), or for APAN mode, a tuple of
    ``(MailEntry, flat_nid_cpu)`` for CPU-side neighbor scatter.
    """

    def __init__(
        self,
        mem_box_store: MemBoxStore,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        wr_q: Queue,
        device: torch.device,
        name: str,
        n_neg: int = 1,
        fanout: list[int] | None = None,
    ) -> None:
        super().__init__(name=name, stop_ev=stop_ev)
        self.device = device
        self.mem_box_store = mem_box_store
        self.h2d_sem = h2d_sem
        self.wr_q = wr_q
        self.stop_ev = stop_ev
        self.n_neg = n_neg
        self.fanout = fanout[0] if fanout and len(fanout) > 0 else 0

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

    def write_back_apan(
        self,
        writeback: torch.Tensor | None,
        seed_mail: MailEntry,
        flat_nid_cpu: torch.Tensor,
    ) -> None:
        """CPU-side scatter of seed mails to neighbor ring buffers.

        Instead of transferring the postman-expanded mails back from GPU,
        this method takes the raw 2N seed mails (already on CPU) and
        scatters them to each seed node's 1-hop neighbors using the
        CPU-resident neighbor nids from ``flat_nid_cpu``.

        Deduplication: when multiple seed nodes write to the same neighbor,
        the entry with the latest timestamp wins (keep_last semantics).
        """
        if writeback is None:
            return

        # --- Extract neighbor section from flat_nid_cpu ---
        # flat layout: [root: n_src*(2+n_neg)] [ngh: n_src*2*fanout]
        n_total_mail = seed_mail.nid.shape[0]  # 2N (src mails + dst mails)
        n_pos = n_total_mail  # 2*n_src
        n_src = n_pos // 2
        fanout = self.fanout

        if fanout <= 0 or n_src <= 0:
            return

        ngh_start = n_src * (2 + self.n_neg)
        ngh_len = n_pos * fanout  # n_src * 2 * fanout
        ngh_nids = flat_nid_cpu[ngh_start: ngh_start + ngh_len]  # [n_pos * fanout]

        # Reshape to [n_pos, fanout]
        ngh_nids = ngh_nids.view(n_pos, fanout)
        valid_mask = ngh_nids != -1  # [n_pos, fanout]

        # --- Broadcast seed mail to neighbors ---
        D = seed_mail.data.shape[-1]

        # [n_pos, D] -> [n_pos, fanout, D]
        expanded_data = seed_mail.data.unsqueeze(1).expand(n_pos, fanout, D)
        # [n_pos] -> [n_pos, fanout]
        expanded_ts = seed_mail.ts.unsqueeze(1).expand(n_pos, fanout)

        # Apply mask -> [K, D] and [K]
        nbr_nids = ngh_nids[valid_mask]        # [K]
        nbr_data = expanded_data[valid_mask]    # [K, D]
        nbr_ts = expanded_ts[valid_mask]        # [K]

        if nbr_nids.numel() == 0:
            return

        # --- keep_last: deduplicate by nid, keeping latest ts ---
        # Sort by nid, then by ts (ascending) so last occurrence = latest ts
        sort_order = torch.argsort(
            nbr_nids * (nbr_ts.max().long() + 1) + nbr_ts.long(),
            stable=True,
        )
        nbr_nids = nbr_nids[sort_order]
        nbr_data = nbr_data[sort_order]
        nbr_ts = nbr_ts[sort_order]

        # After sorting by (nid, ts), the LAST occurrence per nid has the highest ts
        unique_nids, inverse = torch.unique_consecutive(nbr_nids, return_inverse=True)
        # Use scatter to keep the last index per unique nid
        last_idx = torch.zeros(unique_nids.shape[0], dtype=torch.long)
        idx_range = torch.arange(nbr_nids.shape[0])
        last_idx.scatter_(0, inverse, idx_range)  # last write wins = latest ts

        dedup_nids = nbr_nids[last_idx]
        dedup_data = nbr_data[last_idx]
        dedup_ts = nbr_ts[last_idx]

        # --- Filter by writeback set ---
        if writeback.device.type != "cpu":
            writeback = writeback.cpu()
        wb_mask = torch.isin(dedup_nids, writeback)
        final_nids = dedup_nids[wb_mask]
        final_data = dedup_data[wb_mask]
        final_ts = dedup_ts[wb_mask]

        if final_nids.numel() == 0:
            return

        mail = MailEntry(
            nid=final_nids,
            data=final_data,
            ts=final_ts,
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
            if data is not None and isinstance(data, torch.Tensor) and data.shape[0] > 0:
                torch.cuda.current_stream().wait_event(ready_event)

            # APAN fast path: entry is (MailEntry, Tensor) tuple
            if isinstance(entry, tuple):
                seed_mail, flat_nid_cpu = entry
                self.write_back_apan(data, seed_mail, flat_nid_cpu)
            else:
                self.write_back(data, entry)
            self.h2d_sem.release()
