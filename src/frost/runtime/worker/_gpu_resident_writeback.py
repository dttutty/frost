"""GPU-local memory/mailbox publication for the single-worker backend."""

from __future__ import annotations

import threading

import torch

from frost.datatypes.mem_mail_box._data_ops import compute_last_indices, keep_last
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.tasks import DirectWritebackRequest
from frost.runtime.worker._direct_writeback import _writer_done
from frost.runtime.worker._h2d_credit import CreditGate
from frost.shm.mem_mgr import MemBoxStore


class GpuResidentDynamicStateWriter:
    """Publish state directly into CUDA tensors without CPU or P2P traffic."""

    def __init__(
        self,
        *,
        mem_box_store: MemBoxStore,
        h2d_sem: CreditGate,
        mail_combine: str = "last",
    ) -> None:
        if mem_box_store.mem.device.type != "cuda":
            raise ValueError("GPU-resident writeback requires a CUDA MemBoxStore.")
        if mem_box_store.box.shape[1] != 1:
            raise ValueError("GPU-resident writeback currently requires boxsize=1.")
        self.mem_box_store = mem_box_store
        self.h2d_sem = h2d_sem
        if mail_combine not in {"last", "mean"}:
            raise ValueError(f"Unsupported resident mail_combine={mail_combine!r}.")
        self.mail_combine = mail_combine
        self._event_lock = threading.Lock()
        self._latest_event: torch.cuda.Event | None = None

    @property
    def device(self) -> torch.device:
        return self.mem_box_store.mem.device

    def wait_for_latest_write(self, stream: torch.cuda.Stream) -> None:
        """Order a reader stream after the most recently published batch."""
        with self._event_lock:
            event = self._latest_event
        if event is not None:
            stream.wait_event(event)

    def _filter_mem(
        self, entry: MemEntry | None, filter_nids: torch.Tensor | None
    ) -> MemEntry | None:
        if entry is None:
            return None
        entry = entry.to(self.device)
        if filter_nids is not None:
            keep = torch.isin(entry.nid, filter_nids.to(self.device))
            entry = MemEntry(
                nid=entry.nid[keep], data=entry.data[keep], ts=entry.ts[keep]
            )
        return keep_last(entry)

    def _filter_box(
        self, entry: BoxEntry | None, filter_nids: torch.Tensor | None
    ) -> BoxEntry | None:
        if entry is None:
            return None
        entry = entry.to(self.device)
        if filter_nids is not None:
            keep = torch.isin(entry.nid, filter_nids.to(self.device))
            entry = BoxEntry(
                nid=entry.nid[keep], data=entry.data[keep], ts=entry.ts[keep]
            )
        if entry.nid.numel() == 0:
            return entry
        if self.mail_combine == "mean":
            unique_nids, compact_idx = torch.unique(
                entry.nid, sorted=True, return_inverse=True
            )
            counts = torch.zeros(
                unique_nids.numel(), dtype=entry.ts.dtype, device=self.device
            )
            counts.scatter_add_(0, compact_idx, torch.ones_like(entry.ts[:, 0]))
            data = torch.zeros(
                (unique_nids.numel(), entry.data.shape[2]),
                dtype=entry.data.dtype,
                device=self.device,
            )
            data.scatter_add_(
                0,
                compact_idx[:, None].expand(-1, entry.data.shape[2]),
                entry.data[:, 0],
            )
            timestamps = torch.zeros_like(counts)
            timestamps.scatter_add_(0, compact_idx, entry.ts[:, 0])
            return BoxEntry(
                nid=unique_nids,
                data=(data / counts[:, None]).unsqueeze(1),
                ts=(timestamps / counts).unsqueeze(1),
                is_sorted=True,
                is_unique=True,
            )
        indices = compute_last_indices(entry.nid, entry.ts[:, 0])
        return BoxEntry(
            nid=entry.nid[indices],
            data=entry.data[indices],
            ts=entry.ts[indices],
            is_sorted=True,
            is_unique=True,
        )

    def _write_mem(
        self, entry: MemEntry | None, filter_nids: torch.Tensor | None
    ) -> None:
        entry = self._filter_mem(entry, filter_nids)
        if entry is None or entry.nid.numel() == 0:
            return
        nids = entry.nid.long()
        keep = entry.ts >= self.mem_box_store.mem_ts[nids]
        accepted = nids[keep]
        self.mem_box_store.mem[accepted] = entry.data[keep].to(
            self.mem_box_store.mem.dtype
        )
        self.mem_box_store.mem_ts[accepted] = entry.ts[keep].to(
            self.mem_box_store.mem_ts.dtype
        )

    def _write_box(
        self, entry: BoxEntry | None, filter_nids: torch.Tensor | None
    ) -> None:
        entry = self._filter_box(entry, filter_nids)
        if entry is None or entry.nid.numel() == 0:
            return
        nids = entry.nid.long()
        timestamps = entry.ts[:, 0]
        keep = timestamps >= self.mem_box_store.box_ts[nids, 0]
        accepted = nids[keep]
        self.mem_box_store.box[accepted, 0] = entry.data[keep, 0].to(
            self.mem_box_store.box.dtype
        )
        self.mem_box_store.box_ts[accepted, 0] = timestamps[keep].to(
            self.mem_box_store.box_ts.dtype
        )
        self.mem_box_store.box_pos[accepted] = 0

    @torch.no_grad()
    def write(
        self,
        request: DirectWritebackRequest,
        *,
        release_h2d_credits: bool,
    ) -> None:
        self._write_mem(request.mem_entry, request.mem_filter_nids)
        self._write_box(request.box_entry, request.box_filter_nids)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        with self._event_lock:
            self._latest_event = event
        if release_h2d_credits:
            _writer_done(self.h2d_sem, batch_idx=request.mini_batch_idx, kind="mem")
            _writer_done(self.h2d_sem, batch_idx=request.mini_batch_idx, kind="box")
