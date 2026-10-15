from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.tasks import DirectWritebackRequest
from frost.runtime.worker._h2d_credit import CreditGate
from frost.shm.mem_mgr import MemBoxStore

if TYPE_CHECKING:
    from frost.runtime.worker._direct_writeback_sync import DirectWritebackSynchronizer


def _writer_done(h2d_sem: CreditGate, *, batch_idx: int, kind: str) -> None:
    writer_done = getattr(h2d_sem, "writer_done", None)
    if callable(writer_done):
        writer_done(batch_idx=batch_idx, kind=kind)
        return
    h2d_sem.release()


class DirectDynamicStateWriter:
    """Synchronous CPU writeback path used by ablation modes."""

    def __init__(
        self,
        *,
        mem_box_store: MemBoxStore,
        h2d_sem: CreditGate,
        synchronizer: DirectWritebackSynchronizer | None = None,
        serialize_writes: bool = True,
    ) -> None:
        self.mem_box_store = mem_box_store
        self.h2d_sem = h2d_sem
        self.synchronizer = synchronizer
        self.serialize_writes = serialize_writes

    @staticmethod
    def _sync_cuda_entry(entry: MemEntry | BoxEntry | None) -> None:
        if entry is None:
            return
        if entry.data.device.type == "cuda":
            torch.cuda.current_stream(device=entry.data.device).synchronize()

    @staticmethod
    def _cpu_filter_nids(filter_nids: torch.Tensor | None) -> torch.Tensor | None:
        if filter_nids is None:
            return None
        return filter_nids.detach().cpu().long()

    def _normalize_mem(
        self,
        entry: MemEntry | None,
        filter_nids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if entry is None:
            return None
        self._sync_cuda_entry(entry)
        nid_cpu = entry.nid.detach().cpu().long()
        data_cpu = entry.data.detach().cpu().to(self.mem_box_store.mem.dtype)
        ts_cpu = entry.ts.detach().cpu().to(self.mem_box_store.mem_ts.dtype)
        filter_cpu = self._cpu_filter_nids(filter_nids)
        if filter_cpu is not None:
            keep = torch.isin(nid_cpu, filter_cpu)
            nid_cpu = nid_cpu[keep]
            data_cpu = data_cpu[keep]
            ts_cpu = ts_cpu[keep]
        return nid_cpu, data_cpu, ts_cpu

    def _normalize_box(
        self,
        entry: BoxEntry | None,
        filter_nids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if entry is None:
            return None
        if self.mem_box_store.box.shape[1] != 1:
            raise ValueError(
                "DirectDynamicStateWriter only supports mailbox boxsize=1."
            )
        self._sync_cuda_entry(entry)
        nid_cpu = entry.nid.detach().cpu().long()
        data_cpu = entry.data.detach().cpu().to(self.mem_box_store.box.dtype)
        ts_cpu = entry.ts.detach().cpu().to(self.mem_box_store.box_ts.dtype)
        filter_cpu = self._cpu_filter_nids(filter_nids)
        if filter_cpu is not None:
            keep = torch.isin(nid_cpu, filter_cpu)
            nid_cpu = nid_cpu[keep]
            data_cpu = data_cpu[keep]
            ts_cpu = ts_cpu[keep]
        return nid_cpu, data_cpu, ts_cpu

    def _prepare_mem(
        self,
        entry: MemEntry | None,
        filter_nids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self._normalize_mem(entry, filter_nids)

    def _prepare_box(
        self,
        entry: BoxEntry | None,
        filter_nids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self._normalize_box(entry, filter_nids)

    def _apply_mem(
        self,
        prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> None:
        if prepared is None:
            return
        nid_cpu, data_cpu, ts_cpu = prepared
        for row_idx in range(nid_cpu.numel()):
            nid = int(nid_cpu[row_idx])
            ts = ts_cpu[row_idx]
            if ts >= self.mem_box_store.mem_ts[nid]:
                self.mem_box_store.mem[nid] = data_cpu[row_idx]
                self.mem_box_store.mem_ts[nid] = ts

    def _apply_box(
        self,
        prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> None:
        if prepared is None:
            return
        nid_cpu, data_cpu, ts_cpu = prepared
        for row_idx in range(nid_cpu.numel()):
            nid = int(nid_cpu[row_idx])
            ts = ts_cpu[row_idx, 0]
            if ts >= self.mem_box_store.box_ts[nid, 0]:
                self.mem_box_store.box[nid, 0] = data_cpu[row_idx, 0]
                self.mem_box_store.box_ts[nid, 0] = ts
                self.mem_box_store.box_pos[nid] = 0

    def write(
        self,
        request: DirectWritebackRequest,
        *,
        release_h2d_credits: bool,
    ) -> None:
        mem = self._prepare_mem(request.mem_entry, request.mem_filter_nids)
        box = self._prepare_box(request.box_entry, request.box_filter_nids)
        transaction = (
            self.synchronizer.transaction(serialize_writes=self.serialize_writes)
            if self.synchronizer is not None
            else nullcontext()
        )
        with transaction:
            self._apply_mem(mem)
            self._apply_box(box)
        if release_h2d_credits:
            _writer_done(self.h2d_sem, batch_idx=request.mini_batch_idx, kind="mem")
            _writer_done(self.h2d_sem, batch_idx=request.mini_batch_idx, kind="box")
