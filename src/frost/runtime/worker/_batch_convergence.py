"""Opt-in per-batch training-loss capture without step-wise GPU synchronization."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


class BatchConvergenceRecorder:
    """Buffer scalar losses asynchronously and write one NPZ file per epoch."""

    _chunk_size = 65_536

    def __init__(self, *, system: str, rank: int, output_dir: Path | None) -> None:
        self.system = system
        self.rank = rank
        self.output_dir = output_dir
        self._chunks: list[torch.Tensor] = []
        self._events: list[int] = []
        self._count = 0
        self._contains_cuda = False

    @classmethod
    def from_environment(cls, *, system: str, rank: int) -> BatchConvergenceRecorder:
        encoded = os.environ.get("FROST_BATCH_CONVERGENCE_DIR")
        output_dir = Path(encoded).expanduser() if encoded else None
        return cls(system=system, rank=rank, output_dir=output_dir)

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def record(self, *, events: int, loss: torch.Tensor) -> None:
        if not self.enabled:
            return
        if loss.numel() != 1:
            raise ValueError(f"batch loss must be scalar, got shape {tuple(loss.shape)}")
        chunk_index, offset = divmod(self._count, self._chunk_size)
        if chunk_index == len(self._chunks):
            self._chunks.append(
                torch.empty(
                    self._chunk_size,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=loss.is_cuda,
                )
            )
        value = loss.detach().float().reshape(())
        self._chunks[chunk_index][offset].copy_(
            value,
            non_blocking=loss.is_cuda,
        )
        self._events.append(events)
        self._count += 1
        self._contains_cuda = self._contains_cuda or loss.is_cuda

    def flush_epoch(self, epoch: int) -> Path | None:
        if not self.enabled or self._count == 0:
            self._reset()
            return None
        if self._contains_cuda:
            torch.cuda.current_stream().synchronize()
        losses = np.concatenate(
            [
                chunk[: min(self._chunk_size, self._count - index * self._chunk_size)]
                .numpy()
                .copy()
                for index, chunk in enumerate(self._chunks)
            ]
        )
        events = np.asarray(self._events, dtype=np.int64)
        if losses.shape != events.shape:
            raise RuntimeError(
                f"loss/event curve length mismatch: {losses.shape} != {events.shape}"
            )
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / (
            f"{self.system}_rank{self.rank}_epoch{epoch:03d}.npz"
        )
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                system=np.asarray(self.system),
                rank=np.asarray(self.rank, dtype=np.int64),
                epoch=np.asarray(epoch, dtype=np.int64),
                batch=np.arange(self._count, dtype=np.int64),
                events=events,
                loss=losses,
            )
        os.replace(temporary, destination)
        self._reset()
        return destination

    def _reset(self) -> None:
        self._chunks.clear()
        self._events.clear()
        self._count = 0
        self._contains_cuda = False
