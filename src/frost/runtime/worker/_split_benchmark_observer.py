"""Opt-in end-to-end split throughput timing for worker benchmarks."""

from __future__ import annotations

import json
import os
from typing import Literal

import torch

from frost.datatypes.env import Env


_ENV_NAME = "FROST_BENCHMARK_SPLIT_THROUGHPUT"
_Split = Literal["train", "val", "test"]


def _enabled_from_env() -> bool:
    value = os.environ.get(_ENV_NAME, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{_ENV_NAME} must be one of 1/true/yes/on or 0/false/no/off, "
        f"got {value!r}"
    )


class WorkerSplitBenchmarkObserver:
    """Measure each split from its first queued batch through its last GPU work.

    The start event is recorded before waiting for the first batch's H2D-ready
    event.  The end event is recorded after the last batch's model/metric work.
    Consequently the interval includes H2D waits, worker compute, and pipeline
    stalls between batches, but excludes preparation of the next split.
    """

    def __init__(self, *, env: Env, device: torch.device) -> None:
        self.enabled = _enabled_from_env()
        self.rank = env.local_rank
        self.device = device
        self._split: _Split | None = None
        self._epoch = 0
        self._events = 0
        self._batches = 0
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None

    def on_batch_start(self, *, split: _Split, epoch: int, events: int) -> None:
        if not self.enabled:
            return
        if events <= 0:
            raise ValueError(f"split benchmark events must be positive, got {events}")
        if self._split != split:
            self._finalize()
            self._split = split
            self._epoch = epoch
            self._events = 0
            self._batches = 0
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record(torch.cuda.current_stream(self.device))
        elif self._epoch != epoch:
            raise RuntimeError(
                "split changed epoch without an end-of-epoch boundary: "
                f"active={self._epoch}, received={epoch}"
            )
        self._events += events
        self._batches += 1

    def on_batch_end(self) -> None:
        if self.enabled:
            assert self._end is not None
            self._end.record(torch.cuda.current_stream(self.device))

    def on_epoch_end(self) -> None:
        if self.enabled:
            self._finalize()

    def finalize(self) -> None:
        if self.enabled:
            self._finalize()

    def _finalize(self) -> None:
        if self._split is None:
            return
        assert self._start is not None
        assert self._end is not None
        self._end.synchronize()
        elapsed_s = self._start.elapsed_time(self._end) / 1000.0
        if elapsed_s <= 0:
            raise RuntimeError(
                f"split benchmark measured non-positive duration {elapsed_s}"
            )
        payload = {
            "system": "frost",
            "rank": self.rank,
            "epoch": self._epoch,
            "split": self._split,
            "events": self._events,
            "batches": self._batches,
            "elapsed_s": elapsed_s,
            "events_per_sec": self._events / elapsed_s,
        }
        print("FROST_SPLIT_THROUGHPUT " + json.dumps(payload, sort_keys=True), flush=True)
        self._split = None
        self._start = None
        self._end = None
