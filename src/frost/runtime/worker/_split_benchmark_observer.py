"""Opt-in end-to-end split throughput timing for worker benchmarks."""

from __future__ import annotations

import json
import os
from typing import Literal

import torch

from frost.datatypes.env import Env


_ENV_NAME = "FROST_BENCHMARK_SPLIT_THROUGHPUT"
_WARMUP_ENV_NAME = "FROST_SPLIT_THROUGHPUT_WARMUP_BATCHES"
_Split = Literal["train", "val", "test"]


def _enabled_from_env() -> bool:
    value = os.environ.get(_ENV_NAME, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{_ENV_NAME} must be one of 1/true/yes/on or 0/false/no/off, got {value!r}"
    )


def _warmup_batches_from_env(split: _Split) -> int:
    split_name = f"{_WARMUP_ENV_NAME}_{split.upper()}"
    encoded = os.environ.get(split_name, os.environ.get(_WARMUP_ENV_NAME, "0"))
    try:
        warmup_batches = int(encoded)
    except ValueError as exc:
        raise ValueError(f"{split_name} must be an integer, got {encoded!r}") from exc
    if warmup_batches < 0:
        raise ValueError(f"{split_name} must be >= 0, got {warmup_batches}")
    return warmup_batches


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
        self._scored_pairs = 0
        self._batches = 0
        self._seen_batches = 0
        self._warmup_batches = 0
        self._measure_current_batch = False
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None

    def on_batch_start(
        self, *, split: _Split, epoch: int, events: int, scored_pairs: int
    ) -> None:
        if not self.enabled:
            return
        if events <= 0:
            raise ValueError(f"split benchmark events must be positive, got {events}")
        if scored_pairs < events:
            raise ValueError(
                "split benchmark scored pairs must be at least the event count, "
                f"got scored_pairs={scored_pairs}, events={events}"
            )
        if self._split != split:
            self._finalize()
            self._split = split
            self._epoch = epoch
            self._events = 0
            self._scored_pairs = 0
            self._batches = 0
            self._seen_batches = 0
            self._warmup_batches = _warmup_batches_from_env(split)
        elif self._epoch != epoch:
            raise RuntimeError(
                "split changed epoch without an end-of-epoch boundary: "
                f"active={self._epoch}, received={epoch}"
            )
        self._seen_batches += 1
        self._measure_current_batch = self._seen_batches > self._warmup_batches
        if not self._measure_current_batch:
            return
        if self._start is None:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record(torch.cuda.current_stream(self.device))
        self._events += events
        self._scored_pairs += scored_pairs
        self._batches += 1

    def on_batch_end(self) -> None:
        if self.enabled and self._measure_current_batch:
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
        if self._start is None:
            self._reset()
            return
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
            "scored_pairs": self._scored_pairs,
            "batches": self._batches,
            "warmup_batches": self._warmup_batches,
            "elapsed_s": elapsed_s,
            "events_per_sec": self._events / elapsed_s,
            "scored_pairs_per_sec": self._scored_pairs / elapsed_s,
        }
        print(
            "FROST_SPLIT_THROUGHPUT " + json.dumps(payload, sort_keys=True), flush=True
        )
        self._reset()

    def _reset(self) -> None:
        self._split = None
        self._measure_current_batch = False
        self._start = None
        self._end = None
