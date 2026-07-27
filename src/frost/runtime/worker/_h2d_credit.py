from __future__ import annotations

import threading
import time
from typing import Literal, Protocol


class CreditGate(Protocol):
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool: ...

    def release(self, n: int = 1) -> None: ...


class BatchAwareCreditManager:
    """Throttle DynamicAssembler with explicit batch-drained semantics.

    The intended invariant is:
    - batches ``0..max_in_flight_batches-1`` may start immediately
    - batch ``n`` may start only after batch ``n-max_in_flight_batches`` has
      fully drained both memory and mailbox writeback

    With the default ``max_in_flight_batches=2``, this means batch ``n`` sees
    at least batch ``n-2`` persisted in shared memory before it begins shared
    fetch assembly.

    ``acquire`` / ``release`` are kept only as a compatibility layer for older
    tests and debug scaffolding. The main runtime should use
    ``wait_for_batch(batch_idx)``.
    """

    def __init__(self, *, max_in_flight_batches: int = 2) -> None:
        if max_in_flight_batches < 1:
            raise ValueError("max_in_flight_batches must be >= 1")
        self._sem = threading.Semaphore(0)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._next_batch_to_release = 0
        self._done: dict[int, set[str]] = {}
        self._batch_to_local_idx: dict[int, int] = {}
        self._next_local_idx = 0
        self._max_in_flight_batches = max_in_flight_batches

    def _local_batch_idx(self, batch_idx: int) -> int:
        """Map global batch ids to a dense per-worker sequence.

        Host-assigned ``mini_batch_idx`` values are global across all workers,
        so each worker observes a strided subsequence such as ``0, 2, 4, ...``
        or ``1, 3, 5, ...``. Credit accounting must therefore operate on the
        local observation order instead of the raw global ids.
        """
        local_idx = self._batch_to_local_idx.get(batch_idx)
        if local_idx is not None:
            return local_idx

        local_idx = self._next_local_idx
        self._batch_to_local_idx[batch_idx] = local_idx
        self._next_local_idx += 1
        return local_idx

    def set_max_in_flight_batches(self, batches: int) -> None:
        if batches < 1:
            raise ValueError("batches must be >= 1")
        with self._cv:
            self._max_in_flight_batches = batches
            self._cv.notify_all()

    def wait_for_batch(
        self,
        batch_idx: int,
        *,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        if batch_idx < 0:
            raise ValueError("batch_idx must be >= 0")

        with self._cv:
            local_idx = self._local_batch_idx(batch_idx)

            def _ready() -> bool:
                return local_idx < (
                    self._next_batch_to_release + self._max_in_flight_batches
                )

            if not blocking:
                return _ready()

            if timeout is None:
                while not _ready():
                    self._cv.wait()
                return True

            end_time = time.monotonic() + timeout
            while not _ready():
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
            return True

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        if timeout is None:
            return self._sem.acquire(blocking=blocking)
        return self._sem.acquire(blocking=blocking, timeout=timeout)

    def release(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError("n must be >= 0")
        for _ in range(n):
            self._sem.release()

    def writer_done(self, *, batch_idx: int, kind: Literal["mem", "box"]) -> None:
        if batch_idx < 0:
            raise ValueError("batch_idx must be >= 0")
        if kind not in {"mem", "box"}:
            raise ValueError(f"Unsupported writeback kind: {kind}")

        releases = 0
        with self._lock:
            local_idx = self._local_batch_idx(batch_idx)
            done = self._done.setdefault(local_idx, set())
            if kind in done:
                raise ValueError(
                    f"Duplicate writeback completion for batch {batch_idx}, kind={kind}"
                )
            done.add(kind)

            while True:
                next_done = self._done.get(self._next_batch_to_release)
                if next_done is None or next_done != {"mem", "box"}:
                    break
                del self._done[self._next_batch_to_release]
                self._next_batch_to_release += 1
                releases += 2
            if releases > 0:
                self._cv.notify_all()

        if releases > 0:
            self.release(releases)
