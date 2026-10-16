"""Cross-process ordering for the shared CPU dynamic-state table.

This protocol is for the two-macro-batch asynchronous P2P pipeline.  A CPU
snapshot for macro-batch k follows every worker's writeback through k-2;
writeback j follows every worker's snapshot j+1.  The latter condition keeps
fast workers from overwriting a slow worker's CPU snapshot.  P2P/reuse supplies
the intervening update.  These are CPU publication fences, not CUDA events:
callers must finish the actual CPU copy before reporting completion.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable, Literal

import torch

from frost.shm import _core


class SharedDynamicStateSync:
    """Order CPU snapshots and writebacks across independent worker processes.

    The host constructs the owner before workers attach, using the same unique
    ``run_id``.  No distributed collective is used by this class.  ``macro_id``
    must be a contiguous, run-global sequence starting at zero on every worker;
    ``batch_idx`` may instead be the host's strided mini-batch identifier.

    ``close_epoch`` is called after the last local snapshot, before waiting for
    writers to drain.  ``finish_epoch`` is called by every worker only after its
    epoch threads/queues are quiescent; worker zero performs the shared reset.
    """

    _VERSION = 1
    _WIDTH = 8
    _VERSION_COL = 0
    _N_WORKERS = 1
    _ABORT = 2
    _RESET_DONE = 3
    _SNAPSHOT = 0
    _WRITE = 1
    _CLOSED_LAST = 2
    _CLOSED_EPOCH = 3
    _RESET_READY = 4

    def __init__(
        self,
        env,
        run_id: str,
        *,
        timeout: float = 300.0,
        poll_interval: float = 0.001,
        stop_ev: threading.Event | None = None,
    ) -> None:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        self.env = env
        self.n_workers = int(env.n_gpu)
        self.worker_id = None if env.is_host else int(env.local_rank)
        if self.n_workers < 1:
            raise ValueError("At least one worker is required")
        if self.worker_id is not None and not 0 <= self.worker_id < self.n_workers:
            raise ValueError("Worker rank is outside the shared-state worker set")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.stop_ev = stop_ev
        digest = hashlib.sha256(run_id.encode()).hexdigest()[:32]
        self._name = f"{run_id}_dynamic_state_sync"
        self.lock_path = Path(tempfile.gettempdir()) / f"frost-state-sync-{digest}.lock"
        self._thread_lock = threading.Lock()
        self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        self._closed = False
        self._table: torch.Tensor | None = None
        self._shm: SharedMemory | None = None
        self._batch_to_macro: dict[int, int] = {}
        self._macro_to_epoch: dict[int, int] = {}
        self._done: dict[int, set[str]] = {}
        self._snapshots_done: set[int] = set()
        self._next_registered = 0
        self._epoch_start_macro = 0
        self._last_macro: int | None = None
        self._epoch_id: int | None = None
        self._last_finished_epoch = -1
        shape = (self.n_workers + 1, self._WIDTH)
        if env.is_host:
            with self._locked():
                table = _core.create_shared_mem_array(self._name, shape, torch.int64)
                table.fill_(-1)
                table[0, self._ABORT] = 0
                table[0, self._N_WORKERS] = self.n_workers
                table[0, self._VERSION_COL] = self._VERSION
                self._table = table
                self._shm = _core._TRACKED_SHM_OBJECTS[
                    _core._normalize_shm_name(self._name)
                ]
        else:
            end = time.monotonic() + self.timeout
            while True:
                with self._locked():
                    try:
                        table = _core.get_shared_mem_array(self._name, shape, torch.int64)
                    except FileNotFoundError:
                        table = None
                    if table is not None:
                        if int(table[0, self._VERSION_COL]) != self._VERSION:
                            raise RuntimeError("Unsupported shared dynamic-state protocol")
                        if int(table[0, self._N_WORKERS]) != self.n_workers:
                            raise RuntimeError("Shared dynamic-state worker count mismatch")
                        self._table = table
                        self._shm = _core._TRACKED_SHM_OBJECTS[
                            _core._normalize_shm_name(self._name)
                        ]
                        break
                if self.stop_ev is not None and self.stop_ev.is_set():
                    raise RuntimeError("Stopped while attaching shared dynamic-state protocol")
                if time.monotonic() >= end:
                    raise TimeoutError("Timed out attaching shared dynamic-state protocol")
                time.sleep(self.poll_interval)

    @contextmanager
    def _locked(self):
        # flock on the same open file description does not exclude sibling
        # threads, so each process additionally owns a normal threading lock.
        with self._thread_lock:
            if self._closed:
                raise RuntimeError("Shared dynamic-state protocol is closed")
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _worker_row(self) -> int:
        if self.worker_id is None:
            raise RuntimeError("Only workers may publish dynamic-state progress")
        return self.worker_id + 1

    def _check_abort_locked(self) -> None:
        assert self._table is not None
        abort_rank = int(self._table[0, self._ABORT])
        if abort_rank:
            raise RuntimeError(f"Shared dynamic-state protocol aborted by rank {abort_rank - 1}")

    def _wait(self, description: str, ready: Callable[[], bool]) -> None:
        end = time.monotonic() + self.timeout
        while True:
            if self.stop_ev is not None and self.stop_ev.is_set():
                self.abort()
                raise RuntimeError(f"Stopped while {description}")
            with self._locked():
                self._check_abort_locked()
                if ready():
                    return
                if time.monotonic() >= end:
                    assert self._table is not None
                    progress = self._table[1:, :4].tolist()
                    self._table[0, self._ABORT] = int(self.env.local_rank) + 1
                    raise TimeoutError(f"Timed out {description}; worker progress={progress}")
            time.sleep(self.poll_interval)

    def _all_reached(self, column: int, required: int) -> bool:
        assert self._table is not None
        return bool(torch.all(self._table[1:, column] >= required))

    def _reset_completed(self, epoch_id: int) -> bool:
        assert self._table is not None
        return int(self._table[0, self._RESET_DONE]) >= epoch_id

    def register_batch(self, batch_idx: int, macro_id: int, epoch_id: int) -> None:
        """Register a batch before any snapshot/writeback work is queued."""
        if min(batch_idx, macro_id, epoch_id) < 0:
            raise ValueError("Batch, macro-batch and epoch identifiers must be nonnegative")
        with self._locked():
            self._check_abort_locked()
            if epoch_id <= self._last_finished_epoch:
                raise ValueError("Epoch identifiers must strictly increase after a reset")
            previous = self._batch_to_macro.get(batch_idx)
            if previous is not None:
                if previous != macro_id or self._macro_to_epoch[macro_id] != epoch_id:
                    raise ValueError(f"Conflicting registration for batch {batch_idx}")
                return
            if macro_id != self._next_registered:
                raise ValueError(
                    f"Expected macro-batch {self._next_registered}, received {macro_id}"
                )
            if self._epoch_id is not None and epoch_id != self._epoch_id:
                raise ValueError("finish_epoch must complete before registering a new epoch")
            self._batch_to_macro[batch_idx] = macro_id
            self._macro_to_epoch[macro_id] = epoch_id
            self._next_registered += 1
            self._last_macro = macro_id
            self._epoch_id = epoch_id

    def _registered_macro(self, batch_idx: int) -> int:
        try:
            return self._batch_to_macro[batch_idx]
        except KeyError as exc:
            raise ValueError(f"Unregistered batch {batch_idx}") from exc

    def wait_before_snapshot(self, macro_id: int) -> None:
        """Wait before copying shared memory/mailbox data into private buffers."""
        required = max(self._epoch_start_macro - 1, macro_id - 2)
        self._wait(
            f"waiting to snapshot macro-batch {macro_id}",
            lambda: self._all_reached(self._WRITE, required),
        )

    def snapshot_done(self, macro_id: int) -> None:
        """Publish completion of all CPU gathers, including mailbox metadata."""
        row = self._worker_row()
        with self._locked():
            self._check_abort_locked()
            assert self._table is not None
            if macro_id not in self._macro_to_epoch:
                raise ValueError(f"Unregistered macro-batch {macro_id}")
            watermark = int(self._table[row, self._SNAPSHOT])
            if macro_id <= watermark or macro_id in self._snapshots_done:
                raise ValueError(f"Duplicate snapshot completion for macro-batch {macro_id}")
            self._snapshots_done.add(macro_id)
            while watermark + 1 in self._snapshots_done:
                watermark += 1
                self._snapshots_done.remove(watermark)
            self._table[row, self._SNAPSHOT] = watermark

    def wait_before_write(self, batch_idx: int) -> None:
        """Wait before either writer mutates shared CPU memory/mailbox state."""
        macro_id = self._registered_macro(batch_idx)
        epoch_id = self._macro_to_epoch[macro_id]

        def ready() -> bool:
            assert self._table is not None
            workers = self._table[1:]
            if not bool(torch.all(workers[:, self._WRITE] >= macro_id - 1)):
                return False
            snapshotted = bool(torch.all(workers[:, self._SNAPSHOT] >= macro_id + 1))
            closed = bool(
                torch.all(workers[:, self._CLOSED_LAST] == macro_id)
                and torch.all(workers[:, self._CLOSED_EPOCH] == epoch_id)
                and torch.all(workers[:, self._SNAPSHOT] >= macro_id)
            )
            return snapshotted or closed

        self._wait(f"waiting to write macro-batch {macro_id}", ready)

    def writer_done(self, *, batch_idx: int, kind: Literal["mem", "box"]) -> None:
        """Publish actual CPU writes; empty writes must report completion too."""
        if kind not in {"mem", "box"}:
            raise ValueError(f"Unsupported writeback kind: {kind}")
        row = self._worker_row()
        with self._locked():
            self._check_abort_locked()
            assert self._table is not None
            macro_id = self._registered_macro(batch_idx)
            watermark = int(self._table[row, self._WRITE])
            if macro_id <= watermark or kind in self._done.get(macro_id, set()):
                raise ValueError(f"Duplicate {kind} completion for macro-batch {macro_id}")
            self._done.setdefault(macro_id, set()).add(kind)
            while self._done.get(watermark + 1) == {"mem", "box"}:
                watermark += 1
                del self._done[watermark]
            self._table[row, self._WRITE] = watermark

    def close_epoch(self, last_macro: int, epoch_id: int | None = None) -> None:
        """Declare that no snapshot exists after this epoch's final batch."""
        row = self._worker_row()
        epoch_id = self._epoch_id if epoch_id is None else epoch_id
        if epoch_id is None or last_macro != self._last_macro:
            raise ValueError("Epoch closure must identify the last registered macro-batch")
        with self._locked():
            self._check_abort_locked()
            assert self._table is not None
            if int(self._table[row, self._SNAPSHOT]) < last_macro:
                raise ValueError("The final CPU snapshot must complete before closing the epoch")
            self._table[row, self._CLOSED_LAST] = last_macro
            self._table[row, self._CLOSED_EPOCH] = epoch_id

    def wait_epoch_drained(self, last_macro: int | None = None) -> None:
        """Wait until every worker has finished both final CPU writers."""
        last_macro = self._last_macro if last_macro is None else last_macro
        if last_macro is None:
            return
        self._wait(
            f"draining through macro-batch {last_macro}",
            lambda: self._all_reached(self._WRITE, last_macro),
        )

    def finish_epoch(self, reset_fn: Callable[[], None]) -> None:
        """Drain, rendezvous, reset once on worker zero, and release next epoch."""
        row = self._worker_row()
        if self._last_macro is None or self._epoch_id is None:
            raise ValueError("Cannot finish an epoch without registered batches")
        last_macro, epoch_id = self._last_macro, self._epoch_id
        self.wait_epoch_drained(last_macro)
        with self._locked():
            assert self._table is not None
            self._check_abort_locked()
            if int(self._table[row, self._CLOSED_EPOCH]) != epoch_id:
                raise ValueError("close_epoch must precede finish_epoch")
            self._table[row, self._RESET_READY] = epoch_id
        self._wait(
            f"waiting for epoch {epoch_id} reset rendezvous",
            lambda: self._all_reached(self._RESET_READY, epoch_id),
        )
        if self.worker_id == 0:
            try:
                # Keep publication and reset under the same process-shared lock.
                with self._locked():
                    self._check_abort_locked()
                    assert self._table is not None
                    reset_fn()
                    self._table[0, self._RESET_DONE] = epoch_id
            except BaseException:
                self.abort()
                raise
        else:
            self._wait(
                f"waiting for epoch {epoch_id} reset completion",
                lambda: self._reset_completed(epoch_id),
            )
        with self._locked():
            self._batch_to_macro.clear()
            self._macro_to_epoch.clear()
            self._done.clear()
            self._snapshots_done.clear()
            self._epoch_start_macro = last_macro + 1
            self._last_finished_epoch = epoch_id
            self._epoch_id = None
            self._last_macro = None

    def abort(self, reason: str | None = None) -> None:
        """Wake peers after a local failure or stop request."""
        del reason  # The shared flag identifies the failed rank; caller logs detail.
        if self._closed or self._table is None:
            return
        with self._locked():
            if not int(self._table[0, self._ABORT]):
                self._table[0, self._ABORT] = int(self.env.local_rank) + 1

    def close(self) -> None:
        """Close this protocol instance; the host removes run-owned resources."""
        with self._thread_lock:
            if self._closed:
                return
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                self._table = None
                normalized = _core._normalize_shm_name(self._name)
                shm, self._shm = self._shm, None
                if _core._TRACKED_SHM_OBJECTS.get(normalized) is shm:
                    _core._TRACKED_SHM_OBJECTS.pop(normalized, None)
                if shm is not None:
                    try:
                        if self.env.is_host:
                            try:
                                shm.unlink()
                            except FileNotFoundError:
                                # Global SHM cleanup may already have unlinked it.
                                pass
                    finally:
                        shm.close()
                if self.env.is_host:
                    self.lock_path.unlink(missing_ok=True)
                self._closed = True
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
