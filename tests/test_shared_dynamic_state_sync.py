"""CPU-only regression tests with independent spawned shared-memory clients."""

from __future__ import annotations

import multiprocessing as mp
import gc
import queue
import time
import threading
from types import MethodType, SimpleNamespace
import uuid

import pytest
import torch

from frost.shm import _core
from frost.shm.dynamic_state_sync import SharedDynamicStateSync


def _env(rank: int, n_workers: int = 2, *, host: bool = False):
    return SimpleNamespace(
        local_rank=n_workers if host else rank,
        n_gpu=n_workers,
        is_host=host,
    )


def _complete(sync, batch: int) -> None:
    sync.writer_done(batch_idx=batch, kind="mem")
    sync.writer_done(batch_idx=batch, kind="box")


def _client(run_id, rank, scenario, events, results):
    torch.set_num_threads(1)
    sync = SharedDynamicStateSync(_env(rank), run_id, timeout=15)
    values = _core.get_shared_mem_array(f"{run_id}_values", (8,), torch.int64)
    try:
        if scenario == "late_writer":
            for macro in range(3):
                sync.register_batch(macro * 2 + rank, macro, 0)
            sync.wait_before_snapshot(0)
            sync.snapshot_done(0)
            sync.wait_before_snapshot(1)
            sync.snapshot_done(1)
            if rank == 0:
                assert events["release"].wait(10)
                sync.wait_before_write(0)
                # Include node memory, memory timestamp, mailbox data, mailbox
                # timestamp and its ring pointer in the same publication test.
                values[:5] = torch.tensor([101, 102, 103, 104, 105])
                _complete(sync, 0)
                assert events["read_done"].wait(10)
            else:
                _complete(sync, 1)  # empty writes still publish both completions
                events["waiting"].set()
                sync.wait_before_snapshot(2)
                snapshot = values[:5].clone().tolist()
                sync.snapshot_done(2)
                results.put(("snapshot", snapshot))
                events["read_done"].set()
        elif scenario == "late_reader":
            for macro in range(2):
                sync.register_batch(macro * 2 + rank, macro, 0)
            sync.wait_before_snapshot(0)
            sync.snapshot_done(0)
            if rank == 0:
                sync.wait_before_snapshot(1)
                sync.snapshot_done(1)
                events["waiting"].set()
                sync.wait_before_write(0)
                values[:5] = 9
                _complete(sync, 0)
                events["write_done"].set()
            else:
                assert events["release"].wait(10)
                sync.wait_before_snapshot(1)
                snapshot = values[:5].clone().tolist()
                sync.snapshot_done(1)
                results.put(("snapshot", snapshot))
                assert events["write_done"].wait(10)
        elif scenario == "epochs":
            for epoch in range(2):
                macro = epoch
                batch = 2 * macro + rank
                sync.register_batch(batch, macro, epoch)
                sync.wait_before_snapshot(macro)
                assert values[rank].item() == 0
                sync.snapshot_done(macro)
                sync.close_epoch(macro)
                sync.wait_before_write(batch)
                values[rank] = rank + 1
                _complete(sync, batch)

                def reset():
                    assert values[:2].tolist() == [1, 2]
                    values[:2] = 0
                    values[2] += 1

                sync.finish_epoch(reset)
            results.put(("reset", rank, values[2].item()))
        elif scenario == "stress":
            for epoch in range(3):
                failures = []
                queues = {kind: queue.Queue() for kind in ("mem", "box")}

                def writer(kind):
                    try:
                        for _ in range(6):
                            macro, batch = queues[kind].get(timeout=10)
                            sync.wait_before_write(batch)
                            time.sleep(0.001 * ((macro + rank + (kind == "box")) % 3))
                            offset = rank * 3 + (0 if kind == "mem" else 1)
                            values[offset] = macro + 1
                            if kind == "box":
                                values[rank * 3 + 2] = macro + 1
                            sync.writer_done(batch_idx=batch, kind=kind)
                    except BaseException as exc:
                        failures.append(exc)
                        sync.abort(str(exc))

                writers = [threading.Thread(target=writer, args=(kind,)) for kind in queues]
                for thread in writers:
                    thread.start()
                for local_macro in range(6):
                    macro = epoch * 6 + local_macro
                    batch = 2 * macro + rank
                    sync.register_batch(batch, macro, epoch)
                    sync.wait_before_snapshot(macro)
                    time.sleep(0.001 * ((macro + rank) % 3))
                    snapshot = values[:6].clone().tolist()
                    expected = 0 if local_macro < 2 else macro - 1
                    assert snapshot == [expected] * 6, (macro, rank, snapshot, expected)
                    sync.snapshot_done(macro)
                    for q in queues.values():
                        q.put((macro, batch))
                sync.close_epoch(macro)
                for thread in writers:
                    thread.join(10)
                    assert not thread.is_alive()
                if failures:
                    raise failures[0]

                def reset_stress():
                    values[:6] = 0
                    values[7] += 1

                sync.finish_epoch(reset_stress)
            results.put(("stress", rank, values[7].item()))
        elif scenario == "abort":
            for macro in range(3):
                sync.register_batch(macro * 2 + rank, macro, 0)
            if rank == 0:
                assert events["waiting"].wait(10)
                sync.abort("injected writer failure")
            else:
                events["waiting"].set()
                with pytest.raises(RuntimeError, match="aborted by rank 0"):
                    sync.wait_before_snapshot(2)
                results.put(("abort_seen", rank))
        else:
            raise AssertionError(scenario)
    except BaseException as exc:
        sync.abort(str(exc))
        results.put(("error", rank, repr(exc)))
        raise
    finally:
        sync.close()


class _World:
    def __init__(self, n_workers=2, timeout=10):
        self.n_workers = n_workers
        self.run_id = f"state_sync_test_{uuid.uuid4().hex}"
        self.owner = SharedDynamicStateSync(
            _env(0, n_workers, host=True), self.run_id, timeout=timeout
        )
        self.values = _core.create_shared_mem_array(
            f"{self.run_id}_values", (8,), torch.int64
        )
        self.ctx = mp.get_context("spawn")
        self.results = self.ctx.Queue()
        self.events = {
            name: self.ctx.Event()
            for name in ("release", "waiting", "read_done", "write_done")
        }
        self.processes = []

    def spawn(self, scenario):
        for rank in range(self.n_workers):
            process = self.ctx.Process(
                target=_client,
                args=(self.run_id, rank, scenario, self.events, self.results),
            )
            process.start()
            self.processes.append(process)

    def finish(self):
        for process in self.processes:
            process.join(25)
            assert not process.is_alive(), f"Worker {process.pid} did not finish"
            assert process.exitcode == 0

    def close(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
        self.owner.close()
        normalized = _core._normalize_shm_name(f"{self.run_id}_values")
        shm = _core._TRACKED_SHM_OBJECTS.pop(normalized, None)
        self.values = None
        if shm is not None:
            shm.unlink()
            shm.close()
        self.results.close()
        self.results.join_thread()
        self.results = None
        for process in self.processes:
            process.close()
        self.processes.clear()
        self.events.clear()


@pytest.fixture(autouse=True)
def _restore_tracker_semaphore_dispatch(monkeypatch):
    # The existing _core patch recurses for non-shared-memory resources when
    # its singleton methods are patched too. Keep its public suppression but
    # restore the underlying dispatch for multiprocessing's test semaphores.
    from multiprocessing import resource_tracker

    tracker = resource_tracker._resource_tracker
    monkeypatch.setattr(
        tracker, "register", MethodType(resource_tracker.ResourceTracker.register, tracker)
    )
    monkeypatch.setattr(
        tracker, "unregister", MethodType(resource_tracker.ResourceTracker.unregister, tracker)
    )
    yield
    gc.collect()


@pytest.fixture
def world():
    world = _World()
    try:
        yield world
    finally:
        world.close()


def test_cpu_reader_waits_for_other_workers_actual_writeback(world):
    world.spawn("late_writer")
    assert world.events["waiting"].wait(15)
    assert not world.events["read_done"].wait(0.15)
    world.events["release"].set()
    world.finish()
    assert world.results.get(timeout=2) == ("snapshot", [101, 102, 103, 104, 105])


def test_fast_writer_cannot_overwrite_slow_workers_cpu_snapshot(world):
    world.spawn("late_reader")
    assert world.events["waiting"].wait(15)
    assert not world.events["write_done"].wait(0.15)
    world.events["release"].set()
    world.finish()
    assert world.results.get(timeout=2) == ("snapshot", [0, 0, 0, 0, 0])


def test_final_batch_drains_and_two_epochs_reset_once_each(world):
    world.spawn("epochs")
    world.finish()
    assert sorted(world.results.get(timeout=2) for _ in range(2)) == [
        ("reset", 0, 2),
        ("reset", 1, 2),
    ]
    assert world.values[2].item() == 2


def test_two_writers_preserve_exact_snapshots_under_repeated_process_skew(world):
    world.spawn("stress")
    world.finish()
    assert sorted(world.results.get(timeout=2) for _ in range(2)) == [
        ("stress", 0, 3),
        ("stress", 1, 3),
    ]


def test_shared_abort_wakes_an_independent_waiting_process(world):
    world.spawn("abort")
    world.finish()
    assert world.results.get(timeout=2) == ("abort_seen", 1)


def test_completion_watermarks_require_both_writers_without_skipping_holes():
    world = _World(n_workers=1)
    worker = SharedDynamicStateSync(_env(0, 1), world.run_id)
    try:
        for macro in range(3):
            worker.register_batch(100 + macro, macro, 0)
        worker.snapshot_done(1)
        assert worker._table[1, worker._SNAPSHOT].item() == -1
        worker.snapshot_done(0)
        assert worker._table[1, worker._SNAPSHOT].item() == 1
        _complete(worker, 101)
        assert worker._table[1, worker._WRITE].item() == -1
        worker.writer_done(batch_idx=100, kind="mem")
        assert worker._table[1, worker._WRITE].item() == -1
        worker.writer_done(batch_idx=100, kind="box")
        assert worker._table[1, worker._WRITE].item() == 1
        with pytest.raises(ValueError, match="Duplicate"):
            worker.writer_done(batch_idx=101, kind="box")
        with pytest.raises(ValueError, match="Duplicate"):
            worker.snapshot_done(1)
    finally:
        worker.close()
        world.close()


def test_timeout_publishes_abort_to_peers(world):
    waiting = SharedDynamicStateSync(_env(0), world.run_id, timeout=0.02)
    peer = SharedDynamicStateSync(_env(1), world.run_id)
    try:
        for macro in range(3):
            waiting.register_batch(macro * 2, macro, 0)
            peer.register_batch(macro * 2 + 1, macro, 0)
        with pytest.raises(TimeoutError, match="snapshot macro-batch 2"):
            waiting.wait_before_snapshot(2)
        with pytest.raises(RuntimeError, match="aborted by rank 0"):
            peer.wait_before_snapshot(0)
    finally:
        waiting.close()
        peer.close()


def test_stop_event_aborts_shared_wait(world):
    stop = threading.Event()
    worker = SharedDynamicStateSync(_env(0), world.run_id, stop_ev=stop)
    try:
        stop.set()
        with pytest.raises(RuntimeError, match="Stopped"):
            worker.wait_before_snapshot(0)
        assert world.owner._table[0, worker._ABORT].item() == 1
    finally:
        worker.close()


def test_close_requires_final_snapshot_and_register_requires_contiguous_macro(world):
    worker = SharedDynamicStateSync(_env(0), world.run_id)
    try:
        with pytest.raises(ValueError, match="Expected macro-batch 0"):
            worker.register_batch(2, 1, 0)
        worker.register_batch(0, 0, 0)
        with pytest.raises(ValueError, match="snapshot"):
            worker.close_epoch(0)
        worker.snapshot_done(0)
        worker.close_epoch(0)
    finally:
        worker.close()


def test_host_close_is_idempotent_after_global_shared_memory_cleanup(world):
    owner = world.owner
    normalized = _core._normalize_shm_name(owner._name)
    shm = _core._TRACKED_SHM_OBJECTS.pop(normalized)
    # Reproduce cleanup_all_shm's ordering without mutating its global flag.
    shm.close()
    shm.unlink()

    owner.close()
    assert owner._closed
    assert owner._shm is None
    assert not owner.lock_path.exists()
    owner.close()
