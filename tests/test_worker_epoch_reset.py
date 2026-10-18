import threading
import time
from queue import Queue

import torch

from frost.runtime.worker._engine import Worker
from frost.runtime.worker._pipeline import _initial_h2d_pipeline_slots
from frost.runtime.worker._h2d_credit import BatchAwareCreditManager
from frost.shm.mem_mgr import MemBoxStore


def _make_store() -> MemBoxStore:
    return MemBoxStore(
        mem=torch.zeros(4, 2, dtype=torch.float32),
        mem_ts=torch.zeros(4, dtype=torch.float32),
        box=torch.zeros(4, 1, 3, dtype=torch.float32),
        box_ts=torch.zeros(4, 1, dtype=torch.float32),
        box_pos=torch.zeros(4, dtype=torch.int64),
    )


def _make_worker(store: MemBoxStore) -> Worker:
    worker = Worker.__new__(Worker)
    worker.mem_box_store = store
    worker.mem_task_q = Queue()
    worker.box_task_q = Queue()
    worker.mem_p2p_q = Queue()
    worker.box_p2p_q = Queue()
    worker.mem_wr_q = Queue()
    worker.box_wr_q = Queue()
    worker.p2p_sem = threading.Semaphore(0)
    worker.env = type("EnvStub", (), {"local_rank": 0})()
    return worker


def _start_delayed_consumer(
    q: Queue,
    *,
    delay_s: float,
    write_fn=None,
) -> threading.Thread:
    def _consume() -> None:
        item = q.get(block=True)
        try:
            time.sleep(delay_s)
            if write_fn is not None:
                write_fn(item)
        finally:
            q.task_done()

    t = threading.Thread(target=_consume, daemon=True)
    t.start()
    return t


def test_reset_without_flush_allows_late_writeback_to_leak_into_next_epoch():
    store = _make_store()
    worker = _make_worker(store)

    worker.mem_task_q.put("mem-task")
    worker.box_task_q.put("box-task")
    worker.mem_wr_q.put("mem-write")
    worker.box_wr_q.put("box-write")

    threads = [
        _start_delayed_consumer(worker.mem_task_q, delay_s=0.01),
        _start_delayed_consumer(worker.box_task_q, delay_s=0.01),
        _start_delayed_consumer(
            worker.mem_wr_q,
            delay_s=0.05,
            write_fn=lambda _item: store.mem.__setitem__(0, torch.tensor([7.0, 9.0])),
        ),
        _start_delayed_consumer(
            worker.box_wr_q,
            delay_s=0.05,
            write_fn=lambda _item: store.box.__setitem__(
                (0, 0), torch.tensor([1.0, 2.0, 3.0])
            ),
        ),
    ]

    store.reset()
    for t in threads:
        t.join()

    assert torch.equal(store.mem[0], torch.tensor([7.0, 9.0]))
    assert torch.equal(store.box[0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_flush_epoch_writebacks_blocks_until_async_writes_finish_before_reset():
    store = _make_store()
    worker = _make_worker(store)

    worker.mem_task_q.put("mem-task")
    worker.box_task_q.put("box-task")
    worker.mem_wr_q.put("mem-write")
    worker.box_wr_q.put("box-write")

    threads = [
        _start_delayed_consumer(worker.mem_task_q, delay_s=0.01),
        _start_delayed_consumer(worker.box_task_q, delay_s=0.01),
        _start_delayed_consumer(
            worker.mem_wr_q,
            delay_s=0.05,
            write_fn=lambda _item: store.mem.__setitem__(0, torch.tensor([7.0, 9.0])),
        ),
        _start_delayed_consumer(
            worker.box_wr_q,
            delay_s=0.05,
            write_fn=lambda _item: store.box.__setitem__(
                (0, 0), torch.tensor([1.0, 2.0, 3.0])
            ),
        ),
    ]

    worker._flush_epoch_writebacks()
    store.reset()
    for t in threads:
        t.join()

    assert torch.equal(store.mem, torch.zeros_like(store.mem))
    assert torch.equal(store.box, torch.zeros_like(store.box))


def test_flush_epoch_writebacks_waits_for_p2p_routing_queues_too():
    store = _make_store()
    worker = _make_worker(store)

    worker.mem_p2p_q.put("mem-p2p")
    worker.box_p2p_q.put("box-p2p")

    threads = [
        _start_delayed_consumer(worker.mem_p2p_q, delay_s=0.05),
        _start_delayed_consumer(worker.box_p2p_q, delay_s=0.05),
    ]

    start = time.perf_counter()
    worker._flush_epoch_writebacks()
    elapsed = time.perf_counter() - start
    for t in threads:
        t.join()

    assert elapsed >= 0.04


def test_drain_stale_p2p_guard_tokens_clears_epoch_end_leftover():
    store = _make_store()
    worker = _make_worker(store)

    worker.p2p_sem.release()
    worker.p2p_sem.release()

    drained = worker._drain_stale_p2p_guard_tokens()

    assert drained == 2
    assert not worker.p2p_sem.acquire(blocking=False)


def test_initial_h2d_pipeline_slots_use_single_batch_for_single_worker_memory_runs():
    assert _initial_h2d_pipeline_slots(p2p_world_size=1) == 2
    assert _initial_h2d_pipeline_slots(p2p_world_size=2) == 4


def test_batch_aware_credit_manager_releases_batches_in_order():
    mgr = BatchAwareCreditManager(max_in_flight_batches=2)

    assert mgr.wait_for_batch(0, blocking=False)
    assert mgr.wait_for_batch(1, blocking=False)
    assert not mgr.wait_for_batch(2, blocking=False)
    assert not mgr.wait_for_batch(3, blocking=False)

    mgr.writer_done(batch_idx=1, kind="mem")
    mgr.writer_done(batch_idx=1, kind="box")
    assert not mgr.wait_for_batch(2, blocking=False)

    mgr.writer_done(batch_idx=0, kind="mem")
    assert not mgr.wait_for_batch(2, blocking=False)

    mgr.writer_done(batch_idx=0, kind="box")
    assert mgr.wait_for_batch(2, blocking=False)
    assert mgr.wait_for_batch(3, blocking=False)
    assert not mgr.wait_for_batch(4, blocking=False)


def test_batch_aware_credit_manager_normalizes_strided_global_batch_ids():
    mgr = BatchAwareCreditManager(max_in_flight_batches=2)

    assert mgr.wait_for_batch(1, blocking=False)
    assert mgr.wait_for_batch(3, blocking=False)
    assert not mgr.wait_for_batch(5, blocking=False)

    mgr.writer_done(batch_idx=3, kind="mem")
    mgr.writer_done(batch_idx=3, kind="box")
    assert not mgr.wait_for_batch(5, blocking=False)

    mgr.writer_done(batch_idx=1, kind="mem")
    assert not mgr.wait_for_batch(5, blocking=False)

    mgr.writer_done(batch_idx=1, kind="box")
    assert mgr.wait_for_batch(5, blocking=False)
    assert mgr.wait_for_batch(7, blocking=False)
    assert not mgr.wait_for_batch(9, blocking=False)


def test_batch_aware_credit_manager_single_inflight_batch_waits_for_both_writers():
    mgr = BatchAwareCreditManager(max_in_flight_batches=1)

    assert mgr.wait_for_batch(0, blocking=False)
    assert not mgr.wait_for_batch(1, blocking=False)

    mgr.writer_done(batch_idx=0, kind="mem")
    assert not mgr.wait_for_batch(1, blocking=False)

    mgr.writer_done(batch_idx=0, kind="box")
    assert mgr.wait_for_batch(1, blocking=False)
    assert not mgr.wait_for_batch(2, blocking=False)
