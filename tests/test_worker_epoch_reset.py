import json
import threading
import time
from queue import Queue

import torch
import torch.nn as nn

from frost.runtime.worker._engine import (
    Worker,
    _debug_sync_train_step_boundaries,
    _debug_initial_h2d_slots_override,
    _initial_h2d_pipeline_slots,
)
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
    assert (
        _initial_h2d_pipeline_slots(
            debug_force_cpu_freshness_single_worker=True,
            p2p_world_size=1,
        )
        == 2
    )
    assert (
        _initial_h2d_pipeline_slots(
            debug_force_cpu_freshness_single_worker=False,
            p2p_world_size=1,
        )
        == 2
    )
    assert (
        _initial_h2d_pipeline_slots(
            debug_force_cpu_freshness_single_worker=False,
            p2p_world_size=2,
        )
        == 4
    )


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


def test_initial_h2d_pipeline_slots_honors_env_override(monkeypatch):
    monkeypatch.setenv("FROST_DEBUG_INITIAL_H2D_SLOTS", "2")
    assert _debug_initial_h2d_slots_override() == 2
    assert (
        _initial_h2d_pipeline_slots(
            debug_force_cpu_freshness_single_worker=False,
            p2p_world_size=1,
        )
        == 2
    )


def test_debug_sync_train_step_boundaries_env(monkeypatch):
    monkeypatch.setenv("FROST_DEBUG_SYNC_TRAIN_STEP_BOUNDARIES", "1")
    assert _debug_sync_train_step_boundaries() is True


def test_param_checksum_trace_writes_only_selected_batches(tmp_path, monkeypatch):
    path = tmp_path / "param_checksums.jsonl"
    monkeypatch.setenv("FROST_DEBUG_PARAM_CHECKSUMS_PATH", str(path))
    monkeypatch.setenv("FROST_DEBUG_PARAM_CHECKSUM_BATCHES", "3")

    worker = Worker.__new__(Worker)
    worker.model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        worker.model.weight.copy_(torch.tensor([[1.5, -2.0]]))
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.125)

    meta_skip = type(
        "MetaStub",
        (),
        {
            "mini_batch_idx": 2,
            "cur_epoch": 0,
            "split": "train",
            "cur_macro_of_split": 2,
            "cur_macro_of_epoch": 2,
        },
    )()
    worker._maybe_append_param_checksums(
        meta=meta_skip,
        loss=torch.tensor(1.0),
        grad_scale_before=1024.0,
        grad_scale_after=1024.0,
    )
    assert not path.exists()

    meta_take = type(
        "MetaStub",
        (),
        {
            "mini_batch_idx": 3,
            "cur_epoch": 1,
            "split": "train",
            "cur_macro_of_split": 3,
            "cur_macro_of_epoch": 7,
        },
    )()
    worker._maybe_append_param_checksums(
        meta=meta_take,
        loss=torch.tensor(2.5),
        grad_scale_before=2048.0,
        grad_scale_after=1024.0,
    )

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["mini_batch_idx"] == 3
    assert record["cur_epoch"] == 1
    assert record["loss"] == 2.5
    assert record["grad_scale_before"] == 2048.0
    assert record["grad_scale_after"] == 1024.0
    assert record["lr"] == 0.125
    assert record["num_params"] == 2
    assert record["param_sum"] == -0.5
    assert record["param_abs_sum"] == 3.5
    assert record["param_sq_sum"] == 6.25
    assert record["params"] == [
        {
            "name": "weight",
            "numel": 2,
            "sum": -0.5,
            "abs_sum": 3.5,
            "sq_sum": 6.25,
        }
    ]


def test_grad_checksum_trace_writes_only_selected_batches(tmp_path, monkeypatch):
    path = tmp_path / "grad_checksums.jsonl"
    monkeypatch.setenv("FROST_DEBUG_GRAD_CHECKSUMS_PATH", str(path))
    monkeypatch.setenv("FROST_DEBUG_GRAD_CHECKSUM_BATCHES", "5")

    worker = Worker.__new__(Worker)
    worker.model = nn.Linear(2, 1, bias=False)
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.25)
    worker.model.weight.grad = torch.tensor([[0.25, -0.5]])

    meta_skip = type(
        "MetaStub",
        (),
        {
            "mini_batch_idx": 4,
            "cur_epoch": 0,
            "split": "train",
            "cur_macro_of_split": 4,
            "cur_macro_of_epoch": 4,
        },
    )()
    worker._maybe_append_grad_checksums(
        meta=meta_skip,
        loss=torch.tensor(3.0),
        grad_scale=512.0,
    )
    assert not path.exists()

    meta_take = type(
        "MetaStub",
        (),
        {
            "mini_batch_idx": 5,
            "cur_epoch": 2,
            "split": "train",
            "cur_macro_of_split": 5,
            "cur_macro_of_epoch": 11,
        },
    )()
    worker._maybe_append_grad_checksums(
        meta=meta_take,
        loss=torch.tensor(3.0),
        grad_scale=512.0,
    )

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["mini_batch_idx"] == 5
    assert record["cur_epoch"] == 2
    assert record["loss"] == 3.0
    assert record["grad_scale"] == 512.0
    assert record["num_grad_params"] == 2
    assert record["grad_sum"] == -0.25
    assert record["grad_abs_sum"] == 0.75
    assert record["grad_sq_sum"] == 0.3125
    assert record["grads"] == [
        {
            "name": "weight",
            "numel": 2,
            "sum": -0.25,
            "abs_sum": 0.75,
            "sq_sum": 0.3125,
        }
    ]


def test_step_tensor_dump_writes_selected_name_and_batch(tmp_path, monkeypatch):
    dump_dir = tmp_path / "step_dump"
    monkeypatch.setenv("FROST_DEBUG_DUMP_STEP_TENSOR_DIR", str(dump_dir))
    monkeypatch.setenv("FROST_DEBUG_DUMP_STEP_TENSOR_BATCHES", "2")
    monkeypatch.setenv("FROST_DEBUG_DUMP_STEP_TENSOR_NAMES", "weight")

    worker = Worker.__new__(Worker)
    worker.model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        worker.model.weight.copy_(torch.tensor([[2.0, -3.0]]))
    worker.model.weight.grad = torch.tensor([[0.5, -0.75]])

    meta_skip = type(
        "MetaStub",
        (),
        {
            "mini_batch_idx": 1,
            "cur_epoch": 0,
            "split": "train",
        },
    )()
    worker._maybe_dump_step_tensors(meta=meta_skip, phase="grad")
    assert not dump_dir.exists()

    meta_take = type(
        "MetaStub",
        (),
        {
            "mini_batch_idx": 2,
            "cur_epoch": 3,
            "split": "train",
        },
    )()
    worker._maybe_dump_step_tensors(meta=meta_take, phase="grad")
    worker._maybe_dump_step_tensors(meta=meta_take, phase="param")

    grad_payload = torch.load(dump_dir / "batch_2_grad_weight.pt")
    param_payload = torch.load(dump_dir / "batch_2_param_weight.pt")
    assert grad_payload["name"] == "weight"
    assert grad_payload["phase"] == "grad"
    assert torch.equal(grad_payload["tensor"], torch.tensor([[0.5, -0.75]]))
    assert param_payload["phase"] == "param"
    assert torch.equal(param_payload["tensor"], torch.tensor([[2.0, -3.0]]))
