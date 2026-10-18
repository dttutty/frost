"""Exercise state-version fences at the real assembler, writer, and worker loops."""

from __future__ import annotations

from contextlib import nullcontext
from queue import Queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, PlanSet
from frost.datatypes.result import Result
from frost.models.contracts import RootExpansion
from frost.runtime.experimental.dynamic_state import resolve_dynamic_state_policy
from frost.runtime.worker._dynamic_assembler import DynamicAssembler
from frost.runtime.worker._engine import Worker
from frost.runtime.worker.p2p_syncer._writer import BoxMailWriter, SharedMemWriter
from frost.shm.mem_mgr import MemBoxStore
from frost.utils.tensor_set import TensorSet


class _Stream:
    def __init__(self, log: list) -> None:
        self.log = log

    def wait_event(self, _event) -> None:
        self.log.append("event_wait")

    def synchronize(self) -> None:
        self.log.append("stream_sync")


class _Event:
    def record(self, *_args) -> None:
        pass


def _store(*, boxsize: int = 1) -> MemBoxStore:
    return MemBoxStore(
        mem=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        mem_ts=torch.arange(4, dtype=torch.int64),
        box=torch.arange(8 * boxsize, dtype=torch.float32).reshape(4, boxsize, 2),
        box_ts=torch.arange(4 * boxsize, dtype=torch.int64).reshape(4, boxsize),
        box_pos=torch.zeros(4, dtype=torch.int64),
    )


def _plan() -> DynamicPlan:
    fetch = TensorSet(torch.tensor([0]))
    return DynamicPlan(
        p2p=[],
        fetch=fetch,
        fetch_on_gpu=fetch.data.clone(),
        recv_counts=np.zeros(1, dtype=np.int32),
        send_counts=np.zeros(1, dtype=np.int32),
        writeback=None,
    )


@pytest.mark.parametrize("state_sync_enabled", [False, True])
def test_assembler_snapshots_before_delta_and_preserves_eval_progress(
    monkeypatch,
    state_sync_enabled: bool,
) -> None:
    log = []
    store = _store(boxsize=3)
    store.box_pos[1] = 2
    graph = FlatGraph(
        nid=torch.tensor([0, 1]), eid=torch.tensor([-1, -1]), ts=torch.zeros(2)
    )
    plan = _plan()
    meta = Meta(
        split="val",
        n_src=1,
        mini_batch_idx=9,
        macro_batch_idx=4,
        epoch_idx=2,
        flat_graph=graph,
        buf_len=2,
        n_neg=1,
        plan_set=PlanSet(mem_plan=plan, box_plan=plan),
        is_last_macro_of_epoch=True,
    )
    old_mem = store.mem.clone()
    old_box = store.box.clone()
    old_box_ts = store.box_ts.clone()

    class _TaskQueue:
        def join(self):
            if state_sync_enabled:
                pytest.fail(
                    "q.join before snapshot deadlocks a writer waiting for that snapshot"
                )
            log.append("legacy_join")

    class _DeltaQueue(Queue):
        def get(self, *args, **kwargs):
            log.append("delta_get")
            return super().get(*args, **kwargs)

    class _Sync:
        def register_batch(self, mini_idx, macro_idx, epoch_idx):
            log.append(("register", mini_idx, macro_idx, epoch_idx))

        def wait_before_snapshot(self, macro_idx):
            assert macro_idx == 4
            log.append("snapshot_wait")

        def snapshot_done(self, macro_idx):
            assert macro_idx == 4
            assert torch.equal(graph.mem[0], old_mem[0])
            assert torch.equal(graph.box[1], old_box[1])
            assert torch.equal(graph.box_ts[1], old_box_ts[1])
            assert graph._box_pos_delta.tolist() == [2]
            log.append("snapshot_done")
            # A writer can now replace the shared version without changing this batch.
            store.mem.fill_(-1)
            store.box.fill_(-1)
            store.box_ts.fill_(-1)
            store.box_pos.zero_()

        def close_epoch(self, macro_idx, *, epoch_id):
            log.append(("close_epoch", macro_idx, epoch_id))

    assembler = DynamicAssembler.__new__(DynamicAssembler)
    assembler.device = torch.device("cpu")
    assembler.shared = store
    assembler.boxsize = 3
    assembler.d_mail = 2
    assembler.n_neg = 1
    assembler.root_expansion = RootExpansion.ALL_ROOTS
    assembler.in_q = Queue()
    assembler.in_q.put((meta, _Event()))
    assembler.in_q.put("end_of_epoch")
    assembler.in_q.put(None)
    assembler.delta_queue = _DeltaQueue()
    assembler.delta_queue.put("nothing")
    assembler.full_q = Queue()
    assembler.stop_ev = threading.Event()
    assembler.state_sync = _Sync() if state_sync_enabled else None
    assembler.h2d_sem = SimpleNamespace(
        wait_for_batch=lambda idx: log.append(("credit", idx))
    )
    assembler.mem_task_q = assembler.box_task_q = _TaskQueue()
    assembler.mem_wr_q = assembler.box_wr_q = _TaskQueue()
    prepare = assembler.prepare_shared_features

    def prepare_shared_features(**kwargs):
        log.append("prepare")
        return prepare(**kwargs)

    assembler.prepare_shared_features = prepare_shared_features
    stream = _Stream(log)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: stream)
    monkeypatch.setattr(torch.cuda, "Event", _Event)

    assembler.logic()

    assert assembler.full_q.get_nowait()[0] is meta
    assert assembler.full_q.get_nowait() == "end_of_epoch"
    assert assembler.full_q.get_nowait() is None
    if state_sync_enabled:
        assert log.index(("register", 9, 4, 2)) < log.index("snapshot_wait")
        assert log.index("snapshot_wait") < log.index("prepare")
        assert (
            log.index("prepare") < log.index("snapshot_done") < log.index("delta_get")
        )
        assert log.count(("close_epoch", 4, 2)) == 2
        assert log.index(("close_epoch", 4, 2)) < log.index("delta_get")
        assert torch.equal(graph.mem[0], old_mem[0])
        assert torch.equal(graph.box[1], old_box[1])
        assert torch.equal(graph.box_ts[1], old_box_ts[1])
        assert graph._box_pos_delta.tolist() == [2]
    else:
        assert log.count("legacy_join") == 4
        assert log.index("legacy_join") < log.index("prepare")


@pytest.mark.parametrize("writer_kind", ["mem", "box"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("state_sync_enabled", [False, True])
def test_writer_publishes_completion_after_shared_write_and_stream_sync(
    monkeypatch,
    writer_kind: str,
    empty: bool,
    state_sync_enabled: bool,
) -> None:
    log = []
    store = _store()
    store.mem.zero_()
    store.mem_ts.zero_()
    store.box.zero_()
    store.box_ts.zero_()
    expected = torch.tensor([5.0, 7.0])
    writer_type = SharedMemWriter if writer_kind == "mem" else BoxMailWriter
    writer = writer_type.__new__(writer_type)
    writer.shared_data = store.mem
    writer.shared_ts = store.mem_ts
    writer.mem_box_store = store
    writer.stop_ev = threading.Event()
    writer.writer_stream = _Stream(log)
    entry = (
        MemEntry(nid=torch.tensor([1]), data=expected[None], ts=torch.tensor([11]))
        if writer_kind == "mem"
        else BoxEntry(
            nid=torch.tensor([1]), data=expected[None, None], ts=torch.tensor([[11]])
        )
    )
    selected = torch.tensor([], dtype=torch.int64) if empty else torch.tensor([1])
    writer.wr_q = Queue()
    writer.wr_q.put((selected, 13, entry, _Event()))
    writer.wr_q.put(None)
    shared = store.mem if writer_kind == "mem" else store.box[:, 0]

    class _Sync:
        def wait_before_write(self, batch_idx):
            assert batch_idx == 13
            assert torch.count_nonzero(shared) == 0
            log.append("write_fence")

        def writer_done(self, *, batch_idx, kind):
            assert batch_idx == 13 and kind == writer_kind
            assert torch.equal(shared[1], torch.zeros(2) if empty else expected)
            if not empty:
                assert log[-1] == "stream_sync"
            log.append("global_done")

    writer.state_sync = _Sync() if state_sync_enabled else None
    writer.h2d_sem = SimpleNamespace(
        writer_done=lambda **_kwargs: log.append("local_done")
    )
    original_write = writer.write_back

    def write_back(*args):
        if state_sync_enabled:
            assert "write_fence" in log
        log.append("cpu_write")
        return original_write(*args)

    writer.write_back = write_back
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

    writer.logic()

    assert writer.wr_q.unfinished_tasks == 0
    assert torch.equal(shared[1], torch.zeros(2) if empty else expected)
    if state_sync_enabled:
        assert (
            log.index("write_fence") < log.index("cpu_write") < log.index("global_done")
        )
        assert log.index("global_done") < log.index("local_done")
    if not empty:
        assert (
            log.index("cpu_write") < log.index("stream_sync") < log.index("local_done")
        )


def test_neighborhood_mail_publishes_ring_pointer_before_global_completion():
    log = []
    store = _store(boxsize=3)
    store.box.zero_()
    store.box_ts.zero_()
    writer = BoxMailWriter.__new__(BoxMailWriter)
    writer.mem_box_store = store
    writer.fanout = 1
    writer.stop_ev = threading.Event()
    seed_mail = MailEntry(
        nid=torch.tensor([0, 1]),
        data=torch.tensor([[5.0, 7.0], [9.0, 11.0]]),
        ts=torch.tensor([13, 17]),
    )
    writer.wr_q = Queue()
    writer.wr_q.put(
        (torch.tensor([2, 3]), 19, (seed_mail, torch.tensor([0, 1, 2, 2, 3]), 1), None)
    )
    writer.wr_q.put(None)

    def writer_done(*, batch_idx, kind):
        assert batch_idx == 19 and kind == "box"
        assert store.box_pos.tolist() == [0, 0, 1, 1]
        assert torch.equal(store.box[2:4, 0], seed_mail.data)
        assert torch.equal(store.box_ts[2:4, 0], seed_mail.ts)
        log.append("done")

    writer.state_sync = SimpleNamespace(
        wait_before_write=lambda idx: log.append(("wait", idx)),
        writer_done=writer_done,
    )
    writer.h2d_sem = SimpleNamespace(writer_done=lambda **_kwargs: None)

    writer.logic()

    assert log == [("wait", 19), "done"]
    assert writer.wr_q.unfinished_tasks == 0


@pytest.mark.parametrize("state_sync_enabled", [False, True])
def test_worker_releases_epoch_result_only_after_flush_and_reset(
    monkeypatch,
    state_sync_enabled: bool,
) -> None:
    log = []
    store = _store()
    worker = Worker.__new__(Worker)
    worker.mem_box_store = store
    worker.dynamic_state_policy = resolve_dynamic_state_policy("baseline")
    worker.nccl_p2p = object()
    worker.h2d_sem = SimpleNamespace(set_max_in_flight_batches=lambda _capacity: None)
    worker.delta_queue = None
    worker.stop_ev = threading.Event()
    worker.start_all_threads = lambda: None
    worker.full_q = Queue()
    worker.full_q.put("end_of_epoch")
    worker.full_q.put(None)
    worker.mem_task_q = Queue()
    worker.box_task_q = Queue()
    worker.mem_p2p_q = Queue()
    worker.curr_epoch = 0
    worker.eval_metric = "ap"
    worker.train_only = True
    result = Result(test_auc=0.0, test_ap=0.0, train_loss=1.0, val_auc=0.0, val_ap=0.0)
    worker.on_train_epoch_end = lambda: result
    observer = SimpleNamespace(on_epoch_end=lambda: None, finalize=lambda: None)
    worker.split_benchmark_observer = observer
    worker.benchmark_observer = observer

    def flush():
        # Delayed writes land as part of draining the queues.
        store.mem.fill_(99)
        log.append("flush")

    def finish_epoch(reset):
        assert log == ["flush", "drain_guard"]
        reset()
        log.append("global_reset")

    class _Results:
        def put(self, actual):
            assert actual is result
            assert "flush" in log
            assert torch.count_nonzero(store.mem) == 0
            assert torch.count_nonzero(store.box) == 0
            if state_sync_enabled:
                assert log[-1] == "global_reset"
            log.append("result")

    worker._flush_epoch_writebacks = flush
    worker._drain_stale_p2p_guard_tokens = lambda: log.append("drain_guard") or 0
    worker.state_sync = (
        SimpleNamespace(finish_epoch=finish_epoch) if state_sync_enabled else None
    )
    worker.result_q = _Results()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)

    worker.run()

    assert log == ["flush", "drain_guard"] + (
        ["global_reset"] if state_sync_enabled else []
    ) + ["result"]
    assert worker.curr_epoch == 1
