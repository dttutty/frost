import threading
from queue import Queue
from types import SimpleNamespace

import numpy as np
import torch

from frost.models.contracts import RootExpansion

from frost.datatypes.graph import FlatGraph
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, PlanSet
from frost.datatypes.tasks import P2PTask
from frost.runtime.worker._dynamic_assembler import DynamicAssembler
from frost.runtime.worker._dynamic_state import DynamicPrepareProfiler, MemAndBox
from frost.runtime.worker.p2p_syncer._p2p_comm import P2PQHandler
from frost.shm.mem_mgr import MemBoxStore
from frost.utils.tensor_set import TensorSet


class FakeStream:
    def wait_event(self, _event) -> None:
        pass

    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class FakeCudaEvent:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def record(self, *_args, **_kwargs) -> None:
        pass

    def elapsed_time(self, _other) -> float:
        return 1.25


class FakeWork:
    def __init__(self) -> None:
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1


def _make_store(
    boxsize: int = 1, d_mem: int = 2, d_mail: int = 3, n_nodes: int = 5
) -> MemBoxStore:
    return MemBoxStore(
        mem=torch.arange(n_nodes * d_mem, dtype=torch.float32).view(n_nodes, d_mem),
        mem_ts=torch.arange(n_nodes, dtype=torch.float32),
        box=torch.arange(n_nodes * boxsize * d_mail, dtype=torch.float32).view(
            n_nodes, boxsize, d_mail
        ),
        box_ts=torch.arange(n_nodes * boxsize, dtype=torch.float32).view(
            n_nodes, boxsize
        ),
        box_pos=torch.zeros(n_nodes, dtype=torch.int64),
    )


def _make_plan(fetch_nids: list[int]) -> DynamicPlan:
    fetch = TensorSet(torch.tensor(fetch_nids, dtype=torch.long))
    return DynamicPlan(
        p2p=[],
        fetch=fetch,
        recv_counts=np.zeros(1, dtype=np.int64),
        send_counts=np.zeros(1, dtype=np.int64),
        writeback=None,
        fetch_on_gpu=fetch.data.clone(),
    )


def _make_graph(nids: list[int]) -> FlatGraph:
    nid = torch.tensor(nids, dtype=torch.long)
    return FlatGraph(
        nid=nid,
        eid=torch.full_like(nid, -1),
        ts=torch.zeros(nid.shape[0], dtype=torch.float32),
    )


def _make_meta(
    graph: FlatGraph,
    mem_plan: DynamicPlan,
    box_plan: DynamicPlan,
    *,
    n_src: int = 1,
) -> Meta:
    return Meta(
        split="train",
        n_src=n_src,
        mini_batch_idx=0,
        flat_graph=graph,
        buf_len=graph.nid.numel(),
        plan_set=PlanSet(mem_plan=mem_plan, box_plan=box_plan),
        n_neg=1,
    )


def _make_assembler(
    store: MemBoxStore,
    *,
    n_neg: int = 1,
    root_expansion: RootExpansion = RootExpansion.ALL_ROOTS,
) -> DynamicAssembler:
    assembler = DynamicAssembler.__new__(DynamicAssembler)
    assembler.device = torch.device("cpu")
    assembler.shared = store
    assembler.boxsize = store.box.shape[1]
    assembler.d_mail = store.box.shape[2]
    assembler.n_neg = n_neg
    assembler.root_expansion = root_expansion
    return assembler


def _make_logic_assembler(
    store: MemBoxStore,
    *,
    n_neg: int = 1,
    root_expansion: RootExpansion = RootExpansion.ALL_ROOTS,
) -> DynamicAssembler:
    assembler = _make_assembler(
        store,
        n_neg=n_neg,
        root_expansion=root_expansion,
    )
    assembler.in_q = Queue()
    assembler.delta_queue = Queue()
    assembler.full_q = Queue()
    assembler.stop_ev = threading.Event()
    assembler.h2d_sem = threading.Semaphore(0)
    return assembler


def _make_p2p_task(data_shape: tuple[int, ...]) -> P2PTask:
    length = data_shape[0]
    return P2PTask(
        send_nid=torch.arange(length, dtype=torch.long),
        recv_nid=torch.arange(length, dtype=torch.long),
        data2send=torch.arange(np.prod(data_shape), dtype=torch.float32).view(
            data_shape
        ),
        data2recv=torch.zeros(data_shape, dtype=torch.float32),
        ts2send=torch.arange(length, dtype=torch.float32),
        ts2recv=torch.zeros(length, dtype=torch.float32),
        recv_counts=np.array([length], dtype=np.int64),
        send_counts=np.array([length], dtype=np.int64),
        mini_batch_idx=0,
    )


def _make_handler() -> P2PQHandler:
    handler = P2PQHandler.__new__(P2PQHandler)
    handler.device = torch.device("cpu")
    handler.mem_p2p_q = Queue()
    handler.box_p2p_q = Queue()
    handler.p2p_sem = threading.Semaphore(0)
    handler.env = SimpleNamespace(local_rank=0)
    handler.stop_ev = threading.Event()
    handler.delta_queue = Queue()
    handler.nccl_p2p = object()
    handler.count = 0
    return handler


class TestDynamicAssembler:
    def test_dynamic_prepare_profiler_splits_fetch_and_assembly(self, monkeypatch):
        store = _make_store(boxsize=1)
        assembler = _make_assembler(store)
        graph = _make_graph([1, 4, 2])
        shared_plan = _make_plan([1, 2])
        profiler = DynamicPrepareProfiler(enabled=True)
        monkeypatch.setattr(torch.cuda, "Event", FakeCudaEvent)

        assembler.prepare_shared_features(
            graph,
            shared_plan,
            shared_plan,
            n_src=1,
            profiler=profiler,
        )
        payload = profiler.payload()

        assert payload["mem_fetch_rows_curve"] == [2]
        assert payload["box_fetch_rows_curve"] == [2]
        assert len(payload["mem_cpu_gather_s_curve"]) == 1
        assert len(payload["box_cpu_gather_s_curve"]) == 1
        assert payload["mem_h2d_cuda_ms_curve"] == [1.25]
        assert payload["box_h2d_cuda_ms_curve"] == [1.25]
        assert payload["mem_scatter_cuda_ms_curve"] == [1.25]
        assert payload["box_scatter_cuda_ms_curve"] == [1.25]
        assert payload["zero_init_cuda_ms_curve"] == [1.25]
        assert payload["lookup_cuda_ms_curve"] == [1.25]

    def test_dynamic_prepare_reuses_cpu_gather_buffers(self):
        store = _make_store(boxsize=1)
        assembler = _make_assembler(store)
        first_graph = _make_graph([1, 4, 2])
        first_plan = _make_plan([1, 2])

        assembler.prepare_shared_features(
            first_graph,
            first_plan,
            first_plan,
            n_src=1,
        )
        mem_buffer = assembler._shared_gather_buffers["mem"]
        box_buffer = assembler._shared_gather_buffers["box"]
        mem_data_ptr = mem_buffer.data.data_ptr()
        box_data_ptr = box_buffer.data.data_ptr()

        store.mem[1] = torch.tensor([101.0, 102.0])
        store.box[1, 0] = torch.tensor([103.0, 104.0, 105.0])
        second_graph = _make_graph([1, 2])
        second_plan = _make_plan([1, 2])
        assembler.prepare_shared_features(
            second_graph,
            second_plan,
            second_plan,
            n_src=1,
        )

        assert assembler._shared_gather_buffers["mem"].data.data_ptr() == mem_data_ptr
        assert assembler._shared_gather_buffers["box"].data.data_ptr() == box_data_ptr
        assert second_graph.mem is not None
        assert second_graph.box is not None
        assert second_graph.mem[0].tolist() == [101.0, 102.0]
        assert second_graph.box[0, 0].tolist() == [103.0, 104.0, 105.0]

    def test_shared_plan_reuses_layout_and_delta_order(self):
        store = _make_store(boxsize=1)
        assembler = _make_assembler(store)
        graph = _make_graph([1, 4, 2])
        shared_plan = _make_plan([1, 2])

        layout = assembler.prepare_shared_features(
            graph, shared_plan, shared_plan, n_src=1
        )
        delta = MemAndBox(
            mem=MemEntry(
                nid=torch.tensor([4]),
                data=torch.tensor([[40.0, 41.0]]),
                ts=torch.tensor([4.0]),
                is_sorted=True,
            ),
            box=BoxEntry(
                nid=torch.tensor([4]),
                data=torch.tensor([[[42.0, 43.0, 44.0]]]),
                ts=torch.tensor([[4.0]]),
                is_sorted=True,
            ),
        )

        assembler.apply_delta_features(
            graph,
            delta,
            shared_plan,
            shared_plan,
            n_src=1,
            layout=layout,
        )

        assert layout.mem is layout.box
        assert graph.mem is not None
        assert graph.box is not None
        assert graph.mem.tolist() == [
            store.mem[1].tolist(),
            [40.0, 41.0],
            store.mem[2].tolist(),
        ]
        assert graph.box[1].tolist() == [[42.0, 43.0, 44.0]]

    def test_four_h2d_slots_allow_n_plus_2_to_read_stale_shared_state(self):
        store = _make_store(boxsize=1)
        assembler = _make_assembler(store)
        graph = _make_graph([1])
        mem_plan = _make_plan([1])
        box_plan = _make_plan([1])

        old_mem = store.mem[1].clone()
        old_box = store.box[1].clone()

        h2d_sem = threading.Semaphore(4)
        for _ in range(2):
            h2d_sem.acquire()  # batch N
        for _ in range(2):
            h2d_sem.acquire()  # batch N+1

        # batch N+1 finishes first, so batch N+2 is allowed to start even
        # though batch N's writeback has not landed in shared memory yet.
        h2d_sem.release()
        h2d_sem.release()
        assert h2d_sem.acquire(blocking=False)
        assert h2d_sem.acquire(blocking=False)

        assembler.prepare_shared_features(graph, mem_plan, box_plan, n_src=1)

        store.mem[1] = torch.tensor([99.0, 101.0])
        store.mem_ts[1] = torch.tensor(77.0)
        store.box[1, 0] = torch.tensor([5.0, 6.0, 7.0])
        store.box_ts[1, 0] = torch.tensor(77.0)

        assert graph.mem is not None
        assert graph.mem_ts is not None
        assert graph.box is not None
        assert graph.box_ts is not None
        assert torch.equal(graph.mem[0], old_mem)
        assert torch.equal(graph.mem_ts[0], torch.tensor(1.0))
        assert torch.equal(graph.box[0], old_box)
        assert torch.equal(graph.box_ts[0], torch.tensor([1.0]))

    def test_logic_with_nothing_skips_remote_fill(self, monkeypatch):
        store = _make_store(boxsize=1)
        assembler = _make_logic_assembler(store)
        graph = _make_graph([1, 4])
        mem_plan = _make_plan([1])
        box_plan = _make_plan([1])
        meta = _make_meta(graph, mem_plan, box_plan)

        assembler.in_q.put((meta, FakeCudaEvent()))
        assembler.in_q.put(None)
        assembler.delta_queue.put("nothing")
        assembler.h2d_sem.release()
        assembler.h2d_sem.release()

        monkeypatch.setattr(torch.cuda, "current_stream", lambda: FakeStream())
        monkeypatch.setattr(torch.cuda, "Event", FakeCudaEvent)

        assembler.logic()

        full = assembler.full_q.get_nowait()
        assert isinstance(full, tuple)
        assert graph.mem is not None
        assert graph.box is not None
        assert torch.equal(graph.mem[0], store.mem[1])
        assert torch.equal(graph.mem[1], torch.zeros_like(store.mem[4]))
        assert torch.equal(graph.box[1], torch.zeros_like(store.box[4]))


class TestP2PQHandler:
    def test_keeps_nothing_sentinel(self):
        handler = _make_handler()

        handler.mem_p2p_q.put("nothing_to_p2p")
        handler.mem_p2p_q.put(None)
        handler.box_p2p_q.put("nothing_to_p2p")

        handler.logic()

        assert handler.delta_queue.get_nowait() == "nothing"
        assert handler.p2p_sem.acquire(blocking=False)

    def test_default_mode_still_uses_nccl_and_payload(self, monkeypatch):
        handler = _make_handler()
        mem_task = _make_p2p_task((2, 2))
        box_task = _make_p2p_task((2, 1, 3))
        calls = []

        handler.mem_p2p_q.put((mem_task, object()))
        handler.mem_p2p_q.put(None)
        handler.box_p2p_q.put((box_task, object()))

        monkeypatch.setattr(torch.cuda, "current_stream", lambda: FakeStream())

        def _fake_all_to_all_single(**kwargs):
            calls.append(kwargs)
            return FakeWork()

        monkeypatch.setattr(
            torch.distributed, "all_to_all_single", _fake_all_to_all_single
        )

        handler.logic()

        mem_out, box_out, works = handler.delta_queue.get_nowait()
        assert mem_out is mem_task
        assert box_out is box_task
        assert len(calls) == 6
        assert len(works) == 6
        assert handler.p2p_sem.acquire(blocking=False)

    def test_single_rank_bypass_synchronizes_stream_before_publishing_delta(
        self, monkeypatch
    ):
        handler = _make_handler()
        mem_task = _make_p2p_task((2, 2))
        box_task = _make_p2p_task((2, 1, 3))
        fake_stream = FakeStream()

        handler.mem_p2p_q.put((mem_task, object()))
        handler.mem_p2p_q.put(None)
        handler.box_p2p_q.put((box_task, object()))

        monkeypatch.setattr(torch.cuda, "current_stream", lambda: fake_stream)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 1)

        handler.logic()

        mem_out, box_out, works = handler.delta_queue.get_nowait()
        assert mem_out is mem_task
        assert box_out is box_task
        assert works == []
        assert fake_stream.synchronize_calls == 1
        assert handler.p2p_sem.acquire(blocking=False)
