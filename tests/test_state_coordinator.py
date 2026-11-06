"""Tests for runtime-owned dynamic-state publication."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from frost.datatypes.graph import FlatGraph
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.meta import Meta
from frost.launch.plan import compile_execution_plan
from frost.models.contracts import RepresentationKind, RootExpansion
from frost.models.outputs import EdgeRepresentations, EncoderResult
from frost.models.recipes import TGNRecipe
from frost.models.temporal_link_model import TemporalLinkModel
from frost.runtime.dynamic_state_policy import resolve_dynamic_state_policy
from frost.runtime.worker._state_coordinator import DynamicStateCoordinator
from frost.utils.tensor_set import TensorSet


class _FakeMemUpdater(nn.Module):
    def forward(self, flat_graph, *_args, **_kwargs):
        flat_graph.detached_nid = torch.tensor([1, 2], dtype=torch.long)
        flat_graph.detached_mem = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        )
        flat_graph.detached_ts = torch.tensor([5.0, 5.0], dtype=torch.float32)
        return flat_graph


class _MisalignedMemUpdater(nn.Module):
    def forward(self, flat_graph, *_args, **_kwargs):
        flat_graph.detached_nid = torch.tensor([1, 2], dtype=torch.long)
        flat_graph.detached_mem = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        flat_graph.detached_ts = torch.tensor([5.0, 5.0], dtype=torch.float32)
        flat_graph.mem = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        return flat_graph


class _FakeEncoder(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def forward(self, graph):
        del graph
        self.events.append("encoder")
        return EncoderResult(torch.ones((3, 1), dtype=torch.float32))


class _IdentityHead(nn.Module):
    def forward(self, edges: EdgeRepresentations):
        return torch.cat([edges.positive_source, edges.negative_source])


def _plan():
    recipe = TGNRecipe(
        embedding_dim=2,
        neighborhood={"fanout": [1]},
        encoder={},
        state={},
        head={},
    )
    return compile_execution_plan(recipe)


def _make_meta() -> Meta:
    graph = FlatGraph(
        nid=torch.tensor([1, 2, 3], dtype=torch.long),
        eid=torch.tensor([0, -1, -1], dtype=torch.long),
        ts=torch.tensor([5.0, 5.0, 5.0]),
        efeat=torch.tensor([[0.5], [0.0], [0.0]]),
    )
    return Meta(
        split="train",
        n_src=1,
        mini_batch_idx=7,
        flat_graph=graph,
        buf_len=3,
        n_neg=1,
        plan_set=None,
    )


def _model(events: list[str]) -> TemporalLinkModel:
    return TemporalLinkModel(
        encoder=_FakeEncoder(events),
        mem_updater=_FakeMemUpdater(),
        head=_IdentityHead(),
        representation=RepresentationKind.NODE_ALIGNED,
        fanout=(),
        root_expansion=RootExpansion.ALL_ROOTS,
    )


def test_planned_p2p_union_deduplicates_once_across_peers() -> None:
    plan = SimpleNamespace(
        p2p=[
            TensorSet(torch.tensor([5, 1, 3], dtype=torch.long)),
            TensorSet(torch.tensor([7, 3, 1], dtype=torch.long)),
        ]
    )

    actual = DynamicStateCoordinator._union_planned_p2p_nids(plan)

    assert torch.equal(actual, torch.tensor([1, 3, 5, 7], dtype=torch.long))


def test_single_peer_union_reuses_sorted_tensor() -> None:
    peer = TensorSet(torch.tensor([5, 1, 3], dtype=torch.long))
    actual = DynamicStateCoordinator._union_planned_p2p_nids(
        SimpleNamespace(p2p=[peer])
    )
    assert actual is peer.data


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "policy_name",
    [
        "no_plan_no_delta_post_step_raw",
        "no_plan_no_delta_post_mem_update_raw",
        "plan_no_delta_dedup_post_mem_update_raw",
    ],
)
def test_direct_payload_deduplication_preserves_interaction_pairing(
    device, policy_name
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    policy = resolve_dynamic_state_policy(policy_name)
    captured = []
    coordinator = DynamicStateCoordinator(
        plan=_plan(), dynamic_state_policy=policy, carry_over_route=False,
        mem_task_q=None, box_task_q=None, direct_writeback=captured.append,
    )
    meta = _make_meta()
    meta.n_src = 3
    meta.plan_set = SimpleNamespace(
        mem_plan=SimpleNamespace(writeback=TensorSet(torch.tensor([1, 3]))),
        box_plan=SimpleNamespace(writeback=TensorSet(torch.tensor([1, 2]))),
    )
    graph = meta.flat_graph
    graph.detached_nid = torch.tensor([1, 1, 3, 2, 2, 1], device=device)
    graph.detached_ts = torch.tensor([3., 5., 4., 3., 5., 4.], device=device)
    graph.detached_mem = torch.arange(12, device=device, dtype=torch.float32).view(6, 2)
    graph.efeat = torch.tensor([[10.], [20.], [30.]], device=device)
    original = graph.detached_mem.clone()
    with coordinator.batch(meta):
        coordinator._after_state_update(None, (), graph)
    request = captured[0] if captured else meta.direct_writeback_request
    mem, box = request.mem_entry, request.box_entry
    assert mem.data.device.type == box.data.device.type == device
    assert request.mem_filter_nids is request.box_filter_nids is None
    assert torch.equal(graph.detached_mem, original)
    if policy.deduplicate_writeback:
        assert mem.nid.tolist() == [1, 3]
        assert mem.data.tolist() == [[2., 3.], [4., 5.]]
        assert box.nid.tolist() == [1, 2]
        assert box.data[:, 0].tolist() == [
            [2., 3., 8., 9., 20.], [8., 9., 2., 3., 20.]
        ]
    else:
        assert mem.nid.tolist() == box.nid.tolist() == [1, 1, 3, 2, 2, 1]
        assert mem.data.shape[0] == box.data.shape[0] == 6


@pytest.mark.parametrize(
    "policy_name,expected",
    [
        ("no_plan_no_delta_post_step_raw", ["encoder", "backward", "step", "write"]),
        ("no_plan_no_delta_post_mem_update_raw", ["write", "encoder", "backward", "step"]),
    ],
)
def test_direct_writeback_timing_in_actual_training_step(policy_name, expected):
    from frost.runtime.worker._trainer import WorkerTrainerMixin

    events = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.))
            self.updater = _FakeMemUpdater()

        def forward(self, meta):
            self.updater(meta.flat_graph)
            events.append("encoder")
            return self.weight.expand(2, 1)

    class Trainer(WorkerTrainerMixin):
        def _release_meta_tensors(self, _meta):
            pass

    trainer = Trainer()
    trainer.model = Model()
    trainer.model.weight.register_hook(lambda _grad: events.append("backward"))
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.optimizer.register_step_post_hook(lambda *_: events.append("step"))
    trainer.dynamic_state_policy = resolve_dynamic_state_policy(policy_name)
    trainer.direct_state_writer = SimpleNamespace(write=lambda *_args, **_kw: events.append("write"))
    trainer.state_coordinator = DynamicStateCoordinator(
        plan=_plan(), dynamic_state_policy=trainer.dynamic_state_policy,
        carry_over_route=False, mem_task_q=None, box_task_q=None,
        direct_writeback=trainer._write_direct_dynamic_state_request,
    )
    trainer.state_coordinator.attach(trainer.model.updater)
    trainer.amp = False
    trainer.mem_box_store = None
    trainer.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.train_loss_tracker = SimpleNamespace(update=lambda *_: None)
    trainer.batch_convergence = SimpleNamespace(record=lambda **_: None)
    meta = _make_meta()
    meta.labels = torch.tensor([1., 0.])
    trainer.train(meta)
    assert events == expected
    assert meta.direct_writeback_request is None


@pytest.mark.parametrize(
    ("policy_name", "carry_over_route", "expect_stale"),
    [
        ("baseline", False, False),
        ("baseline", True, True),
        ("no_plan_no_delta_post_step_raw", True, False),
        ("no_plan_no_delta_post_mem_update_raw", True, False),
        ("plan_no_delta_dedup_post_mem_update_raw", True, False),
    ],
)
def test_stale_capture_only_for_async_carry_over(
    policy_name: str, carry_over_route: bool, expect_stale: bool
) -> None:
    meta = _make_meta()
    graph = meta.flat_graph
    assert graph is not None
    graph.mem = torch.tensor([[1.0], [2.0], [3.0]])
    graph.mem_ts = torch.tensor([1.0, 2.0, 3.0])
    graph.box = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    graph.box_ts = torch.tensor([[1.0], [2.0], [3.0]])
    if expect_stale:
        plan = SimpleNamespace(p2p=[TensorSet(torch.tensor([3]))])
        meta.plan_set = SimpleNamespace(mem_plan=plan, box_plan=plan)
    coordinator = DynamicStateCoordinator(
        plan=_plan(),
        dynamic_state_policy=resolve_dynamic_state_policy(policy_name),
        carry_over_route=carry_over_route,
        mem_task_q=None,
        box_task_q=None,
        direct_writeback=lambda _request: None,
    )

    stale_mem, stale_box = coordinator._capture_stale_state(meta)

    if expect_stale:
        assert stale_mem is not None and stale_box is not None
        assert stale_mem.nid.tolist() == [3]
        assert stale_mem.data.tolist() == [[3.0]]
        assert stale_box.nid.tolist() == [3]
        assert stale_box.data.tolist() == [[[30.0]]]
    else:
        assert stale_mem is None and stale_box is None


def test_post_mem_update_mode_writes_before_encoder() -> None:
    events: list[str] = []
    requests = []
    model = _model(events)
    coordinator = DynamicStateCoordinator(
        plan=_plan(),
        dynamic_state_policy=resolve_dynamic_state_policy(
            "no_plan_no_delta_post_mem_update_raw"
        ),
        carry_over_route=False,
        mem_task_q=None,
        box_task_q=None,
        direct_writeback=lambda request: (
            events.append("write"),
            requests.append(request),
        ),
    )
    coordinator.attach(model.mem_updater)
    meta = _make_meta()

    with coordinator.batch(meta):
        model(meta)

    assert events == ["write", "encoder"]
    assert len(requests) == 1
    assert meta.direct_writeback_request is None


def test_post_step_mode_defers_writeback_until_worker_step() -> None:
    events: list[str] = []
    model = _model(events)
    coordinator = DynamicStateCoordinator(
        plan=_plan(),
        dynamic_state_policy=resolve_dynamic_state_policy(
            "no_plan_no_delta_post_step_raw"
        ),
        carry_over_route=False,
        mem_task_q=None,
        box_task_q=None,
        direct_writeback=lambda _request: events.append("write"),
    )
    coordinator.attach(model.mem_updater)
    meta = _make_meta()

    with coordinator.batch(meta):
        model(meta)

    assert events == ["encoder"]
    assert meta.direct_writeback_request is not None
    assert meta.direct_writeback_request.mini_batch_idx == 7


def test_coordinator_rejects_misaligned_updater_state_synchronously() -> None:
    events: list[str] = []
    model = TemporalLinkModel(
        encoder=_FakeEncoder(events),
        mem_updater=_MisalignedMemUpdater(),
        head=_IdentityHead(),
        representation=RepresentationKind.NODE_ALIGNED,
        fanout=(),
        root_expansion=RootExpansion.ALL_ROOTS,
    )
    coordinator = DynamicStateCoordinator(
        plan=_plan(),
        dynamic_state_policy=resolve_dynamic_state_policy(
            "no_plan_no_delta_post_mem_update_raw"
        ),
        carry_over_route=False,
        mem_task_q=None,
        box_task_q=None,
        direct_writeback=lambda _request: None,
    )
    coordinator.attach(model.mem_updater)
    meta = _make_meta()

    with coordinator.batch(meta), pytest.raises(
        RuntimeError, match="misaligned positive-root state"
    ):
        model(meta)


def test_async_publication_snapshots_box_input_before_memory_queue_put(
    monkeypatch,
) -> None:
    class _FakeEvent:
        def record(self, *, stream) -> None:
            del stream

    class _MutatingMemQueue:
        def put(self, item) -> None:
            task, _event = item
            # Model the in-place field replacement performed by keep_last().
            task.root_new_mem.data = task.root_new_mem.data[:1]

    class _RecordingBoxQueue:
        def __init__(self) -> None:
            self.item = None

        def put(self, item) -> None:
            self.item = item

    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    mem_q = _MutatingMemQueue()
    box_q = _RecordingBoxQueue()
    coordinator = DynamicStateCoordinator(
        plan=_plan(),
        dynamic_state_policy=resolve_dynamic_state_policy("baseline"),
        carry_over_route=False,
        mem_task_q=mem_q,
        box_task_q=box_q,
        direct_writeback=lambda _request: None,
    )
    meta = _make_meta()
    meta.plan_set = SimpleNamespace(mem_plan=object(), box_plan=object())
    context = SimpleNamespace(meta=meta, stale_mem=None, stale_box=None)
    root_new_mem = MemEntry(
        nid=torch.tensor([1, 2]),
        data=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ts=torch.tensor([5.0, 5.0]),
    )

    coordinator._publish_async(context, root_new_mem, None, None, None)

    assert box_q.item is not None
    box_task, _event = box_q.item
    assert box_task.root_new_mem.nid.shape == (2,)
    assert box_task.root_new_mem.data.shape == (2, 2)
    assert box_task.root_new_mem.ts.shape == (2,)
