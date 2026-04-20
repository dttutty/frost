from types import SimpleNamespace

import torch
import torch.nn as nn

from frost.ablation import (
    DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_MEM_UPDATE_RAW,
    DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_STEP_RAW,
    resolve_dynamic_state_ablation,
)
from frost.datatypes.meta import Meta
from frost.datatypes.plan import PlanSet
from frost.models._assembled_model import AssembledModel


class _FakeFlatGraph:
    def __init__(self) -> None:
        self.nid = torch.tensor([1, 2], dtype=torch.long)
        self.efeat = torch.tensor([[0.5]], dtype=torch.float32)
        self.mem = None
        self.mem_ts = None
        self.box = None
        self.box_ts = None
        self.detached_nid = None
        self.detached_mem = None
        self.detached_ts = None

    def unflatten(self, **_kwargs):
        return SimpleNamespace()


class _FakeMemUpdater(nn.Module):
    def forward(self, flat_graph, *_args, **_kwargs):
        flat_graph.detached_nid = torch.tensor([1, 2], dtype=torch.long)
        flat_graph.detached_mem = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        )
        flat_graph.detached_ts = torch.tensor([5.0, 5.0], dtype=torch.float32)
        return flat_graph


class _FakeEncoder(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def forward(self, graph):
        del graph
        self.events.append("encoder")
        return torch.ones((1, 1), dtype=torch.float32)


class _IdentityHeader(nn.Module):
    def forward(self, h):
        return h


def _make_meta() -> Meta:
    return Meta(
        cur_epoch=0,
        cur_macro_of_split=0,
        n_macro_of_split=1,
        cur_macro_of_epoch=0,
        split="train",
        n_src=1,
        mini_batch_idx=7,
        flat_graph=_FakeFlatGraph(),
        buf_len=2,
        n_neg=1,
        graph=None,
        plan_set=PlanSet(),
        fanout=[],
    )


def test_post_mem_update_mode_writes_before_encoder():
    events: list[str] = []
    requests = []
    model = AssembledModel(
        encoder=_FakeEncoder(events),
        mem_updater=_FakeMemUpdater(),
        header=_IdentityHeader(),
        postman=None,
        n_neg=1,
        sample_4_neg=True,
        dynamic_state_ablation=resolve_dynamic_state_ablation(
            DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_MEM_UPDATE_RAW
        ),
    )
    meta = _make_meta()

    model(
        meta,
        direct_writeback=lambda request: (
            events.append("write"),
            requests.append(request),
        ),
    )

    assert events == ["write", "encoder"]
    assert len(requests) == 1
    assert meta.direct_writeback_request is None


def test_post_step_mode_defers_writeback_until_worker_step():
    events: list[str] = []
    model = AssembledModel(
        encoder=_FakeEncoder(events),
        mem_updater=_FakeMemUpdater(),
        header=_IdentityHeader(),
        postman=None,
        n_neg=1,
        sample_4_neg=True,
        dynamic_state_ablation=resolve_dynamic_state_ablation(
            DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_STEP_RAW
        ),
    )
    meta = _make_meta()

    model(
        meta,
        direct_writeback=lambda _request: events.append("write"),
    )

    assert events == ["encoder"]
    assert meta.direct_writeback_request is not None
    assert meta.direct_writeback_request.mini_batch_idx == 7
