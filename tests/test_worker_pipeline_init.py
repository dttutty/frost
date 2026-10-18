from __future__ import annotations

import threading
from types import SimpleNamespace

import torch

from frost.models.contracts import NodeSet, RootExpansion
from frost.runtime.experimental.dynamic_state import resolve_dynamic_state_policy
from frost.runtime.worker import _pipeline
from frost.runtime.worker._pipeline import WorkerPipelineMixin


class _PipelineHarness(WorkerPipelineMixin):
    pass


class _CaptureDynamicAssembler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_direct_state_ablation_initializes_delta_queue(monkeypatch):
    monkeypatch.setattr(_pipeline, "Receiver", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(_pipeline, "H2D", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(_pipeline, "DynamicAssembler", _CaptureDynamicAssembler)

    worker = _PipelineHarness()
    worker.mem_box_store = object()
    worker.dynamic_state_policy = resolve_dynamic_state_policy(
        "no_plan_no_delta_post_step_raw"
    )
    worker.no_memory_queue_size = 4
    worker.env = SimpleNamespace()
    worker.stop_ev = threading.Event()
    worker.device = torch.device("cpu")
    worker.n_neg = 1
    worker.h2d_sem = object()
    worker.delta_queue = None
    worker.h2d_out_q = None
    worker.mem_task_q = None
    worker.box_task_q = None
    worker.mem_wr_q = None
    worker.box_wr_q = None
    worker.execution_plan = SimpleNamespace(
        node_feature_scope=NodeSet.ALL_SAMPLED
    )

    worker._init_pipeline(
        ring_buf=SimpleNamespace(root_expansion=RootExpansion.ALL_ROOTS),
        control_plane=object(),
        nfeat=None,
        efeat=None,
    )

    assert worker.delta_queue is not None
    assert worker.dy_asm.kwargs["delta_queue"] is worker.delta_queue
