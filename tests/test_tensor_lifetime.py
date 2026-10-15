from __future__ import annotations

import numpy as np
import pytest
import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, PlanSet, StaticPlan
from frost.runtime.worker import _tensor_lifetime
from frost.utils.tensor_set import TensorSet


def _meta(*, graph: FlatGraph, plan_set: PlanSet | None = None) -> Meta:
    return Meta(
        split="train",
        n_src=1,
        mini_batch_idx=0,
        flat_graph=graph,
        buf_len=1,
        n_neg=1,
        plan_set=plan_set,
    )


def test_record_batch_stream_visits_graph_labels_and_shared_plans(monkeypatch) -> None:
    shared = torch.tensor([1])
    graph = FlatGraph(nid=shared, eid=torch.tensor([2]), ts=shared)
    dynamic = DynamicPlan(
        p2p=[TensorSet(torch.tensor([3]), sort_now=False)],
        fetch=TensorSet(torch.tensor([4]), sort_now=False),
        recv_counts=np.array([0]),
        send_counts=np.array([0]),
        writeback=TensorSet(torch.tensor([5]), sort_now=False),
        fetch_on_gpu=torch.tensor([6]),
    )
    static = StaticPlan(
        fetch=TensorSet(torch.tensor([7]), sort_now=False),
        to_cache=TensorSet(torch.tensor([8]), sort_now=False),
    )
    meta = _meta(
        graph=graph,
        plan_set=PlanSet(mem_plan=dynamic, box_plan=dynamic, nfeat_plan=static),
    )
    meta.labels = shared
    meta.neg_valid_mask = torch.tensor([True])
    meta.flat_nid_cpu = torch.tensor([9])

    visited = []
    monkeypatch.setattr(
        _tensor_lifetime,
        "_record_if_cuda",
        lambda tensor, _: visited.append(tensor),
    )
    _tensor_lifetime.record_batch_stream(meta, object())

    expected = [
        shared,
        graph.eid,
        dynamic.p2p[0].data,
        dynamic.fetch.data,
        dynamic.writeback.data,
        dynamic.fetch_on_gpu,
        static.fetch.data,
        static.to_cache.data,
        meta.neg_valid_mask,
        meta.flat_nid_cpu,
    ]
    assert {id(value) for value in visited} == {id(value) for value in expected}
    assert len(visited) == len(expected)


def test_record_batch_stream_ignores_cpu_tensors(monkeypatch) -> None:
    def unexpected_record_stream(self, stream):
        raise AssertionError("record_stream called for a CPU tensor")

    monkeypatch.setattr(torch.Tensor, "record_stream", unexpected_record_stream)
    graph = FlatGraph(
        nid=torch.tensor([1]), eid=torch.tensor([2]), ts=torch.tensor([3])
    )
    _tensor_lifetime.record_batch_stream(_meta(graph=graph), object())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_producer_storage_survives_delayed_consumer_and_allocator_pressure() -> None:
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA sleep kernel is unavailable")

    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    count = 1 << 18
    for _ in range(3):
        with torch.cuda.stream(producer):
            source = torch.full((count,), 7.0, device="cuda")
            ready = torch.cuda.Event()
            ready.record()

        meta = _meta(graph=FlatGraph(nid=source, eid=source, ts=source))
        del source
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            _tensor_lifetime.record_batch_stream(meta, consumer)
            torch.cuda._sleep(100_000_000)
            output = torch.empty((count,), device="cuda")
            output.copy_(meta.flat_graph.nid)

        meta.flat_graph = None
        with torch.cuda.stream(producer):
            pressure = [torch.full((count,), -3.0, device="cuda") for _ in range(8)]

        consumer.synchronize()
        producer.synchronize()
        assert torch.all(output == 7.0).item()
        del pressure, output
