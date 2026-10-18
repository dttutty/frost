from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.meta import Meta
from frost.datatypes.plan import PlanSet, StaticPlan
from frost.runtime.worker import _feature_staging
from frost.runtime.worker._feature_staging import (
    FeaturePrefetcher,
    FeatureStageRing,
    FeatureStageSlot,
    StagedFeatureBatch,
)
from frost.utils.tensor_set import TensorSet


def _pageable(rows: int, columns: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((rows, columns), dtype=dtype)


def test_feature_stage_slot_reuses_node_buffer(monkeypatch) -> None:
    monkeypatch.setattr(_feature_staging, "_allocate_staging_tensor", _pageable)
    table = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    slot = FeatureStageSlot(index=0)

    first = slot.stage_node_rows(table, torch.tensor([3, 1, 8]))
    assert first is not None
    assert torch.equal(first, table[[3, 1, 8]])
    storage = slot.node_values

    second = slot.stage_node_rows(table, torch.tensor([2, 4]))
    assert torch.equal(second, table[[2, 4]])
    assert slot.node_values is storage


def test_feature_stage_slot_parallel_node_gather_preserves_order(monkeypatch) -> None:
    monkeypatch.setattr(_feature_staging, "_allocate_staging_tensor", _pageable)
    table = torch.arange(7000 * 4, dtype=torch.float32).reshape(7000, 4)
    row_ids = torch.randperm(7000)[:5000]
    slot = FeatureStageSlot(index=0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        staged = slot.stage_node_rows(table, row_ids, executor=executor)

    assert staged is not None
    assert torch.equal(staged, table[row_ids])


def test_feature_stage_slot_packs_only_consumed_edge_rows(monkeypatch) -> None:
    monkeypatch.setattr(_feature_staging, "_allocate_staging_tensor", _pageable)
    table = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    slot = FeatureStageSlot(index=0)
    edge_ids = torch.tensor([1, 2, -1, 3, -1])

    values, positions = slot.stage_edge_rows(table, edge_ids, n_src=1)

    assert positions.tolist() == [0, 3]
    assert torch.equal(values, table[[1, 3]])


def test_feature_stage_ring_has_independent_reusable_slots() -> None:
    ring = FeatureStageRing(depth=2)
    first = ring.acquire()
    second = ring.acquire()
    assert first.index != second.index

    ring.release(first)
    assert ring.acquire() is first


def test_feature_prefetch_timing_reports_static_work(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FROST_PIPELINE_TIMING_BENCH", "1")
    monkeypatch.setattr(_feature_staging, "_allocate_staging_tensor", _pageable)
    in_q: Queue = Queue()
    out_q: Queue = Queue()
    plan = StaticPlan(
        fetch=TensorSet(torch.tensor([1, 3])),
        to_cache=TensorSet(torch.tensor([3])),
    )
    meta = Meta(
        split="train",
        n_src=1,
        mini_batch_idx=0,
        flat_graph=FlatGraph(
            nid=torch.tensor([1, 3, -1]),
            eid=torch.full((3,), -1),
            ts=torch.zeros(3, dtype=torch.long),
        ),
        buf_len=3,
        n_neg=1,
        plan_set=PlanSet(nfeat_plan=plan),
    )
    in_q.put(meta)
    in_q.put(None)
    prefetcher = FeaturePrefetcher(
        nfeat=torch.arange(20, dtype=torch.float32).reshape(5, 4),
        efeat=None,
        in_q=in_q,
        out_q=out_q,
        ring=FeatureStageRing(depth=1),
        stop_ev=threading.Event(),
    )

    prefetcher.logic()

    staged = out_q.get_nowait()
    assert isinstance(staged, StagedFeatureBatch)
    assert torch.equal(staged.node_values, prefetcher.nfeat[[1, 3]])
    marker, encoded = capsys.readouterr().out.strip().split(" ", 1)
    assert marker == "APPENDIX_FEATURE_PREFETCH_TIMING_CURVE"
    payload = json.loads(encoded)
    assert payload["flat_node_rows_curve"] == [3]
    assert payload["valid_node_rows_curve"] == [2]
    assert payload["node_fetch_rows_curve"] == [2]
    assert payload["next_cache_rows_curve"] == [1]
