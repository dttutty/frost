from __future__ import annotations

import torch

from frost.runtime.worker import _feature_staging
from frost.runtime.worker._feature_staging import (
    FeatureStageRing,
    FeatureStageSlot,
    _encode_feature_route,
)


def _pageable(
    rows: int, columns: int, dtype: torch.dtype
) -> torch.Tensor:
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


def test_feature_route_encodes_fetched_cached_and_missing_rows() -> None:
    route = _encode_feature_route(
        torch.tensor([-1, 7, 4, 3, 1, 2]),
        torch.tensor([1, 4]),
        torch.tensor([2, 7]),
    )

    missing = torch.iinfo(torch.int64).min
    assert route.tolist() == [missing, -2, 1, missing, 0, -1]


def test_feature_stage_slot_reuses_route_buffers(monkeypatch) -> None:
    monkeypatch.setattr(_feature_staging, "_allocate_staging_tensor", _pageable)
    slot = FeatureStageSlot(index=0)

    fill, cache = slot.stage_node_routes(
        graph_nids=torch.tensor([4, 2, 7, -1]),
        fetched_nids=torch.tensor([2, 4]),
        cached_nids=torch.tensor([7]),
        to_cache_nids=torch.tensor([2, 7]),
    )
    fill_storage = slot.node_fill_route
    cache_storage = slot.node_cache_route
    assert fill.tolist()[:3] == [1, 0, -1]
    assert cache.tolist() == [0, -1]

    fill, cache = slot.stage_node_routes(
        graph_nids=torch.tensor([2, 7]),
        fetched_nids=torch.tensor([2]),
        cached_nids=torch.tensor([7]),
        to_cache_nids=torch.tensor([7]),
    )
    assert fill.tolist() == [0, -1]
    assert cache.tolist() == [-1]
    assert slot.node_fill_route is fill_storage
    assert slot.node_cache_route is cache_storage


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
