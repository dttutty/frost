from __future__ import annotations

import torch

from frost.runtime.worker import _feature_staging
from frost.runtime.worker._feature_staging import FeatureStageRing, FeatureStageSlot


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
