"""Equivalence tests for batched dynamic-state ownership routing."""

from __future__ import annotations

import threading
from queue import Queue

import numpy as np
import pytest
import torch

from frost.runtime.host.p2p_planner._dynamic_planner import (
    DynamicPlanner,
    _deduplicate_and_route_updates,
    compute_p2p_transfer_plan,
    deduplicate_updates,
    handle_remaining,
)
from frost.utils.tensor_set import TensorSet


def _ts(values: list[int]) -> TensorSet:
    return TensorSet(torch.tensor(values, dtype=torch.long))


def _values(tensor_set: TensorSet) -> list[int]:
    return tensor_set.data.tolist()


def _matrix_values(matrix: list[list[TensorSet]]) -> list[list[list[int]]]:
    return [[_values(cell) for cell in row] for row in matrix]


def _naive_transfer(
    needs: list[TensorSet], avails: list[TensorSet]
) -> tuple[list[list[int]], list[list[list[int]]]]:
    need_sets = [set(_values(value)) for value in needs]
    avail_sets = [set(_values(value)) for value in avails]
    routes = [[sorted(avail & need) for need in need_sets] for avail in avail_sets]
    all_needs = set().union(*need_sets)
    leftovers = [sorted(avail - all_needs) for avail in avail_sets]
    return leftovers, routes


def _naive_remaining(
    needs: list[TensorSet],
    avails: list[TensorSet],
    distances: np.ndarray,
) -> tuple[list[list[int]], list[list[list[int]]]]:
    remaining = [set(_values(value)) for value in needs]
    available = [set(_values(value)) for value in avails]
    n_gpus = len(needs)
    routes: list[list[set[int]]] = [
        [set() for _ in range(n_gpus)] for _ in range(n_gpus)
    ]
    for distance in sorted(np.unique(distances)):
        for sender, receiver in np.argwhere(distances == distance):
            routed = available[int(sender)] & remaining[int(receiver)]
            routes[int(sender)][int(receiver)].update(routed)
            remaining[int(receiver)].difference_update(routed)
    return (
        [sorted(values) for values in remaining],
        [[sorted(values) for values in row] for row in routes],
    )


def test_batched_transfer_preserves_overlapping_sender_semantics() -> None:
    needs = [_ts([1, 2, 4]), _ts([2, 3, 4]), _ts([7])]
    avails = [_ts([2, 4, 5]), _ts([1, 2, 3]), _ts([4, 6])]

    leftovers, routes = compute_p2p_transfer_plan(needs, avails)

    expected_leftovers, expected_routes = _naive_transfer(needs, avails)
    assert [_values(value) for value in leftovers] == expected_leftovers
    assert _matrix_values(routes) == expected_routes


@pytest.mark.parametrize("prefer_higher_rank", [False, True])
def test_batched_dedup_preserves_rank_preference(prefer_higher_rank: bool) -> None:
    updates = [_ts([1, 2, 5]), _ts([2, 3, 5]), _ts([1, 4, 5])]

    owned, union = deduplicate_updates(updates, prefer_higher_rank=prefer_higher_rank)

    assert _values(union) == [1, 2, 3, 4, 5]
    if prefer_higher_rank:
        assert [_values(value) for value in owned] == [[], [2, 3], [1, 4, 5]]
    else:
        assert [_values(value) for value in owned] == [[1, 2, 5], [3], [4]]


def test_batched_remaining_preserves_distance_and_rank_tie_breaks() -> None:
    needs = [_ts([1, 2, 3, 8]), _ts([1, 4, 5]), _ts([2, 4, 6])]
    avails = [_ts([1, 2, 4]), _ts([1, 3, 5]), _ts([2, 4, 6])]
    distances = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int32,
    )

    remaining, routes = handle_remaining(needs, avails, distances)

    expected_remaining, expected_routes = _naive_remaining(needs, avails, distances)
    assert [_values(value) for value in remaining] == expected_remaining
    assert _matrix_values(routes) == expected_routes


@pytest.mark.parametrize("n_gpus", [1, 2, 4])
def test_batched_routing_matches_pairwise_reference_randomized(n_gpus: int) -> None:
    rng = np.random.default_rng(20260801 + n_gpus)
    for _ in range(20):
        needs = [
            _ts(rng.choice(80, size=int(rng.integers(0, 30)), replace=False).tolist())
            for _ in range(n_gpus)
        ]
        avails = [
            _ts(rng.choice(80, size=int(rng.integers(0, 30)), replace=False).tolist())
            for _ in range(n_gpus)
        ]
        distances = rng.integers(0, 4, size=(n_gpus, n_gpus), dtype=np.int32)
        np.fill_diagonal(distances, 0)

        leftovers, fresh_routes = compute_p2p_transfer_plan(needs, avails)
        expected_leftovers, expected_fresh = _naive_transfer(needs, avails)
        assert [_values(value) for value in leftovers] == expected_leftovers
        assert _matrix_values(fresh_routes) == expected_fresh

        remaining, stale_routes = handle_remaining(needs, avails, distances)
        expected_remaining, expected_stale = _naive_remaining(needs, avails, distances)
        assert [_values(value) for value in remaining] == expected_remaining
        assert _matrix_values(stale_routes) == expected_stale


def test_batched_routing_handles_all_empty_sets() -> None:
    needs = [_ts([]), _ts([])]
    avails = [_ts([]), _ts([])]

    leftovers, routes = compute_p2p_transfer_plan(needs, avails)
    remaining, stale_routes = handle_remaining(
        needs, avails, np.array([[0, 1], [1, 0]], dtype=np.int32)
    )

    assert [_values(value) for value in leftovers] == [[], []]
    assert [_values(value) for value in remaining] == [[], []]
    assert _matrix_values(routes) == [[[], []], [[], []]]
    assert _matrix_values(stale_routes) == [[[], []], [[], []]]


def test_batched_transfer_with_empty_availability_leaves_needs_unrouted() -> None:
    needs = [_ts([1, 2]), _ts([3])]
    avails = [_ts([]), _ts([])]

    leftovers, routes = compute_p2p_transfer_plan(needs, avails)
    remaining, stale_routes = handle_remaining(
        needs, avails, np.array([[0, 1], [1, 0]], dtype=np.int32)
    )

    assert [_values(value) for value in leftovers] == [[], []]
    assert [_values(value) for value in remaining] == [[1, 2], [3]]
    assert _matrix_values(routes) == [[[], []], [[], []]]
    assert _matrix_values(stale_routes) == [[[], []], [[], []]]


def test_batched_transfer_rejects_mismatched_rank_counts() -> None:
    with pytest.raises(ValueError, match="equal rank counts"):
        compute_p2p_transfer_plan([_ts([1])], [_ts([1]), _ts([2])])


@pytest.mark.parametrize("node_offset", [0, 5_000_000])
@pytest.mark.parametrize("prefer_higher_rank", [False, True])
def test_native_owned_routes_match_reference_for_dense_and_sparse_ids(
    node_offset: int,
    prefer_higher_rank: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "frost.runtime.host.p2p_planner._dynamic_planner._NATIVE_ROUTE_MIN_ENTRIES",
        0,
    )
    updates = [
        _ts([node_offset + 1, node_offset + 3]),
        _ts([node_offset + 2, node_offset + 3]),
        _ts([node_offset + 1, node_offset + 4]),
    ]
    needs = [
        _ts([node_offset + 1, node_offset + 2, node_offset + 5]),
        _ts([node_offset + 3, node_offset + 4]),
        _ts([node_offset + 1, node_offset + 3]),
    ]

    owned, all_updates, routes = _deduplicate_and_route_updates(
        updates, needs, prefer_higher_rank=prefer_higher_rank
    )

    expected_owned, expected_union = deduplicate_updates(
        updates, prefer_higher_rank=prefer_higher_rank
    )
    _, expected_routes = _naive_transfer(needs, expected_owned)
    assert [_values(value) for value in owned] == [
        _values(value) for value in expected_owned
    ]
    assert _values(all_updates) == _values(expected_union)
    assert _matrix_values(routes) == expected_routes


def test_dynamic_planner_reuses_deduplicated_ownership_for_routes() -> None:
    planner = DynamicPlanner(
        gpus_distance=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
    )
    planner.fetch = [_ts([10]), _ts([20]), _ts([30])]

    plans = planner._plan(
        all_list=[_ts([1]), _ts([2]), _ts([3])],
        update_list=[_ts([1, 2]), _ts([2, 3]), _ts([1, 4])],
        next_all_list=[_ts([1, 2, 9]), _ts([3, 4]), _ts([1, 3])],
        next_update_list=[_ts([]), _ts([]), _ts([])],
        metas=[],
    )

    assert [_values(plan.fetch) for plan in plans] == [[10], [20], [30]]
    assert [_values(plan.writeback) for plan in plans] == [[], [2, 3], [1, 4]]
    assert _matrix_values([plan.p2p for plan in plans]) == [
        [[], [], []],
        [[2], [3], [3]],
        [[1], [4], [1]],
    ]
    assert [plan.send_counts.tolist() for plan in plans] == [
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert [plan.recv_counts.tolist() for plan in plans] == [
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
    ]
    assert [_values(value) for value in planner.fetch] == [[9], [], []]
