import threading
from queue import Queue

import numpy as np
import torch

from frost.runtime.host.p2p_planner._static_planner import StaticPlanner
from frost.utils.tensor_set import TensorSet


def _ts(values: list[int]) -> TensorSet:
    return TensorSet(torch.tensor(values, dtype=torch.long))


def _unsorted_ts(values: list[int]) -> TensorSet:
    return TensorSet(
        torch.tensor(values, dtype=torch.long),
        sort_now=False,
        ensure_unique=False,
    )


def test_static_planner_records_feature_cache_counts(monkeypatch):
    monkeypatch.setenv("FROST_HOST_TIMING_BENCH", "1")
    planner = StaticPlanner(
        gpus_distance=np.array([[0, 1], [1, 0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
    )
    planner.fetchs = [_ts([1, 2, 3]), _ts([10, 11])]

    plans = planner._plan(
        avails_of_1=[_ts([1, 2, 3, 4]), _ts([10, 11, 12])],
        needs_of_2=[_ts([2, 4, 5]), _ts([11, 13])],
    )

    assert [plan.fetch.data.tolist() for plan in plans] == [[1, 2, 3], [10, 11]]
    assert planner._need_counts_curve == [[4, 3]]
    assert planner._fetch_counts_curve == [[3, 2]]
    assert planner._cache_hit_counts_curve == [[1, 1]]
    assert planner._next_cache_counts_curve == [[2, 1]]


def test_static_planner_keeps_rows_reused_after_one_batch_gap():
    in_q: Queue = Queue()
    out_q: Queue = Queue()
    planner = StaticPlanner(
        gpus_distance=np.array([[0]], dtype=np.int32),
        in_q=in_q,
        out_q=out_q,
        stop_ev=threading.Event(),
        cache_horizon=2,
    )
    in_q.put([_ts([1, 2])])
    in_q.put([_ts([2, 3])])
    in_q.put([_ts([1, 3, 4])])
    in_q.put("end_of_epoch")
    in_q.put(None)

    planner.logic()

    plans = [out_q.get_nowait()[0] for _ in range(3)]
    assert out_q.get_nowait() is None
    assert [plan.fetch.data.tolist() for plan in plans] == [[1, 2], [3], [4]]
    assert [plan.to_cache.data.tolist() for plan in plans] == [
        [1, 2],
        [1, 3],
        [],
    ]
    assert planner._active_cache_horizon == 2


def test_static_planner_default_horizon_keeps_longer_lived_rows():
    in_q: Queue = Queue()
    out_q: Queue = Queue()
    planner = StaticPlanner(
        gpus_distance=np.array([[0]], dtype=np.int32),
        in_q=in_q,
        out_q=out_q,
        stop_ev=threading.Event(),
        large_batch_need_rows=1,
        very_large_batch_need_rows=100,
    )
    in_q.put([_ts([1, 2])])
    for value in range(3, 10):
        in_q.put([_ts([value])])
    in_q.put([_ts([1, 10])])
    in_q.put("end_of_epoch")
    in_q.put(None)

    planner.logic()

    plans = [out_q.get_nowait()[0] for _ in range(9)]
    assert out_q.get_nowait() is None
    assert plans[0].to_cache.data.tolist() == [1]
    assert plans[-1].fetch.data.tolist() == [10]
    assert planner._active_cache_horizon == 8


def test_static_planner_limits_reuse_for_very_large_batches():
    planner = StaticPlanner(
        gpus_distance=np.array([[0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        large_batch_need_rows=2,
        very_large_batch_need_rows=4,
    )
    planner.in_q.put([_ts([1, 2, 3, 4])])
    planner.in_q.put("end_of_epoch")
    planner.in_q.put(None)

    planner.logic()

    assert planner._active_cache_horizon == 2


def test_static_planner_handles_unsorted_native_need_sets():
    planner = StaticPlanner(
        gpus_distance=np.array([[0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        cache_horizon=2,
    )
    planner.cached = [_ts([2, 8])]
    planner.fetchs = [_unsorted_ts([9, 1, 4])]

    plans = planner._plan_horizon(
        [_unsorted_ts([9, 1, 4])],
        [[_unsorted_ts([8, 4, 3])], [_unsorted_ts([2, 9, 5])]],
    )

    assert plans[0].to_cache.data.tolist() == [2, 4, 8, 9]
    assert planner.fetchs[0].data.tolist() == [3]
