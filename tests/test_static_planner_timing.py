import threading
from queue import Queue

import numpy as np
import torch

from frost.runtime.host.p2p_planner._static_planner import StaticPlanner
from frost.utils.tensor_set import TensorSet


def _ts(values: list[int]) -> TensorSet:
    return TensorSet(torch.tensor(values, dtype=torch.long))


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

