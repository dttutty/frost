"""Tests for dynamic planner writeback behavior."""
import threading
from queue import Queue

import numpy as np
import torch

from frost.datatypes.meta import Meta
from frost.runtime.host.p2p_planner._dynamic_planner import DynamicPlanner
from frost.utils.tensor_set import TensorSet


def _ts(vals: list[int]) -> TensorSet:
    return TensorSet(torch.tensor(vals, dtype=torch.long))


def _meta(mini_batch_idx: int) -> Meta:
    return Meta(
        split="train",
        n_src=1,
        mini_batch_idx=mini_batch_idx,
        flat_graph=None,
        buf_len=0,
        n_neg=1,
    )



def test_dynamic_planner_persist_all_updates_to_cpu_keeps_owned_updates():
    planner = DynamicPlanner(
        gpus_distance=np.array([[0, 1], [1, 0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        plan_kind="mem",
        persist_all_updates_to_cpu=True,
    )
    planner.fetch = [_ts([10]), _ts([20])]

    plans = planner._plan(
        all_list=[_ts([1, 2]), _ts([2, 3])],
        update_list=[_ts([1]), _ts([3])],
        next_all_list=[_ts([1, 3]), _ts([2])],
        next_update_list=[_ts([1]), _ts([])],
        metas=[_meta(10), _meta(11)],
    )

    assert plans[0].writeback is not None
    assert plans[1].writeback is not None
    assert plans[0].writeback.data.tolist() == [1]
    assert plans[1].writeback.data.tolist() == [3]


def test_dynamic_planner_without_stale_handling_keeps_all_owned_updates():
    planner = DynamicPlanner(
        gpus_distance=np.array([[0, 1], [1, 0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        plan_kind="mem",
        if_carry_over_route=False,
    )
    planner.fetch = [_ts([10]), _ts([20])]

    plans = planner._plan(
        all_list=[_ts([1, 2]), _ts([2, 3])],
        update_list=[_ts([1]), _ts([3])],
        next_all_list=[_ts([1, 3]), _ts([2])],
        next_update_list=[_ts([1]), _ts([])],
        metas=[_meta(12), _meta(13)],
    )

    assert plans[0].writeback is not None
    assert plans[1].writeback is not None
    assert plans[0].writeback.data.tolist() == [1]
    assert plans[1].writeback.data.tolist() == [3]


def test_dynamic_planner_current_batch_only_uses_dedup_fetch_writeback_and_no_p2p():
    planner = DynamicPlanner(
        gpus_distance=np.array([[0, 1], [1, 0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        plan_kind="mem",
        current_batch_only=True,
    )

    plans = planner._plan_current_batch_only(
        all_list=[_ts([1, 2, 3]), _ts([3, 4])],
        update_list=[_ts([1, 1, 2]), _ts([4, 4])],
    )

    assert plans[0].fetch.data.tolist() == [1, 2, 3]
    assert plans[1].fetch.data.tolist() == [3, 4]
    assert plans[0].writeback is not None
    assert plans[1].writeback is not None
    assert plans[0].writeback.data.tolist() == [1, 2]
    assert plans[1].writeback.data.tolist() == [4]
    assert plans[0].recv_counts.tolist() == [0, 0]
    assert plans[0].send_counts.tolist() == [0, 0]
    assert all(ts.data.numel() == 0 for ts in plans[0].p2p)
    assert all(ts.data.numel() == 0 for ts in plans[1].p2p)
