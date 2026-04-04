"""Tests for debug-only dynamic planner plan dumps."""

import logging
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
        cur_epoch=0,
        cur_macro_of_split=0,
        n_macro_of_split=1,
        cur_macro_of_epoch=0,
        split="train",
        n_src=1,
        mini_batch_idx=mini_batch_idx,
        flat_graph=None,
        buf_len=0,
        n_neg=1,
        fanout=[10],
    )


def test_dynamic_planner_appends_debug_dumps_per_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("FROST_DUMP_DYNAMIC_PLANS", "1")
    metas = [_meta(7), _meta(8)]

    mem_planner = DynamicPlanner(
        gpus_distance=np.array([[0, 1], [1, 0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        output_dir=str(tmp_path),
        plan_kind="mem",
    )
    mem_planner.fetch = [_ts([10]), _ts([20])]
    mem_planner._plan(
        all_list=[_ts([1, 2]), _ts([2, 3])],
        update_list=[_ts([1]), _ts([3])],
        next_all_list=[_ts([1, 3]), _ts([2])],
        next_update_list=[_ts([1]), _ts([])],
        metas=metas,
    )

    mail_planner = DynamicPlanner(
        gpus_distance=np.array([[0, 1], [1, 0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        output_dir=str(tmp_path),
        plan_kind="mail",
    )
    mail_planner.fetch = [_ts([30]), _ts([40])]
    mail_planner._plan(
        all_list=[_ts([4, 5]), _ts([5, 6])],
        update_list=[_ts([4]), _ts([6])],
        next_all_list=[_ts([4, 6]), _ts([5])],
        next_update_list=[_ts([4]), _ts([])],
        metas=metas,
    )

    dump0 = (tmp_path / "detail" / "7_plan.log").read_text(encoding="ascii")
    dump1 = (tmp_path / "detail" / "8_plan.log").read_text(encoding="ascii")

    assert "[mem]" in dump0
    assert "[mail]" in dump0
    assert "current_need: [1, 2]" in dump0
    assert "plan_fetch: [10]" in dump0
    assert "plan_send_counts: [1, 0]" in dump0
    assert "plan_p2p_to_gpu_0: [1]" in dump0

    assert "[mem]" in dump1
    assert "[mail]" in dump1
    assert "plan_fetch: [20]" in dump1
    assert "plan_writeback: [3]" in dump1


def test_dynamic_planner_debug_cpu_freshness_single_worker(monkeypatch):
    monkeypatch.setenv("FROST_DEBUG_CPU_FRESHNESS_SINGLE_WORKER", "1")

    planner = DynamicPlanner(
        gpus_distance=np.array([[0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        plan_kind="mem",
    )
    planner.fetch = [_ts([1, 2, 3])]

    plans = planner._plan(
        all_list=[_ts([1, 2, 3])],
        update_list=[_ts([2, 4, 4])],
        next_all_list=[_ts([4, 5])],
        next_update_list=[_ts([4])],
        metas=[_meta(9)],
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.fetch.data.tolist() == [1, 2, 3]
    assert plan.writeback.data.tolist() == [2, 4]
    assert plan.send_counts.tolist() == [0]
    assert plan.recv_counts.tolist() == [0]
    assert len(plan.p2p) == 1
    assert plan.p2p[0].data.numel() == 0
    assert planner.fetch[0].data.tolist() == [4, 5]


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
        handle_stale=False,
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


def test_dynamic_planner_logs_epoch_stats_with_self_and_cross_gpu_counts(caplog):
    planner = DynamicPlanner(
        gpus_distance=np.array([[0, 1], [1, 0]], dtype=np.int32),
        in_q=Queue(),
        out_q=Queue(),
        stop_ev=threading.Event(),
        plan_kind="mem",
    )

    planner._accumulate_stats(
        all_list=[_ts([1, 2]), _ts([2, 3, 4])],
        p2p_cts=np.array([[2, 5], [7, 11]], dtype=np.int32),
        writeback_list=[_ts([1, 2, 9]), _ts([3])],
    )

    with caplog.at_level(logging.INFO):
        planner._log_epoch_stats()

    text = caplog.text
    assert "[mem] epoch stats" in text
    assert "avg unique nodes (union across GPUs): 4.0" in text
    assert (
        "GPU 0: avg_unique=2.0  avg_p2p_recv_to_self=2.0  "
        "avg_p2p_recv_from_others=7.0  avg_p2p_send_to_self=2.0  "
        "avg_p2p_send_to_others=5.0  avg_writeback=3.0"
    ) in text
    assert (
        "GPU 1: avg_unique=3.0  avg_p2p_recv_to_self=11.0  "
        "avg_p2p_recv_from_others=5.0  avg_p2p_send_to_self=11.0  "
        "avg_p2p_send_to_others=7.0  avg_writeback=1.0"
    ) in text
