"""Tests for debug-only dynamic planner plan dumps."""

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
