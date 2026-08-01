"""Tests for PlanSet, DynamicPlan, and StaticPlan datatypes."""

import numpy as np
import torch

from frost.datatypes.plan import DynamicPlan, PlanSet, StaticPlan
from frost.utils.tensor_set import TensorSet


def _ts(vals: list[int]) -> TensorSet:
    return TensorSet(torch.tensor(vals, dtype=torch.long))


class TestStaticPlan:
    def test_basic_creation(self):
        to_cache = _ts([1, 2, 3])
        fetch = _ts([1, 2, 3, 4, 5])
        plan = StaticPlan(to_cache=to_cache, fetch=fetch)
        assert plan.to_cache.data.tolist() == [1, 2, 3]
        assert plan.fetch.data.tolist() == [1, 2, 3, 4, 5]

    def test_to_device(self):
        to_cache = _ts([1, 2, 3])
        fetch = _ts([1, 2, 3, 4])
        plan = StaticPlan(to_cache=to_cache, fetch=fetch)
        plan_cpu = plan.to(torch.device("cpu"))
        assert plan_cpu.to_cache.data.device.type == "cpu"
        assert plan_cpu.fetch.data.device.type == "cpu"

    def test_repr(self):
        to_cache = _ts([1, 2])
        fetch = _ts([1, 2, 3])
        plan = StaticPlan(to_cache=to_cache, fetch=fetch)
        r = repr(plan)
        assert "to_cache" in r
        assert "fetch" in r


class TestDynamicPlan:
    def test_basic_creation(self):
        p2p = [_ts([1, 2]), _ts([3, 4])]
        fetch = _ts([5, 6, 7])
        recv_counts = np.array([2, 2])
        send_counts = np.array([2, 2])
        plan = DynamicPlan(
            p2p=p2p,
            fetch=fetch,
            recv_counts=recv_counts,
            send_counts=send_counts,
        )
        assert len(plan.p2p) == 2
        assert plan.fetch.data.tolist() == [5, 6, 7]
        assert plan.recv_counts.tolist() == [2, 2]
        assert plan.writeback is None

    def test_with_writeback(self):
        p2p = [_ts([1])]
        fetch = _ts([2, 3])
        recv_counts = np.array([1])
        send_counts = np.array([1])
        writeback = _ts([4, 5])
        plan = DynamicPlan(
            p2p=p2p,
            fetch=fetch,
            recv_counts=recv_counts,
            send_counts=send_counts,
            writeback=writeback,
        )
        assert plan.writeback is not None
        assert plan.writeback.data.tolist() == [4, 5]

    def test_to_device(self):
        p2p = [_ts([1, 2]), _ts([3, 4])]
        fetch = _ts([5, 6])
        recv_counts = np.array([2, 2])
        send_counts = np.array([2, 2])
        plan = DynamicPlan(
            p2p=p2p,
            fetch=fetch,
            recv_counts=recv_counts,
            send_counts=send_counts,
        )
        plan_cpu = plan.to(torch.device("cpu"))
        assert plan_cpu.p2p[0].data.device.type == "cpu"
        assert plan_cpu.fetch_on_gpu is not None
        assert plan_cpu.fetch_on_gpu.device.type == "cpu"

    def test_to_device_with_writeback(self):
        p2p = [_ts([1])]
        fetch = _ts([2])
        recv_counts = np.array([1])
        send_counts = np.array([1])
        writeback = _ts([3, 4])
        plan = DynamicPlan(
            p2p=p2p,
            fetch=fetch,
            recv_counts=recv_counts,
            send_counts=send_counts,
            writeback=writeback,
        )
        plan_cpu = plan.to(torch.device("cpu"))
        assert plan_cpu.writeback is not None
        assert plan_cpu.writeback.data.device.type == "cpu"

    def test_repr(self):
        p2p = [_ts([1, 2])]
        fetch = _ts([3, 4])
        recv_counts = np.array([2])
        send_counts = np.array([2])
        plan = DynamicPlan(
            p2p=p2p,
            fetch=fetch,
            recv_counts=recv_counts,
            send_counts=send_counts,
        )
        r = repr(plan)
        assert "p2p" in r
        assert "fetch" in r
        assert "recv_counts" in r


class TestPlanSet:
    def test_empty_creation(self):
        ps = PlanSet()
        assert ps.mem_plan is None
        assert ps.box_plan is None
        assert ps.nfeat_plan is None

    def test_with_all_plans(self):
        mem_plan = DynamicPlan(
            p2p=[_ts([1])],
            fetch=_ts([2]),
            recv_counts=np.array([1]),
            send_counts=np.array([1]),
        )
        box_plan = DynamicPlan(
            p2p=[_ts([3])],
            fetch=_ts([4]),
            recv_counts=np.array([1]),
            send_counts=np.array([1]),
        )
        nfeat_plan = StaticPlan(to_cache=_ts([5]), fetch=_ts([5, 6]))
        ps = PlanSet(mem_plan=mem_plan, box_plan=box_plan, nfeat_plan=nfeat_plan)
        assert ps.mem_plan is not None
        assert ps.box_plan is not None
        assert ps.nfeat_plan is not None

    def test_to_device(self):
        mem_plan = DynamicPlan(
            p2p=[_ts([1])],
            fetch=_ts([2]),
            recv_counts=np.array([1]),
            send_counts=np.array([1]),
        )
        nfeat_plan = StaticPlan(to_cache=_ts([5]), fetch=_ts([5, 6]))
        ps = PlanSet(mem_plan=mem_plan, nfeat_plan=nfeat_plan)
        ps_cpu = ps.to(torch.device("cpu"))
        assert ps_cpu.mem_plan is not None
        assert ps_cpu.mem_plan.fetch.data.device.type == "cpu"
        assert ps_cpu.nfeat_plan is not None
        assert ps_cpu.nfeat_plan.to_cache.data.device.type == "cpu"

    def test_to_device_with_none_plans(self):
        ps = PlanSet(mem_plan=None, box_plan=None, nfeat_plan=None)
        ps_moved = ps.to(torch.device("cpu"))
        assert ps_moved.mem_plan is None
        assert ps_moved.box_plan is None
        assert ps_moved.nfeat_plan is None

    def test_to_device_preserves_shared_dynamic_plan(self):
        shared_plan = DynamicPlan(
            p2p=[_ts([1])],
            fetch=_ts([2]),
            recv_counts=np.array([1]),
            send_counts=np.array([1]),
        )
        ps = PlanSet(mem_plan=shared_plan, box_plan=shared_plan)

        ps_moved = ps.to(torch.device("cpu"))

        assert ps_moved.mem_plan is ps_moved.box_plan

    def test_repr(self):
        ps = PlanSet()
        r = repr(ps)
        assert "PlanSet" in r
        assert "mem=" in r
        assert "box=" in r
        assert "nfeat=" in r
