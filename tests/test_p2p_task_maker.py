import numpy as np
import pytest
import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.plan import DynamicPlan
from frost.runtime.worker.p2p_syncer.utils._p2p_task_maker import make_p2p_task
from frost.utils.tensor_set import TensorSet


def test_make_p2p_task_keeps_box_shapes_when_no_send_is_needed():
    empty = TensorSet.empty_like(torch.tensor([], dtype=torch.long))
    entry = BoxEntry(
        nid=torch.tensor([3, 5], dtype=torch.long),
        data=torch.randn(2, 1, 4),
        ts=torch.randn(2, 1),
        is_sorted=True,
        is_unique=True,
    )
    plan = DynamicPlan(
        p2p=[empty],
        fetch=empty,
        recv_counts=np.array([0], dtype=np.int32),
        send_counts=np.array([0], dtype=np.int32),
        writeback=None,
    )

    task = make_p2p_task(
        entry=entry,
        plan=plan,
        device=torch.device("cpu"),
        mini_batch_idx=0,
    )

    assert task.send_nid.shape == (0,)
    assert task.data2send.shape == (0, 1, 4)
    assert task.ts2send.shape == (0, 1)
    assert task.data2recv.shape == (0, 1, 4)
    assert task.ts2recv.shape == (0, 1)


def test_make_p2p_task_gathers_all_peer_buckets_with_one_layout():
    entry = MemEntry(
        nid=torch.tensor([1, 3, 5, 7], dtype=torch.long),
        data=torch.tensor([[10.0], [30.0], [50.0], [70.0]]),
        ts=torch.tensor([1.0, 3.0, 5.0, 7.0]),
        is_sorted=True,
        is_unique=True,
    )
    plan = DynamicPlan(
        p2p=[TensorSet(torch.tensor([1, 5])), TensorSet(torch.tensor([3, 7]))],
        fetch=TensorSet(torch.tensor([], dtype=torch.long)),
        recv_counts=np.array([1, 2], dtype=np.int32),
        send_counts=np.array([2, 2], dtype=np.int32),
    )

    task = make_p2p_task(entry, plan, torch.device("cpu"), mini_batch_idx=4)

    assert task.send_nid.tolist() == [1, 5, 3, 7]
    assert task.data2send.squeeze(-1).tolist() == [10.0, 50.0, 30.0, 70.0]
    assert task.ts2send.tolist() == [1.0, 5.0, 3.0, 7.0]
    assert task.data2recv.shape == (3, 1)


def test_make_p2p_task_rejects_missing_cpu_source_node():
    entry = MemEntry(
        nid=torch.tensor([1, 3], dtype=torch.long),
        data=torch.ones((2, 1)),
        ts=torch.ones(2),
        is_sorted=True,
        is_unique=True,
    )
    plan = DynamicPlan(
        p2p=[TensorSet(torch.tensor([1, 2]))],
        fetch=TensorSet(torch.tensor([], dtype=torch.long)),
        recv_counts=np.array([0], dtype=np.int32),
        send_counts=np.array([2], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="missing planned node ids"):
        make_p2p_task(entry, plan, torch.device("cpu"), mini_batch_idx=9)
