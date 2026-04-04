import numpy as np
import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
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
