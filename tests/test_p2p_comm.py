import numpy as np
import torch

from frost.datatypes.tasks import P2PTask
from frost.runtime.worker.p2p_syncer._p2p_comm import _copy_single_rank_p2p_round


def _make_task(*, width: int, box: bool = False) -> P2PTask:
    send_nid = torch.tensor([3, 5, 8], dtype=torch.long)
    recv_nid = torch.empty_like(send_nid)
    send_counts = np.array([3], dtype=np.int32)
    recv_counts = np.array([3], dtype=np.int32)

    if box:
        send_data = torch.randn(3, 1, width)
        recv_data = torch.empty_like(send_data)
        send_ts = torch.tensor([[10], [11], [12]], dtype=torch.long)
        recv_ts = torch.empty_like(send_ts)
    else:
        send_data = torch.randn(3, width)
        recv_data = torch.empty_like(send_data)
        send_ts = torch.tensor([10, 11, 12], dtype=torch.long)
        recv_ts = torch.empty_like(send_ts)

    return P2PTask(
        send_nid=send_nid,
        recv_nid=recv_nid,
        data2send=send_data,
        data2recv=recv_data,
        ts2send=send_ts,
        ts2recv=recv_ts,
        recv_counts=recv_counts,
        send_counts=send_counts,
        mini_batch_idx=7,
    )


def test_copy_single_rank_p2p_round_copies_mem_and_box() -> None:
    mem_task = _make_task(width=4, box=False)
    box_task = _make_task(width=6, box=True)

    _copy_single_rank_p2p_round(mem_task, box_task)

    assert torch.equal(mem_task.recv_nid, mem_task.send_nid)
    assert torch.equal(mem_task.data2recv, mem_task.data2send)
    assert torch.equal(mem_task.ts2recv, mem_task.ts2send)

    assert torch.equal(box_task.recv_nid, box_task.send_nid)
    assert torch.equal(box_task.data2recv, box_task.data2send)
    assert torch.equal(box_task.ts2recv, box_task.ts2send)
