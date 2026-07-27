import logging

import numpy
import torch

from frost.runtime.worker.types import AllType
from frost.datatypes.plan import DynamicPlan
from frost.datatypes.tasks import P2PTask

logger = logging.getLogger(__name__)


def make_p2p_task(
    data_td_nid: AllType,
    plan: DynamicPlan,
    device: torch.device,
    mini_batch_idx: int,
) -> P2PTask:
    """
    Build a P2PTask from a local data entry and a DynamicPlan.

    Preconditions on data_td_nid:
    - All nids must be unique.
    - For memory: keep only the latest (keep_last) or averaged (keep_mean) entry per node.
    - For mail: keep only updated nodes (pos_src, pos_dst), deduplicated via keep_last/keep_mean.
    - For mailbox: include all nodes' mailboxes.
    - For features: include all nodes' features.
    """
    p2p = plan.p2p
    recv_counts = plan.recv_counts
    send_counts = plan.send_counts

    nid = data_td_nid.nid
    data = data_td_nid.data
    ts = data_td_nid.ts

    x_dim = data.shape[1]

    buckets_ids, buckets_data, buckets_ts = [], [], []

    sizes = numpy.zeros(len(p2p), dtype=int)
    for dst_gpu_id, _ in enumerate(p2p):
        nids_to_send_tensor = p2p[dst_gpu_id].data
        send_mask_for_dst = torch.isin(nid, nids_to_send_tensor)
        matched_count = send_mask_for_dst.sum().item()
        expected_count = nids_to_send_tensor.numel()
        assert matched_count == expected_count, (
            f"make_p2p_task: GPU {dst_gpu_id} expected to send {expected_count} nodes, "
            f"but found {matched_count} in data (total {nid.numel()})."
        )
        buckets_ids.append(nid[send_mask_for_dst])
        buckets_data.append(data[send_mask_for_dst])
        buckets_ts.append(ts[send_mask_for_dst])
        sizes[dst_gpu_id] = send_mask_for_dst.sum().item()
    send_nid = torch.cat(buckets_ids, dim=0)
    send_data = torch.cat(buckets_data, dim=0)
    send_ts = torch.cat(buckets_ts, dim=0)

    total_recv = sum(recv_counts)
    recv_nid = torch.empty(total_recv, dtype=nid.dtype, device=device)
    recv_data = torch.empty(total_recv, x_dim, dtype=data.dtype, device=device)
    recv_ts = torch.empty(total_recv, dtype=ts.dtype, device=device)

    assert numpy.array_equal(sizes, send_counts)

    return P2PTask(
        send_nid=send_nid,
        recv_nid=recv_nid,
        data2send=send_data,
        data2recv=recv_data,
        ts2send=send_ts,
        ts2recv=recv_ts,
        recv_counts=recv_counts,
        send_counts=send_counts,
        mini_batch_idx=mini_batch_idx,
    )
