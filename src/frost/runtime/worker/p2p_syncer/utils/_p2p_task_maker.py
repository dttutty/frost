import numpy
import torch

from frost.datatypes.plan import DynamicPlan
from frost.datatypes.tasks import P2PTask
from frost.runtime.worker.types import AllType


def make_p2p_task(
    entry: AllType,
    plan: DynamicPlan,
    device: torch.device,
    mini_batch_idx: int,
) -> P2PTask:
    """
    Build a P2PTask by gathering per-GPU send buffers from the local entry.

    Uses searchsorted for O(log n) lookup because entry.nid is guaranteed sorted.
    """
    p2p = plan.p2p
    recv_counts = plan.recv_counts
    send_counts = plan.send_counts

    nid: torch.Tensor = entry.nid
    data: torch.Tensor = entry.data
    ts: torch.Tensor = entry.ts

    # entry.nid must be sorted for searchsorted to work correctly
    if not getattr(entry, "is_sorted", False):
        # Sort if not already sorted; upstream code should guarantee this
        entry = entry.sort_by_nid()
        nid = entry.nid
        data = entry.data
        ts = entry.ts

    src_len = nid.numel()
    planned_sizes = numpy.asarray(
        [nodes.data.numel() for nodes in p2p], dtype=send_counts.dtype
    )
    if not numpy.array_equal(planned_sizes, send_counts):
        raise RuntimeError(
            "P2P plan send counts do not match its per-peer node lists: "
            f"counts={send_counts.tolist()} nodes={planned_sizes.tolist()}"
        )

    nonempty = [nodes.data for nodes in p2p if nodes.data.numel() > 0]
    if nonempty:
        planned_nids = nonempty[0] if len(nonempty) == 1 else torch.cat(nonempty)
        if src_len == 0:
            raise RuntimeError(
                f"P2P source is empty for {planned_nids.numel()} planned nodes"
            )

        # All peer buckets use the same sorted source. One combined lookup avoids
        # launching search/mask/compaction kernels separately for every peer.
        gather_indices = torch.searchsorted(nid, planned_nids)
        in_bounds = gather_indices < src_len
        gather_indices = gather_indices.clamp_max(src_len - 1)
        matched = in_bounds & (nid[gather_indices] == planned_nids)
        if nid.device.type == "cuda":
            # Keep the planner invariant check asynchronous on the hot path.
            # A failed assertion is fatal, just like the former split mismatch.
            torch._assert_async(matched.all())
        elif not bool(matched.all()):
            missing = planned_nids[~matched]
            raise RuntimeError(
                "P2P source is missing planned node ids: "
                f"batch={mini_batch_idx} missing={missing.tolist()}"
            )

        send_nid = planned_nids
        send_data = data[gather_indices]
        send_ts = ts[gather_indices]
    else:
        # No data to send to any GPU
        send_nid = nid.new_empty((0,))
        send_data = data.new_empty((0, *data.shape[1:]))
        send_ts = ts.new_empty((0, *ts.shape[1:]))

    total_recv = sum(recv_counts)
    recv_nid = torch.empty(total_recv, dtype=nid.dtype, device=device)

    data_shape = list(data.shape)
    data_shape[0] = total_recv
    recv_data = torch.empty(data_shape, dtype=data.dtype, device=device)

    ts_shape = list(ts.shape)
    ts_shape[0] = total_recv
    recv_ts = torch.empty(ts_shape, dtype=ts.dtype, device=device)

    p2p_task = P2PTask(
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

    return p2p_task
