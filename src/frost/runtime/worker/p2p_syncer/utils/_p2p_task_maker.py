import logging
import sys

import numpy
import torch
import torch.distributed as dist

from frost.aliases.syncer import AllType
from frost.datatypes.plan import DynamicPlan
from frost.datatypes.tasks import P2PTask

logger = logging.getLogger(__name__)


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

    nid = entry.nid
    data = entry.data
    ts = entry.ts

    # entry.nid must be sorted for searchsorted to work correctly
    if not getattr(entry, "is_sorted", False):
        # Sort if not already sorted; upstream code should guarantee this
        entry = entry.sort_by_nid()
        nid = entry.nid
        data = entry.data
        ts = entry.ts

    buckets_ids, buckets_data, buckets_ts = [], [], []
    sizes = numpy.zeros(len(p2p), dtype=int)
    missing_by_dst: list[torch.Tensor] = []

    src_len = nid.numel()

    for dst_gpu_id, _ in enumerate(p2p):
        nids_to_send_tensor = p2p[dst_gpu_id].data
        expected_count = nids_to_send_tensor.numel()

        if expected_count == 0:
            sizes[dst_gpu_id] = 0
            missing_by_dst.append(nids_to_send_tensor)
            continue

        # 1. Find insertion positions via binary search (nid must be 1D sorted)
        indices = torch.searchsorted(nid, nids_to_send_tensor)

        # 2. Bounds check and value match:
        #    searchsorted returns insertion positions that may equal src_len (out of bounds)
        #    or point to a different value — clamp first, then verify equality.
        indices_clamped = indices.clamp(max=src_len - 1)
        mask_found = nid[indices_clamped] == nids_to_send_tensor

        # 3. Keep only the indices that were actually found
        valid_gather_indices = indices_clamped[mask_found]

        matched_count = valid_gather_indices.numel()
        missing_by_dst.append(nids_to_send_tensor[~mask_found])

        if matched_count != expected_count:
            logger.warning(
                f"make_p2p_task: GPU {dst_gpu_id} expected to send {expected_count} nodes, "
                f"but found {matched_count} in data (total {nid.numel()})."
            )

        # Gather data at the matched source positions
        buckets_ids.append(nid[valid_gather_indices])
        buckets_data.append(data[valid_gather_indices])
        buckets_ts.append(ts[valid_gather_indices])

        sizes[dst_gpu_id] = matched_count

    if len(buckets_ids) > 0:
        send_nid = torch.cat(buckets_ids, dim=0)
        send_data = torch.cat(buckets_data, dim=0)
        send_ts = torch.cat(buckets_ts, dim=0)
    else:
        # No data to send to any GPU
        send_nid = nid.new_empty((0,))
        send_data = data.new_empty((0, *data.shape[1:]))
        send_ts = ts.new_empty((0, *ts.shape[1:]))

    total_planned_send = sum(send_counts)
    real_shape = send_nid.shape[0]

    if total_planned_send != real_shape:
        try:
            rank = dist.get_rank()
        except Exception:
            rank = "Unknown"

        name = "make_p2p_task_construction"

        logger.debug("\n[CRITICAL ERROR] P2P Mismatch on Rank %s | Op: %s", rank, name)
        logger.debug("  > Sum of send_counts (Planned): %s", total_planned_send)
        logger.debug("  > Actual Tensor dim0 (Gathered): %s", real_shape)
        logger.debug("  > Tensor Full Shape: %s", send_nid.shape)
        logger.debug("  > Send Cts List (Planned): %s", send_counts)
        logger.debug("  > Actual Gathered Sizes: %s", sizes)
        logger.error(
            "P2P plan/data mismatch: batch=%s rank=%s entry_nids=%s "
            "planned_by_dst=%s missing_by_dst=%s",
            mini_batch_idx,
            rank,
            nid[:32].detach().cpu().tolist(),
            [ts.data[:32].detach().cpu().tolist() for ts in p2p],
            [missing[:32].detach().cpu().tolist() for missing in missing_by_dst],
        )
        sys.stdout.flush()

        raise RuntimeError(
            f"AAA Split size mismatch in {name}: planned {total_planned_send} vs actual {real_shape}"
        )

    total_recv = sum(recv_counts)
    recv_nid = torch.empty(total_recv, dtype=nid.dtype, device=device)

    data_shape = list(data.shape)
    data_shape[0] = total_recv
    recv_data = torch.empty(data_shape, dtype=data.dtype, device=device)

    ts_shape = list(ts.shape)
    ts_shape[0] = total_recv
    recv_ts = torch.empty(ts_shape, dtype=ts.dtype, device=device)

    assert numpy.array_equal(sizes, send_counts)

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
