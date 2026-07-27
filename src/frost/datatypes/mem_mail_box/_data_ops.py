from __future__ import annotations

import logging
from typing import TypeVar

import torch

from .box import BoxEntry
from .mail import MailEntry
from .mem import MemEntry

MemOrMail = TypeVar("MemOrMail", MemEntry, MailEntry)

logger = logging.getLogger(__name__)


def box_to_mail(box_ts_nid) -> MailEntry:
    from .mail import MailEntry

    return MailEntry(
        nid=box_ts_nid.nid,
        data=box_ts_nid.data.squeeze(1),
        ts=box_ts_nid.ts.squeeze(1),
        is_sorted=box_ts_nid.is_sorted,
        is_unique=box_ts_nid.is_unique,
    )


def mail_to_box(mem_ts_nid) -> BoxEntry:
    return BoxEntry(
        nid=mem_ts_nid.nid,
        data=mem_ts_nid.data.unsqueeze(1),
        ts=mem_ts_nid.ts.unsqueeze(1),
        is_sorted=mem_ts_nid.is_sorted,
        is_unique=mem_ts_nid.is_unique,
    )


def compute_last_indices(nid: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    if nid.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=nid.device)

    unique_nids, compact_idx = torch.unique(nid, return_inverse=True)
    M = unique_nids.numel()

    # Step 1: deterministic max-ts per group via scatter_reduce_
    if ts.is_floating_point():
        fill_val = float("-inf")
    else:
        fill_val = torch.iinfo(ts.dtype).min
    max_ts = ts.new_full((M,), fill_val)
    max_ts.scatter_reduce_(0, compact_idx, ts, reduce="amax", include_self=False)

    # Step 2: mask entries that match the per-group maximum
    is_max = ts == max_ts[compact_idx]

    # Step 3: among ties pick the largest original index (deterministic)
    arange = torch.arange(nid.numel(), device=nid.device, dtype=torch.long)
    argmax = torch.full((M,), -1, dtype=torch.long, device=nid.device)
    argmax.scatter_reduce_(
        0, compact_idx[is_max], arange[is_max], reduce="amax", include_self=False
    )
    return argmax


def keep_last(data: MemOrMail) -> MemOrMail:
    assert data.ts is not None, "TS must be provided for keep_last"
    indices = compute_last_indices(data.nid, data.ts)

    # ``indices`` is ordered by the sorted unique node IDs returned by
    # ``torch.unique``.  Apply it even when every input node is unique: in that
    # case it is still the permutation that turns an arbitrary root layout
    # (for example ``[src | dst]``) into sorted order.  Skipping the indexing
    # while setting ``is_sorted=True`` makes downstream ``searchsorted`` calls
    # silently miss nodes.
    data.nid = data.nid[indices]
    data.data = data.data[indices]
    data.ts = data.ts[indices]

    # compute_last_indices relies on unique's group order (sorted by nid by default)
    data.is_sorted = True
    data.is_unique = True
    return data


def merge_1_mail(box: BoxEntry, mail: MailEntry) -> BoxEntry:
    """Insert one new mail per node into the circular mailbox buffer (shift-left FIFO).

    Mirrors TGL_ORIGINAL's ``MailBox.update_mailbox`` circular-buffer write for
    ``deliver_to='neighbors'`` with ``mail_combine='last'`` or ``'mean'``:

        mailbox[nid, next_mail_pos[nid]] = mail
        next_mail_pos[nid] = (next_mail_pos[nid] + 1) % mailbox_size

    Here we implement equivalent behaviour via a left-shift on the slot axis:
    slot-0 (oldest) is dropped, all remaining slots shift left by one, and
    the new mail is written into the last slot (index ``boxsize - 1``).

    Args:
        box : BoxEntry – current mailbox loaded from the flat graph.
                         ``nid``  shape [N_all] – may contain -1 padding and
                         duplicate node IDs (same node can appear multiple times
                         in the flat graph; all copies carry identical data from
                         shared memory).
                         ``data`` shape [N_all, boxsize, D]
                         ``ts``   shape [N_all, boxsize]
        mail: MailEntry – new mail per destination node, already de-duplicated
                         by ``keep_last()`` / ``keep_mean()`` so ``nid`` is
                         unique and sorted.
                         ``nid``  shape [M]
                         ``data`` shape [M, D]
                         ``ts``   shape [M]

    Returns:
        BoxEntry  nid=[M], data=[M, boxsize, D], ts=[M, boxsize].
        The caller (BoxWriter) writes this back to the shared-memory mailbox.
    """
    M = mail.nid.numel()
    boxsize = box.data.shape[1]
    D = box.data.shape[2]
    device = box.data.device

    # Zero-initialised output: nodes with no prior mailbox history start empty.
    out_data = torch.zeros(M, boxsize, D, dtype=box.data.dtype, device=device)
    out_ts = torch.zeros(M, boxsize, dtype=box.ts.dtype, device=device)

    if M == 0:
        return BoxEntry(
            nid=mail.nid,
            data=out_data,
            ts=out_ts,
            is_sorted=mail.is_sorted,
            is_unique=mail.is_unique,
        )

    # Build a sorted, unique lookup table from the flat-graph old-box.
    # box.nid may contain -1 (padding) and duplicate node IDs – filter first.
    valid_mask_box = box.nid != -1
    if valid_mask_box.any():
        valid_box_nid = box.nid[valid_mask_box]  # [V]
        valid_box_indices = torch.nonzero(valid_mask_box, as_tuple=False).squeeze(
            -1
        )  # [V]

        # Sort by nid to enable searchsorted; for duplicates keep the last
        # occurrence (all duplicates hold identical data from shared memory).
        sort_idx = torch.argsort(valid_box_nid)
        sorted_nid = valid_box_nid[sort_idx]  # [V], sorted ascending
        sorted_global_idx = valid_box_indices[sort_idx]  # [V]

        # Deduplicate: keep the LAST occurrence of each nid in sorted order
        # (last = highest index -> most recent write, though values are equal).
        keep = torch.ones(sorted_nid.numel(), dtype=torch.bool, device=device)
        if sorted_nid.numel() > 1:
            keep[:-1] = sorted_nid[:-1] != sorted_nid[1:]

        unique_nid_box = sorted_nid[keep]  # [U], strictly increasing
        unique_global_idx = sorted_global_idx[keep]  # [U]

        # Match each mail destination against the old-box unique nids.
        pos = torch.searchsorted(unique_nid_box, mail.nid)
        in_bound = pos < unique_nid_box.numel()
        pos_safe = pos.clamp(max=unique_nid_box.numel() - 1)
        found_mask = in_bound & (unique_nid_box[pos_safe] == mail.nid)

        if found_mask.any():
            found_mail_idx = torch.nonzero(found_mask, as_tuple=False).squeeze(-1)
            found_box_global_idx = unique_global_idx[pos_safe[found_mask]]

            # Circular-buffer shift: drop slot-0 (oldest), shift everything
            # left by 1, leaving the last slot empty for the incoming mail.
            if boxsize > 1:
                out_data[found_mail_idx, : boxsize - 1, :] = box.data[
                    found_box_global_idx, 1:, :
                ]
                out_ts[found_mail_idx, : boxsize - 1] = box.ts[found_box_global_idx, 1:]

    # Write the new mail into the last (most-recent) slot for every node.
    out_data[:, boxsize - 1, :] = mail.data
    out_ts[:, boxsize - 1] = mail.ts

    return BoxEntry(
        nid=mail.nid,
        data=out_data,
        ts=out_ts,
        is_sorted=mail.is_sorted,
        is_unique=mail.is_unique,
    )


def keep_mean(data: MemOrMail) -> MemOrMail:
    """
    Deduplicate by nid using a uniform-weight mean (no count tensor).
    - Duplicate nids: data and ts are averaged with equal weights.
    - Modifies data in place.
    """
    assert data.ts is not None, "TS must be provided for keep_mean"
    if data.nid.numel() == 0:
        data.is_sorted = True
        data.is_unique = True
        return data

    unique_nids, compact_idx = torch.unique(data.nid, return_inverse=True)
    M = unique_nids.numel()

    # Count per group
    ct = torch.zeros(M, dtype=data.ts.dtype, device=data.ts.device)
    ct.scatter_reduce_(
        0, compact_idx, torch.ones_like(data.ts), reduce="sum", include_self=False
    )

    # Sum data and timestamps
    idx_data = compact_idx.unsqueeze(-1).expand_as(data.data)
    sum_data = torch.zeros(
        M, data.data.shape[1], dtype=data.data.dtype, device=data.data.device
    )
    sum_data.scatter_reduce_(0, idx_data, data.data, reduce="sum", include_self=False)
    sum_ts = torch.zeros(M, dtype=data.ts.dtype, device=data.ts.device)
    sum_ts.scatter_reduce_(0, compact_idx, data.ts, reduce="sum", include_self=False)

    data.nid = unique_nids
    data.data = sum_data / ct.unsqueeze(-1)
    mean_ts = sum_ts / ct
    if not torch.is_floating_point(data.ts):
        mean_ts = mean_ts.round().to(data.ts.dtype)
    data.ts = mean_ts

    data.is_sorted = True
    data.is_unique = True
    return data


def _ensure_sorted_unique_nid(x: MemOrMail) -> MemOrMail:
    """
    Ensure x.nid is sorted in ascending order and contains no duplicates.
    Modifies x in place.
    """
    if not getattr(x, "is_sorted", False):
        x = x.sort_by_nid()

    if not getattr(x, "is_unique", False):
        if x.nid.numel() <= 1:
            x.is_unique = True
            return x

        keep = torch.empty(x.nid.shape[0], dtype=torch.bool, device=x.nid.device)
        keep[0] = True
        keep[1:] = x.nid[1:] != x.nid[:-1]

        if keep.sum().item() != x.nid.numel():
            x.nid = x.nid[keep]
            x.data = x.data[keep]
            if x.ts is not None:
                x.ts = x.ts[keep]

        x.is_unique = True

    return x


def _mask_not_in_sorted_unique(
    high_nid_sorted_unique: torch.Tensor, low_nid: torch.Tensor
) -> torch.Tensor:
    pos = torch.searchsorted(high_nid_sorted_unique, low_nid)
    in_range = pos < high_nid_sorted_unique.numel()
    hit = torch.zeros_like(in_range, dtype=torch.bool)
    hit[in_range] = high_nid_sorted_unique[pos[in_range]] == low_nid[in_range]
    return ~hit


def merge_with_priority(
    high_priority_data: MemOrMail, low_priority_data: MemOrMail
) -> MemOrMail:
    """
    Merge two entries, preferring high_priority_data on conflict.
    Any nid in low_priority_data that also appears in high_priority_data is discarded.
    Both inputs must have ts. No count tensor required.
    """
    if low_priority_data.nid.numel() == 0:
        return type(high_priority_data)(
            nid=high_priority_data.nid,
            data=high_priority_data.data,
            ts=high_priority_data.ts,
            is_sorted=high_priority_data.is_sorted,
            is_unique=high_priority_data.is_unique,
        )
    if high_priority_data.nid.numel() == 0:
        return type(low_priority_data)(
            nid=low_priority_data.nid,
            data=low_priority_data.data,
            ts=low_priority_data.ts,
            is_sorted=low_priority_data.is_sorted,
            is_unique=low_priority_data.is_unique,
        )

    # Precondition: high must be sorted + unique
    high = _ensure_sorted_unique_nid(high_priority_data)

    # low does not need to be sorted; membership test handles it directly
    mask_keep_low = _mask_not_in_sorted_unique(high.nid, low_priority_data.nid)

    ret = type(high)(
        nid=torch.cat([high.nid, low_priority_data.nid[mask_keep_low]], dim=0),
        data=torch.cat([high.data, low_priority_data.data[mask_keep_low]], dim=0),
        ts=torch.cat([high.ts, low_priority_data.ts[mask_keep_low]], dim=0),
        is_sorted=False,
        is_unique=False,
    )
    return ret.sort_by_nid()
