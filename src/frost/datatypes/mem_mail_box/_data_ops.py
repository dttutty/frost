from __future__ import annotations

from typing import TypeVar

import torch

from .box import BoxEntry
from .mail import MailEntry
from .mem import MemEntry

MemOrMail = TypeVar("MemOrMail", MemEntry, MailEntry)

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
