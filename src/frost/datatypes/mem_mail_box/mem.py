from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class MemEntry:
    nid: Tensor
    data: Tensor
    ts: Tensor
    is_sorted: bool = False
    is_unique: bool = False

    def __repr__(self) -> str:
        return (
            f"MemEntry(nid={self.nid.shape}, data={self.data.shape}, "
            f"ts={self.ts.shape}, is_sorted={self.is_sorted}, is_unique={self.is_unique})"
        )

    def clone(self) -> MemEntry:
        return MemEntry(
            nid=self.nid.clone(),
            data=self.data.clone(),
            ts=self.ts.clone(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def cuda(self) -> MemEntry:
        return MemEntry(
            nid=self.nid.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def merge_mem(self, other: MemEntry, is_overlap: bool) -> tuple[MemEntry, Tensor]:
        # 1. Handle empty cases (boundary checks require at least one element)
        if self.nid.numel() == 0:
            return other.clone(), torch.arange(
                other.nid.numel(), device=other.nid.device
            )
        if other.nid.numel() == 0:
            return self.clone(), torch.arange(self.nid.numel(), device=self.nid.device)

        # 2. Concatenate both entries
        cat_nid = torch.cat((self.nid, other.nid), dim=0)
        cat_data = torch.cat((self.data, other.data), dim=0)
        cat_ts = torch.cat((self.ts, other.ts), dim=0)

        # 3. Sort by nid to maintain the sorted invariant required by searchsorted downstream
        sort_idx = torch.argsort(cat_nid)
        cat_nid = cat_nid[sort_idx]
        cat_data = cat_data[sort_idx]
        cat_ts = cat_ts[sort_idx]

        return MemEntry(
            nid=cat_nid,
            data=cat_data,
            ts=cat_ts,
            is_sorted=True,
            is_unique=self.is_unique and other.is_unique and not is_overlap,
        ), sort_idx

    def sort_by_nid(self) -> MemEntry:
        if self.is_sorted:
            return self
        sorted_indices = torch.argsort(self.nid)

        # Return a new instance; self is not modified (immutable pattern)
        return self.__class__(
            nid=self.nid[sorted_indices],
            data=self.data[sorted_indices],
            ts=self.ts[sorted_indices],
            is_sorted=True,
            is_unique=self.is_unique,
        )

    def to(self, device: torch.device) -> MemEntry:
        return MemEntry(
            nid=self.nid.to(device),
            data=self.data.to(device),
            ts=self.ts.to(device),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )
