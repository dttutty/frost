from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class MemTsNid:
    nid: Tensor
    data: Tensor
    ts: Tensor
    is_sorted: bool = False
    is_unique: bool = False

    def __repr__(self) -> str:
        return (
            f"MemTsNid(nid={self.nid.shape}, data={self.data.shape}, "
            f"ts={self.ts.shape}, is_sorted={self.is_sorted}, is_unique={self.is_unique})"
        )

    def clone(self) -> MemTsNid:
        return MemTsNid(
            nid=self.nid.clone(),
            data=self.data.clone(),
            ts=self.ts.clone(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def cuda(self) -> MemTsNid:
        return MemTsNid(
            nid=self.nid.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def merge_mem(self, other: MemTsNid, is_overlap: bool) -> MemTsNid:
        return MemTsNid(
            nid=torch.cat((self.nid, other.nid), dim=0),
            data=torch.cat((self.data, other.data), dim=0),
            ts=torch.cat((self.ts, other.ts), dim=0),
            is_sorted=False,
            is_unique=self.is_unique and other.is_unique and not is_overlap,
        )

    def sort_by_nid(self) -> MemTsNid:
        if self.is_sorted:
            return self
        sorted_indices = torch.argsort(self.nid)
        self.nid = self.nid[sorted_indices]
        self.data = self.data[sorted_indices]
        self.ts = self.ts[sorted_indices]
        self.is_sorted = True
        return self

    def keep_last(self) -> MemTsNid:
        from ._data_ops import keep_last

        ret = keep_last(self)
        return ret

    def keep_mean(self) -> MemTsNid:
        from ._data_ops import keep_mean

        ret = keep_mean(self)
        return ret

    def to(self, device: torch.device) -> MemTsNid:
        return MemTsNid(
            nid=self.nid.to(device),
            data=self.data.to(device),
            ts=self.ts.to(device),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )
