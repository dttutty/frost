from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tensor_set import TensorSet


def is_sorted(t: torch.Tensor) -> bool:
    return torch.all(torch.diff(t) >= 0)


@dataclass(slots=True, kw_only=True)
class MemTsNid:
    nid: Tensor
    data: Tensor
    ts: Tensor
    # ct: Tensor | None = None  # 留着扩展用
    is_sorted: bool = False

    def clone(self) -> MemTsNid:
        return MemTsNid(
            nid=self.nid.clone(), data=self.data.clone(), ts=self.ts.clone()
        )

    def cuda(self) -> MemTsNid:
        return MemTsNid(nid=self.nid.cuda(), data=self.data.cuda(), ts=self.ts.cuda())

    def merge_mem(self, other: MemTsNid) -> MemTsNid:
        assert TensorSet.intersection(self.nid, other.nid).numel() == 0, (
            "NID overlap between MemTsNid instances."
        )
        return MemTsNid(
            nid=torch.cat((self.nid, other.nid), dim=0),
            data=torch.cat((self.data, other.data), dim=0),
            ts=torch.cat((self.ts, other.ts), dim=0),
        )

    def sort_by_nid(self) -> MemTsNid:
        sorted_indices = torch.argsort(self.nid)
        self.nid = self.nid[sorted_indices]
        self.data = self.data[sorted_indices]
        self.ts = self.ts[sorted_indices]
        self.is_sorted = True
        return self

    def keep_last(self) -> MemTsNid:
        from ._data_ops import keep_last

        keep_last(self)
        return self
