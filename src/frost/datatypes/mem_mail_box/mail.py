from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class MailTsNid:
    nid: Tensor
    data: Tensor
    ts: Tensor
    is_sorted: bool = False
    is_unique: bool = False

    def __repr__(self) -> str:
        return (
            f"MailTsNid(nid={self.nid.shape}, data={self.data.shape}, "
            f"ts={self.ts.shape}, is_sorted={self.is_sorted}, is_unique={self.is_unique})"
        )

    def clone(self) -> MailTsNid:
        return MailTsNid(
            nid=self.nid.clone(),
            data=self.data.clone(),
            ts=self.ts.clone(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def _ensure_sorted_unique_nid(self) -> MailTsNid:
        from ._data_ops import _ensure_sorted_unique_nid

        return _ensure_sorted_unique_nid(self)

    def cuda(self) -> MailTsNid:
        return MailTsNid(
            nid=self.nid.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def merge_with_priority(self, other: MailTsNid) -> MailTsNid:
        from ._data_ops import merge_with_priority

        return merge_with_priority(self, other)

    def sort_by_nid(self) -> MailTsNid:
        if self.is_sorted:
            return self
        sorted_indices = torch.argsort(self.nid)
        self.nid = self.nid[sorted_indices]
        self.data = self.data[sorted_indices]
        self.ts = self.ts[sorted_indices]
        self.is_sorted = True
        return self
