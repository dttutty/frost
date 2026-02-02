from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from .mail import MailTsNid


@dataclass(kw_only=True)
class BoxTsNid:
    nid: Tensor
    data: Tensor
    ts: torch.Tensor
    is_sorted: bool = False
    is_unique: bool = False

    def __repr__(self) -> str:
        return (
            f"BoxTsNid(nid={self.nid.shape}, data={self.data.shape}, "
            f"ts={self.ts.shape}, is_sorted={self.is_sorted}, is_unique={self.is_unique})"
        )

    def cuda(self) -> BoxTsNid:
        return BoxTsNid(
            nid=self.nid.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            is_unique=self.is_unique,
            is_sorted=self.is_sorted,
        )

    def to_mail(self) -> MailTsNid:
        from .mail import MailTsNid

        return MailTsNid(
            nid=self.nid,
            data=self.data.squeeze(1),
            ts=self.ts.squeeze(1),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def merge_box(self, other: BoxTsNid, is_overlap: bool) -> BoxTsNid:
        other_data = other.data
        other_ts = other.ts

        self = BoxTsNid(
            nid=torch.cat((self.nid, other.nid), dim=0),
            data=torch.cat((self.data, other_data), dim=0),
            ts=torch.cat((self.ts, other_ts), dim=0),
            is_sorted=False,
            is_unique=self.is_unique and other.is_unique and not is_overlap,
        )
        return self

    def sort_by_nid(self) -> BoxTsNid:
        if self.is_sorted:
            return self
        sorted_indices = torch.argsort(self.nid)
        self.nid = self.nid[sorted_indices]
        self.data = self.data[sorted_indices]
        self.ts = self.ts[sorted_indices]
        self.is_sorted = True
        return self

    def to(self, device: torch.device) -> BoxTsNid:
        return BoxTsNid(
            nid=self.nid.to(device),
            data=self.data.to(device),
            ts=self.ts.to(device),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )
