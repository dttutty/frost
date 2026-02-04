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

    def clone(self) -> BoxTsNid:
        return BoxTsNid(
            nid=self.nid.clone(),
            data=self.data.clone(),
            ts=self.ts.clone(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
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
        # 1. 处理空情况
        if self.nid.numel() == 0:
            return other
        if other.nid.numel() == 0:
            return self

        # 2. [优化] 尝试直接拼接
        # 假设 BoxTsNid 也有 is_sorted 属性 (如果没有，去掉这个检查或默认为 False)
        is_self_sorted = getattr(self, "is_sorted", False)
        is_other_sorted = getattr(other, "is_sorted", False)

        if is_self_sorted and is_other_sorted:
            # Case A: self 完全在 other 左边 -> 直接拼接
            if self.nid[-1] < other.nid[0]:
                return BoxTsNid(
                    nid=torch.cat((self.nid, other.nid), dim=0),
                    data=torch.cat((self.data, other.data), dim=0),
                    ts=torch.cat((self.ts, other.ts), dim=0),
                    is_sorted=True,
                    is_unique=self.is_unique and other.is_unique and not is_overlap,
                )
            # Case B: other 完全在 self 左边 -> 反向拼接
            elif other.nid[-1] < self.nid[0]:
                return BoxTsNid(
                    nid=torch.cat((other.nid, self.nid), dim=0),
                    data=torch.cat((other.data, self.data), dim=0),
                    ts=torch.cat((other.ts, self.ts), dim=0),
                    is_sorted=True,
                    is_unique=self.is_unique and other.is_unique and not is_overlap,
                )

        # 3. Fallback: 必须硬连并排序
        cat_nid = torch.cat((self.nid, other.nid), dim=0)
        cat_data = torch.cat((self.data, other.data), dim=0)
        cat_ts = torch.cat((self.ts, other.ts), dim=0)

        # 计算排序索引
        sort_idx = torch.argsort(cat_nid)

        return BoxTsNid(
            nid=cat_nid[sort_idx],
            data=cat_data[sort_idx],
            ts=cat_ts[sort_idx],
            is_sorted=True,
            is_unique=self.is_unique and other.is_unique and not is_overlap,
        )

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
