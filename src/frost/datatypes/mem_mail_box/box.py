from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


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

    def merge_box(
        self, other: BoxTsNid, is_overlap: bool, sort_idx: torch.Tensor | None = None
    ) -> BoxTsNid:
        # 1. 处理空情况
        if self.nid.numel() == 0:
            return other
        if other.nid.numel() == 0:
            return self

        # 3. Fallback: 必须硬连并排序
        cat_nid = torch.cat((self.nid, other.nid), dim=0)
        cat_data = torch.cat((self.data, other.data), dim=0)
        cat_ts = torch.cat((self.ts, other.ts), dim=0)

        # 计算排序索引
        if sort_idx is None:
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

        # 返回一个新的实例，保持 self 不变（Immutable）
        return self.__class__(
            nid=self.nid[sorted_indices],
            data=self.data[sorted_indices],
            ts=self.ts[sorted_indices],
            is_sorted=True,
            is_unique=self.is_unique,
        )

    def to(self, device: torch.device) -> BoxTsNid:
        return BoxTsNid(
            nid=self.nid.to(device),
            data=self.data.to(device),
            ts=self.ts.to(device),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def merge_1_mail(self, mail: "MailTsNid") -> BoxTsNid:
        from ._data_ops import merge_1_mail

        return merge_1_mail(self, mail)
