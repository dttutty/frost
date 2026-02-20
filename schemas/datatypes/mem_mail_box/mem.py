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

    def merge_mem(self, other: MemTsNid, is_overlap: bool) -> tuple[MemTsNid, Tensor]:
        # 1. 处理空情况 (边界检查需要元素存在)
        if self.nid.numel() == 0:
            return other.clone(), torch.arange(
                other.nid.numel(), device=other.nid.device
            )
        if other.nid.numel() == 0:
            return self.clone(), torch.arange(self.nid.numel(), device=self.nid.device)

        # 2. 准备基础拼接
        cat_nid = torch.cat((self.nid, other.nid), dim=0)
        cat_data = torch.cat((self.data, other.data), dim=0)
        cat_ts = torch.cat((self.ts, other.ts), dim=0)

        # 4. 如果没能通过边界检查达成有序，且你需要它有序，则执行排序
        # (通常 merge_mem 后最好保持有序，方便后续 searchsorted)
        # 使用 stable=True 保持相对顺序（如果需要的话）
        sort_idx = torch.argsort(cat_nid)
        cat_nid = cat_nid[sort_idx]
        cat_data = cat_data[sort_idx]
        cat_ts = cat_ts[sort_idx]

        return MemTsNid(
            nid=cat_nid,
            data=cat_data,
            ts=cat_ts,
            is_sorted=True,
            is_unique=self.is_unique and other.is_unique and not is_overlap,
        ), sort_idx

    def sort_by_nid(self) -> MemTsNid:
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
