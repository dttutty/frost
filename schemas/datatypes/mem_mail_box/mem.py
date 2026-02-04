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
        # 1. 处理空情况 (边界检查需要元素存在)
        if self.nid.numel() == 0:
            return other.clone()  # 或者直接返回 other (如果允许共享引用)
        if other.nid.numel() == 0:
            return self.clone()

        # 2. 准备基础拼接
        cat_nid = torch.cat((self.nid, other.nid), dim=0)
        cat_data = torch.cat((self.data, other.data), dim=0)
        cat_ts = torch.cat((self.ts, other.ts), dim=0)

        is_result_sorted = False

        # 3. [优化] 如果双方都有序，尝试低成本判定
        if self.is_sorted and other.is_sorted:
            # Case A: self 完全在 other 左边 (self_max < other_min)
            # 注意：取 [-1] 和 [0] 是 O(1) 操作，非常快
            if self.nid[-1] < other.nid[0]:
                # 已经是按顺序拼接的
                is_result_sorted = True

            # Case B: other 完全在 self 左边 (other_max < self_min)
            elif other.nid[-1] < self.nid[0]:
                # 这种情况下，我们需要交换拼接顺序才能保持有序
                # 但上面的 cat 已经执行了 (为了代码简洁)，所以这里我们选择
                # 既然已经 cat 了乱序的，不如直接利用 argsort 重新排
                # 或者，你可以为了极致性能重写上面的 cat 逻辑：
                # if other < self: cat(other, self) ...

                # 为了保持代码简洁，这里如果发现是反序或交错，统一走下面的排序逻辑
                pass

        # 4. 如果没能通过边界检查达成有序，且你需要它有序，则执行排序
        # (通常 merge_mem 后最好保持有序，方便后续 searchsorted)
        if not is_result_sorted:
            # 使用 stable=True 保持相对顺序（如果需要的话）
            sort_idx = torch.argsort(cat_nid)
            cat_nid = cat_nid[sort_idx]
            cat_data = cat_data[sort_idx]
            cat_ts = cat_ts[sort_idx]
            is_result_sorted = True

        return MemTsNid(
            nid=cat_nid,
            data=cat_data,
            ts=cat_ts,
            is_sorted=is_result_sorted,
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
