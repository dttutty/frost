from __future__ import annotations

from dataclasses import dataclass

import torch
from numpy.typing import NDArray

from tensor_set import TensorSet


def set_to_tensor(s: set[int]) -> torch.Tensor:
    return torch.tensor(list(s)).long()


@dataclass(slots=True, kw_only=True)
class BatchNodeStateIO:
    mem_needed: TensorSet
    mem_updated: TensorSet
    mailbox_needed: TensorSet | None
    mailbox_updated: TensorSet | None
    nfeat_needed: TensorSet


@dataclass(slots=True, kw_only=True)
class PlanSet:
    """
    统一的 PlanSet，直接持有 DyPlan 和 StPlan。
    """

    mem: DyPlan | None = None
    box: DyPlan | None = None
    nfeat: StPlan | None = None

    def to(self, device: torch.device) -> PlanSet:
        """
        递归地将所有子 Plan 移动到指定设备
        """
        return PlanSet(
            mem=self.mem.to(device) if self.mem is not None else None,
            box=self.box.to(device) if self.box is not None else None,
            nfeat=self.nfeat.to(device) if self.nfeat is not None else None,
        )

    def cuda(self) -> PlanSet:
        return self.to(torch.device("cuda"))


@dataclass(slots=True, kw_only=True)
class DyPlan:
    p2p: list[TensorSet]
    fetch: TensorSet
    rec_cts: NDArray
    send_cts: NDArray
    writeback: TensorSet | None = None

    def to(self, device: torch.device) -> DyPlan:
        """
        递归地将内部的 TensorSet 移动到指定设备。
        注意：这会创建新的 TensorSet 对象（因为 TensorSet 是不可变的）
        """

        # 辅助函数：快速根据 data 创建新 Set，跳过 unique/sort 检查以提速
        def _move(ts: TensorSet) -> TensorSet:
            # 直接操作底层 data 移动，然后 wrap 回 TensorSet
            # 假设 TensorSet 有一个能接受 tensor 的构造器
            # 我们利用 sort_now=False 避免重排序，因为移动设备不改变顺序
            return TensorSet(ts.data.to(device), sort_now=False)

        return DyPlan(
            p2p=[_move(ts) for ts in self.p2p],
            fetch=_move(self.fetch),
            rec_cts=self.rec_cts,  # numpy array 不随 torch device 移动
            send_cts=self.send_cts,  # numpy array 不随 torch device 移动
            writeback=_move(self.writeback) if self.writeback is not None else None,
        )

    def cuda(self) -> DyPlan:
        return self.to(torch.device("cuda"))

    @property
    def device(self) -> torch.device:
        return self.fetch.data.device

    def sort(self) -> DyPlan:
        """
        TensorSet 默认保持内部有序 (sorted invariant)，
        所以这里不需要做任何操作，直接返回 self 即可。
        """
        return self


@dataclass(slots=True, kw_only=True)
class StPlan:
    to_cache: TensorSet
    fetch: TensorSet

    def to(self, device: torch.device) -> StPlan:
        """
        Move internal TensorSets to the target device.
        """

        # Helper: Create new TensorSet on target device without re-sorting
        def _move(ts: TensorSet) -> TensorSet:
            return TensorSet(ts.data.to(device), sort_now=False)

        return StPlan(
            to_cache=_move(self.to_cache),
            fetch=_move(self.fetch),
        )

    def cuda(self) -> StPlan:
        return self.to(torch.device("cuda"))

    def sort(self) -> StPlan:
        return self

    @property
    def device(self) -> torch.device:
        # Access .data to get the underlying tensor's device
        return self.to_cache.data.device
