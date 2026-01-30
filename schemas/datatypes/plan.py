from __future__ import annotations

from dataclasses import dataclass

import torch
from numpy.typing import NDArray

from tensor_set import TensorSet


@dataclass(kw_only=True)
class BatchNodeStateIO:
    mem_needed: TensorSet
    mem_updated: TensorSet
    mailbox_needed: TensorSet | None
    mailbox_updated: TensorSet | None
    nfeat_needed: TensorSet

    def __repr__(self) -> str:
        return (
            f"BatchNodeStateIO(mem_needed={self.mem_needed.data.shape}, "
            f"mem_updated={self.mem_updated.data.shape}, "
            f"mailbox_needed={self.mailbox_needed.data.shape if self.mailbox_needed is not None else None}, "
            f"mailbox_updated={self.mailbox_updated.data.shape if self.mailbox_updated is not None else None}, "
            f"nfeat_needed={self.nfeat_needed.data.shape})"
        )


@dataclass(kw_only=True)
class PlanSet:
    """
    统一的 PlanSet，直接持有 DyPlan 和 StPlan。
    """

    mem: DyPlan | None = None
    box: DyPlan | None = None
    nfeat: StPlan | None = None

    def __repr__(self) -> str:
        return f"PlanSet(mem={self.mem}, box={self.box}, nfeat={self.nfeat})"

    def to(self, device: torch.device, non_blocking: bool = False) -> PlanSet:
        """
        递归地将所有子 Plan 移动到指定设备。
        增加了 non_blocking 参数以支持异步传输。
        """
        return PlanSet(
            # 假设你的 DyPlan/StPlan 的 .to() 方法也能接收 non_blocking 参数
            mem=self.mem.to(device, non_blocking=non_blocking)
            if self.mem is not None
            else None,
            box=self.box.to(device, non_blocking=non_blocking)
            if self.box is not None
            else None,
            nfeat=self.nfeat.to(device, non_blocking=non_blocking)
            if self.nfeat is not None
            else None,
        )


@dataclass(kw_only=True)
class DyPlan:
    p2p: list[TensorSet]
    fetch: TensorSet
    rec_cts: NDArray
    send_cts: NDArray
    writeback: TensorSet | None = None

    def __repr__(self) -> str:
        return (
            f"DyPlan(p2p={[ts.data.shape for ts in self.p2p]}, "
            f"fetch={self.fetch.data.shape}, "
            f"rec_cts={self.rec_cts.shape}, "
            f"send_cts={self.send_cts.shape}, "
            f"writeback={self.writeback.data.shape if self.writeback is not None else None})"
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> DyPlan:
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
            fetch=self.fetch,  # fetch should not be moved to cuda
            rec_cts=self.rec_cts,  # numpy array 不随 torch device 移动
            send_cts=self.send_cts,  # numpy array 不随 torch device 移动
            writeback=_move(self.writeback) if self.writeback is not None else None,
        )

    @property
    def device(self) -> torch.device:
        return self.fetch.data.device

    def sort(self) -> DyPlan:
        """
        TensorSet 默认保持内部有序 (sorted invariant)，
        所以这里不需要做任何操作，直接返回 self 即可。
        """
        return self


@dataclass(kw_only=True)
class StPlan:
    to_cache: TensorSet
    fetch: TensorSet

    def __repr__(self) -> str:
        return (
            f"StPlan(to_cache={self.to_cache.data.shape}, "
            f"fetch={self.fetch.data.shape})"
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> StPlan:
        """
        Move internal TensorSets to the target device.
        """

        # Helper: Create new TensorSet on target device without re-sorting
        def _move(ts: TensorSet) -> TensorSet:
            return TensorSet(ts.data.to(device), sort_now=False)

        return StPlan(
            to_cache=_move(self.to_cache),
            fetch=self.fetch,  # fetch should not be moved to cuda
        )

    def sort(self) -> StPlan:
        return self

    @property
    def device(self) -> torch.device:
        # Access .data to get the underlying tensor's device
        return self.to_cache.data.device
