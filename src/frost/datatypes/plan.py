from __future__ import annotations

from dataclasses import dataclass

import torch
from numpy.typing import NDArray

from frost.utils.tensor_set import TensorSet


@dataclass(kw_only=True)
class PlanSet:
    """
    统一的 PlanSet，直接持有 DynamicPlan 和 StaticPlan。
    """

    mem_plan: DynamicPlan | None = None
    box_plan: DynamicPlan | None = None
    nfeat_plan: StaticPlan | None = None

    def __repr__(self) -> str:
        return f"PlanSet(mem={self.mem_plan}, box={self.box_plan}, nfeat={self.nfeat_plan})"

    def to(self, device: torch.device) -> PlanSet:
        """
        递归地将所有子 Plan 移动到指定设备。
        增加了 non_blocking 参数以支持异步传输。
        """
        return PlanSet(
            # 假设你的 DynamicPlan/StaticPlan 的 .to() 方法也能接收 non_blocking 参数
            mem_plan=self.mem_plan.to(device) if self.mem_plan is not None else None,
            box_plan=self.box_plan.to(device) if self.box_plan is not None else None,
            nfeat_plan=self.nfeat_plan.to(device)
            if self.nfeat_plan is not None
            else None,
        )


@dataclass(kw_only=True)
class DynamicPlan:
    p2p: list[TensorSet]
    fetch: TensorSet
    recv_counts: NDArray
    send_counts: NDArray
    writeback: TensorSet | None = None
    fetch_on_gpu: torch.Tensor | None = (
        None  # 仅在 GPU 上使用，存储 fetch 的 data 部分以加速访问
    )

    def clone(self) -> DynamicPlan:
        """
        深复制 DynamicPlan，包括内部的 TensorSet。
        注意：TensorSet 是不可变的，所以我们可以直接复用它们。
        """
        return DynamicPlan(
            p2p=self.p2p.copy(),  # list 的浅复制足够，因为 TensorSet 是不可变的
            fetch=self.fetch,  # 直接复用同一个 TensorSet 对象
            recv_counts=self.recv_counts.copy(),  # numpy array 需要复制
            send_counts=self.send_counts.copy(),  # numpy array 需要复制
            writeback=self.writeback,  # 直接复用同一个 TensorSet 对象（如果存在）
            fetch_on_gpu=self.fetch_on_gpu,  # 直接复用同一个 tensor 对象（如果存在）
        )

    def __repr__(self) -> str:
        return (
            f"DynamicPlan(p2p={[ts.data.shape for ts in self.p2p]}, "
            f"fetch={self.fetch.data.shape}, "
            f"recv_counts={self.recv_counts.shape}, "
            f"send_counts={self.send_counts.shape}, "
            f"writeback={self.writeback.data.shape if self.writeback is not None else None}, "
            f"fetch_on_gpu={self.fetch_on_gpu.shape if self.fetch_on_gpu is not None else None})"
        )

    def to(self, device: torch.device) -> DynamicPlan:
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

        return DynamicPlan(
            p2p=[_move(ts) for ts in self.p2p],
            fetch=self.fetch,  # fetch should not be moved to cuda
            recv_counts=self.recv_counts,  # numpy array 不随 torch device 移动
            send_counts=self.send_counts,  # numpy array 不随 torch device 移动
            writeback=_move(self.writeback) if self.writeback is not None else None,
            fetch_on_gpu=self.fetch.data.to(device) if self.fetch is not None else None,
        )

    @property
    def device(self) -> torch.device:
        return self.fetch.data.device

    def sort(self) -> DynamicPlan:
        """
        TensorSet 默认保持内部有序 (sorted invariant)，
        所以这里不需要做任何操作，直接返回 self 即可。
        """
        return self


@dataclass(kw_only=True)
class StaticPlan:
    to_cache: TensorSet
    fetch: TensorSet

    def clone(self) -> StaticPlan:
        """
        深复制 StaticPlan，包括内部的 TensorSet。
        注意：TensorSet 是不可变的，所以我们可以直接复用它们。
        """
        return StaticPlan(
            to_cache=self.to_cache,  # 直接复用同一个 TensorSet 对象
            fetch=self.fetch,  # 直接复用同一个 TensorSet 对象
        )

    def __repr__(self) -> str:
        return (
            f"StaticPlan(to_cache={self.to_cache.data.shape}, "
            f"fetch={self.fetch.data.shape})"
        )

    def to(self, device: torch.device) -> StaticPlan:
        """
        Move internal TensorSets to the target device.
        """

        # Helper: Create new TensorSet on target device without re-sorting
        def _move(ts: TensorSet) -> TensorSet:
            return TensorSet(ts.data.to(device), sort_now=False)

        return StaticPlan(
            to_cache=_move(self.to_cache),
            fetch=self.fetch,  # fetch should not be moved to cuda
        )

    def sort(self) -> StaticPlan:
        return self

    @property
    def device(self) -> torch.device:
        # Access .data to get the underlying tensor's device
        return self.to_cache.data.device
