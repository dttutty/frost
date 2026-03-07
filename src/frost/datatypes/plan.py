from __future__ import annotations

from dataclasses import dataclass

import torch
from numpy.typing import NDArray

from frost.utils.tensor_set import TensorSet


@dataclass(kw_only=True)
class PlanSet:
    """Unified container holding the DynamicPlan and StaticPlan for one batch."""

    mem_plan: DynamicPlan | None = None
    box_plan: DynamicPlan | None = None
    nfeat_plan: StaticPlan | None = None

    def __repr__(self) -> str:
        return f"PlanSet(mem={self.mem_plan}, box={self.box_plan}, nfeat={self.nfeat_plan})"

    def to(self, device: torch.device) -> PlanSet:
        """Recursively move all child plans to the target device."""
        return PlanSet(
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
        None  # GPU-side copy of fetch.data for fast access during assembly
    )

    def clone(self) -> DynamicPlan:
        """
        Shallow-clone DynamicPlan. TensorSets are immutable so they can be shared.
        numpy arrays are copied to avoid aliasing.
        """
        return DynamicPlan(
            p2p=self.p2p.copy(),  # shallow list copy is sufficient; TensorSets are immutable
            fetch=self.fetch,  # reuse the same TensorSet object
            recv_counts=self.recv_counts.copy(),  # numpy arrays must be copied
            send_counts=self.send_counts.copy(),  # numpy arrays must be copied
            writeback=self.writeback,  # reuse the same TensorSet object (if present)
            fetch_on_gpu=self.fetch_on_gpu,  # reuse the same tensor object (if present)
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
        Move internal TensorSets to the target device.
        Creates new TensorSet objects (TensorSets are immutable).
        """

        # Helper: move underlying tensor without re-sorting (device move preserves order)
        def _move(ts: TensorSet) -> TensorSet:
            return TensorSet(ts.data.to(device), sort_now=False)

        return DynamicPlan(
            p2p=[_move(ts) for ts in self.p2p],
            fetch=self.fetch,  # fetch stays on CPU
            recv_counts=self.recv_counts,  # numpy arrays are device-agnostic
            send_counts=self.send_counts,  # numpy arrays are device-agnostic
            writeback=_move(self.writeback) if self.writeback is not None else None,
            fetch_on_gpu=self.fetch.data.to(device) if self.fetch is not None else None,
        )

    @property
    def device(self) -> torch.device:
        return self.fetch.data.device

    def sort(self) -> DynamicPlan:
        """
        TensorSet maintains a sorted invariant internally,
        so no action is needed here.
        """
        return self


@dataclass(kw_only=True)
class StaticPlan:
    to_cache: TensorSet
    fetch: TensorSet

    def clone(self) -> StaticPlan:
        """
        Shallow-clone StaticPlan. TensorSets are immutable so they can be shared.
        """
        return StaticPlan(
            to_cache=self.to_cache,  # reuse the same TensorSet object
            fetch=self.fetch,  # reuse the same TensorSet object
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
            fetch=self.fetch,  # fetch stays on CPU
        )

    def sort(self) -> StaticPlan:
        return self

    @property
    def device(self) -> torch.device:
        # Access .data to get the underlying tensor's device
        return self.to_cache.data.device
