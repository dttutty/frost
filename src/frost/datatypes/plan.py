"""Plan datatypes that describe per-batch memory and feature fetch/writeback schedules."""

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
        mem_plan = self.mem_plan.to(device) if self.mem_plan is not None else None
        box_plan = (
            mem_plan
            if self.box_plan is self.mem_plan
            else self.box_plan.to(device)
            if self.box_plan is not None
            else None
        )
        return PlanSet(
            mem_plan=mem_plan,
            box_plan=box_plan,
            nfeat_plan=self.nfeat_plan.to(device)
            if self.nfeat_plan is not None
            else None,
        )


@dataclass(kw_only=True)
class DynamicPlan:
    """Fetch/writeback schedule for dynamic state (memory or mailbox).

    Produced by the Planner thread on the host and consumed by the worker's
    H2D and P2P syncer threads.

    Attributes:
        p2p: Per-peer lists of node IDs that must be exchanged via P2P
            all-to-all communication.  ``p2p[i]`` holds IDs to send to / recv
            from peer worker ``i``.
        fetch: Node IDs whose state must be fetched from the CPU shared-memory
            store by this worker.
        recv_counts: 1-D int array of shape ``[n_gpu]`` – number of node IDs
            to receive from each peer in the P2P exchange.
        send_counts: 1-D int array of shape ``[n_gpu]`` – number of node IDs
            to send to each peer.
        writeback: Node IDs whose updated state must be written back to the
            CPU store after the forward/backward pass; ``None`` if no writeback
            is needed.
        fetch_on_gpu: GPU copy of ``fetch.data`` for fast index lookups during
            ``DynamicAssembler`` assembly; populated by ``DynamicPlan.to()``.
    """

    p2p: list[TensorSet]
    fetch: TensorSet
    recv_counts: NDArray
    send_counts: NDArray
    writeback: TensorSet | None = None
    fetch_on_gpu: torch.Tensor | None = (
        None  # GPU-side copy of fetch.data for fast access during assembly
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

@dataclass(kw_only=True)
class StaticPlan:
    """Fetch and cache schedule for static node features.

    Attributes:
        to_cache: Node IDs whose features should be stored in the worker's
            local GPU cache after loading.
        fetch: Node IDs whose features must be loaded from shared memory for
            this batch (super-set of ``to_cache`` for the first encounter).
    """

    to_cache: TensorSet
    fetch: TensorSet

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
