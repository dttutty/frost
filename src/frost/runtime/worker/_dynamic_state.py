"""Tensor operations for dynamic memory and mailbox assembly."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from queue import Queue
from typing import Literal

import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.plan import DynamicPlan
from frost.models.contracts import RootExpansion
from frost.shm.mem_mgr import MemBoxStore


@dataclass(kw_only=True)
class MemAndBox:
    """Container pairing a memory entry with its corresponding mailbox entry.

    Attributes:
        mem: Node memory entry (NID + data + timestamp).
        box: Mailbox entry (NID + data + timestamp).
    """

    mem: MemEntry
    box: BoxEntry

    @cached_property
    def is_mem_box_sorted(self) -> bool:
        """Return True if both mem and box entries are sorted by NID."""
        assert self.mem is not None
        assert self.box is not None
        return self.mem.is_sorted and self.box.is_sorted

    def sort_mem_and_box(self, *, shared_nid_order: bool = False) -> MemAndBox:
        """Sort both mem and box entries by NID in-place and return self.

        Returns:
            The same ``MemAndBox`` instance with both entries sorted.
        """
        if self.is_mem_box_sorted:
            return self

        if shared_nid_order and not self.mem.is_sorted and not self.box.is_sorted:
            if self.mem.nid.shape != self.box.nid.shape:
                raise RuntimeError(
                    "Shared memory/mailbox plan produced mismatched delta shapes."
                )
            sorted_indices = torch.argsort(self.mem.nid)
            sorted_nids = self.mem.nid[sorted_indices]
            self.mem = MemEntry(
                nid=sorted_nids,
                data=self.mem.data[sorted_indices],
                ts=self.mem.ts[sorted_indices],
                is_sorted=True,
                is_unique=self.mem.is_unique,
            )
            self.box = BoxEntry(
                nid=sorted_nids,
                data=self.box.data[sorted_indices],
                ts=self.box.ts[sorted_indices],
                is_sorted=True,
                is_unique=self.box.is_unique,
            )
            return self

        if self.mem is not None:
            self.mem = self.mem.sort_by_nid()

            assert self.box is not None
            self.box = self.box.sort_by_nid()

        return self


@dataclass(frozen=True, kw_only=True)
class _AssemblyLookup:
    """Precomputed graph/source positions for one dynamic plan."""

    shared_graph_positions: torch.Tensor
    shared_source_positions: torch.Tensor
    delta_graph_positions: torch.Tensor
    delta_nids: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class DynamicAssemblyLayout:
    """Reusable memory/mailbox lookups for one assembled graph."""

    mem: _AssemblyLookup
    box: _AssemblyLookup


class DynamicStateAssemblerMixin:
    """Assemble shared and peer-provided dynamic state into a FlatGraph."""

    device: torch.device
    shared: MemBoxStore
    boxsize: int
    d_mail: int
    n_neg: int
    root_expansion: RootExpansion
    mem_task_q: Queue | None
    box_task_q: Queue | None
    mem_wr_q: Queue | None
    box_wr_q: Queue | None

    def _strict_assembly_mask(
        self, graph: FlatGraph, n_src: int, n_neg: int
    ) -> torch.Tensor:
        """Return the positions whose mem/box state must be assembled this step.

        Positive-root layouts sample neighbors only for ``src`` and ``pos_dst``;
        the current forward pass needs dynamic state only for the root slice
        ``[src | pos_dst | neg_dst]``.  Neighbor rows are intentionally left
        zero-filled on GPU.
        """
        valid_mask = graph.nid != -1
        if self.root_expansion is RootExpansion.ALL_ROOTS:
            return valid_mask

        n_root = min(graph.nid.numel(), n_src * (2 + n_neg))
        root_mask = torch.zeros_like(valid_mask)
        root_mask[:n_root] = valid_mask[:n_root]
        return root_mask

    def _build_assembly_lookup(
        self,
        plan: DynamicPlan | None,
        required_positions: torch.Tensor,
        required_nids: torch.Tensor,
    ) -> _AssemblyLookup:
        if plan is None or plan.fetch_on_gpu is None or plan.fetch_on_gpu.numel() == 0:
            empty = required_positions.new_empty(0)
            return _AssemblyLookup(
                shared_graph_positions=empty,
                shared_source_positions=empty,
                delta_graph_positions=required_positions,
                delta_nids=required_nids,
            )

        shared_nids = plan.fetch_on_gpu
        positions = torch.searchsorted(shared_nids, required_nids)
        in_bounds = positions < shared_nids.numel()
        safe_positions = positions.clamp_max(shared_nids.numel() - 1)
        found = in_bounds & (shared_nids[safe_positions] == required_nids)
        return _AssemblyLookup(
            shared_graph_positions=required_positions[found],
            shared_source_positions=safe_positions[found],
            delta_graph_positions=required_positions[~found],
            delta_nids=required_nids[~found],
        )

    def _build_assembly_layout(
        self,
        graph: FlatGraph,
        mem_plan: DynamicPlan | None,
        box_plan: DynamicPlan | None,
        n_src: int,
        n_neg: int,
    ) -> DynamicAssemblyLayout:
        required_mask = self._strict_assembly_mask(graph, n_src, n_neg)
        required_positions = torch.nonzero(required_mask, as_tuple=False).squeeze(-1)
        required_nids = graph.nid[required_positions]
        mem_lookup = self._build_assembly_lookup(
            mem_plan, required_positions, required_nids
        )
        box_lookup = (
            mem_lookup
            if box_plan is mem_plan
            else self._build_assembly_lookup(
                box_plan, required_positions, required_nids
            )
        )
        return DynamicAssemblyLayout(mem=mem_lookup, box=box_lookup)

    def _fill_from_shared(
        self,
        plan: DynamicPlan | None,
        lookup: _AssemblyLookup,
        shared_data: torch.Tensor,
        shared_ts: torch.Tensor,
        target_data: torch.Tensor,
        target_ts: torch.Tensor,
    ) -> None:
        """Fetch data from shared memory and scatter matched entries into target tensors.

        Assumptions:
        - plan.fetch.data is on CPU and contains row indices for shared_data/shared_ts
        - plan.fetch_on_gpu is on GPU and contains the corresponding NIDs
        - Only graph.nid entries that exist in plan.fetch_on_gpu will be filled
        - Missing graph.nid entries are ignored because they may belong to remote peers
        """
        if plan is None:
            return
        fetch_nids = plan.fetch.data
        fetch_on_gpu = plan.fetch_on_gpu

        assert fetch_on_gpu is not None, (
            "plan.fetch_on_gpu must be prepared in the plan"
        )
        assert fetch_nids is not None, "plan.fetch.data must not be None"

        if fetch_nids.numel() == 0:
            return

        assert fetch_nids.dim() == 1, "plan.fetch.data must be a 1D tensor"
        assert fetch_on_gpu.dim() == 1, "plan.fetch_on_gpu must be a 1D tensor"
        assert fetch_nids.numel() == fetch_on_gpu.numel(), (
            f"Length mismatch: plan.fetch.data has {fetch_nids.numel()} elements, "
            f"but plan.fetch_on_gpu has {fetch_on_gpu.numel()} elements"
        )
        assert shared_data.shape[0] == shared_ts.shape[0], (
            f"shared_data and shared_ts must have the same first dimension, "
            f"got {shared_data.shape[0]} and {shared_ts.shape[0]}"
        )
        unique_data_gpu = shared_data[fetch_nids].to(self.device)
        unique_ts_gpu = shared_ts[fetch_nids].to(self.device)

        target_data[lookup.shared_graph_positions] = unique_data_gpu[
            lookup.shared_source_positions
        ]
        target_ts[lookup.shared_graph_positions] = unique_ts_gpu[
            lookup.shared_source_positions
        ]

    def _fill_from_shm(
        self,
        kind: Literal["mem", "box"],
        target_data: torch.Tensor,
        target_ts: torch.Tensor,
        graph: FlatGraph,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Fill remote positions directly from shared memory when delta payload is skipped."""
        assert valid_mask.dtype == torch.bool, "valid_mask must be a bool tensor"

        valid_nids = graph.nid[valid_mask]
        if valid_nids.numel() == 0:
            return None, None

        if kind == "mem":
            shared_data = self.shared.mem
            shared_ts = self.shared.mem_ts
        else:
            shared_data = self.shared.box
            shared_ts = self.shared.box_ts

        fetch_nids = valid_nids.to(shared_data.device).long()
        global_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        target_data[global_positions] = shared_data[fetch_nids].to(self.device)
        target_ts[global_positions] = shared_ts[fetch_nids].to(self.device)

        local_positions = torch.arange(
            global_positions.numel(), device=global_positions.device
        )
        return global_positions, local_positions

    def _fill_from_delta(
        self,
        delta_source,
        target_data: torch.Tensor,
        target_ts: torch.Tensor,
        lookup: _AssemblyLookup,
        kind: Literal["mem", "box"],
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Fill target tensors with delta data received from P2P network strictly."""
        if delta_source is None:
            return None, None

        assert delta_source.nid.dim() == 1, "delta_source.nid must be a 1D tensor"

        if lookup.delta_nids.numel() == 0:
            return None, None
        if delta_source.nid.numel() == 0:
            raise RuntimeError(
                f"Delta {kind} source is empty for {lookup.delta_nids.numel()} planned nodes"
            )

        pos = torch.searchsorted(delta_source.nid, lookup.delta_nids)
        in_bound = pos < delta_source.nid.numel()
        pos_safe = pos.clamp(max=delta_source.nid.numel() - 1)
        found = in_bound & (delta_source.nid[pos_safe] == lookup.delta_nids)
        if found.device.type == "cuda":
            torch._assert_async(found.all())
        elif not bool(found.all()):
            missing = lookup.delta_nids[~found]
            raise RuntimeError(
                f"Delta {kind} source is missing {missing.numel()} planned node ids"
            )

        global_positions = lookup.delta_graph_positions
        local_positions = pos_safe

        target_data[global_positions] = delta_source.data[local_positions]
        target_ts[global_positions] = delta_source.ts[local_positions]

        return global_positions, local_positions

    def prepare_shared_features(
        self,
        graph: FlatGraph,
        mem_plan: DynamicPlan | None,
        box_plan: DynamicPlan | None,
        n_src: int,
        n_neg: int | None = None,
    ) -> DynamicAssemblyLayout:
        """Step 1: Initialize tensors and fill local shared memory data."""
        num_nodes = graph.nid.shape[0]

        # Initialize attributes with zeros only once
        if graph.mem is None:
            graph.mem = torch.zeros(
                (num_nodes, self.shared.mem.shape[1]),
                dtype=self.shared.mem.dtype,
                device=self.device,
            )
            graph.mem_ts = torch.zeros(
                num_nodes, dtype=self.shared.mem_ts.dtype, device=self.device
            )

        if graph.box is None:
            graph.box = torch.zeros(
                (num_nodes, self.boxsize, self.d_mail),
                dtype=self.shared.box.dtype,
                device=self.device,
            )
            graph.box_ts = torch.zeros(
                (num_nodes, self.boxsize),
                dtype=self.shared.box_ts.dtype,
                device=self.device,
            )

        batch_n_neg = self.n_neg if n_neg is None else n_neg
        layout = self._build_assembly_layout(
            graph, mem_plan, box_plan, n_src, batch_n_neg
        )

        # Fill local unique data immediately
        if mem_plan is not None:
            assert graph.mem is not None
            assert graph.mem_ts is not None
            self._fill_from_shared(
                mem_plan,
                layout.mem,
                self.shared.mem,
                self.shared.mem_ts,
                graph.mem,
                graph.mem_ts,
            )
        if box_plan is not None:
            assert graph.box is not None
            assert graph.box_ts is not None
            self._fill_from_shared(
                box_plan,
                layout.box,
                self.shared.box,
                self.shared.box_ts,
                graph.box,
                graph.box_ts,
            )

            # Neighborhood-mail optimization: pre-fill box for delta nodes.
            # For boxsize > 1, delta nodes will need old ring-buffer contents
            # on GPU so that _apply_box_mail_delta can do a pure GPU scatter.
            # Pre-load them here (async H2D) instead of synchronous CPU→GPU
            # inside the delta path.
            if self.boxsize > 1:
                delta_positions = layout.box.delta_graph_positions
                delta_nids_gpu = layout.box.delta_nids
                if delta_nids_gpu.numel() > 0:
                    delta_nids_cpu = delta_nids_gpu.cpu().long()
                    # Batch H2D: old ring-buffer contents
                    graph.box[delta_positions] = self.shared.box[delta_nids_cpu].to(
                        self.device
                    )
                    graph.box_ts[delta_positions] = self.shared.box_ts[
                        delta_nids_cpu
                    ].to(self.device)
                    # Cache ring-buffer write positions on GPU for later
                    graph._box_pos_delta = self.shared.box_pos[delta_nids_cpu].to(
                        self.device
                    )
                    graph._box_delta_positions = delta_positions
                else:
                    graph._box_pos_delta = None
                    graph._box_delta_positions = None
        return layout

    def _fill_all_required_from_shm(
        self, graph: FlatGraph, n_src: int, n_neg: int | None = None
    ) -> None:
        assert graph.mem is not None
        assert graph.mem_ts is not None
        assert graph.box is not None
        assert graph.box_ts is not None
        batch_n_neg = self.n_neg if n_neg is None else n_neg
        strict_mask = self._strict_assembly_mask(graph, n_src, batch_n_neg)
        self._fill_from_shm(
            "mem",
            graph.mem,
            graph.mem_ts,
            graph,
            strict_mask,
        )
        self._fill_from_shm(
            "box",
            graph.box,
            graph.box_ts,
            graph,
            strict_mask,
        )

    def _apply_box_mail_delta(
        self,
        delta_box: BoxEntry,
        graph: FlatGraph,
        lookup: _AssemblyLookup,
    ) -> None:
        """Insert one-slot P2P delta mail into a multi-slot box on GPU.

        Relies on ``prepare_shared_features`` having already loaded the old
        ring-buffer contents and ``box_pos`` for delta nodes into
        ``graph.box`` / ``graph._box_pos_delta``.  This method performs
        only GPU-local scatter—no CPU↔GPU transfers on the critical path.
        """
        if delta_box.nid.numel() == 0:
            return

        # Use the pre-cached graph positions and box_pos from shared preparation.
        cached_positions = graph._box_delta_positions
        cached_pos = graph._box_pos_delta
        if cached_positions is not None and cached_pos is not None:
            # Fast path: graph.box already has old ring-buffer contents.
            valid_nids = lookup.delta_nids
            if valid_nids.numel() == 0:
                return

            pos = torch.searchsorted(delta_box.nid, valid_nids)
            in_bound = pos < delta_box.nid.numel()
            pos_safe = pos.clamp(max=delta_box.nid.numel() - 1)
            found_mask = in_bound & (delta_box.nid[pos_safe] == valid_nids)

            if not found_mask.any():
                return

            global_positions = cached_positions[found_mask]
            local_positions = pos_safe[found_mask]

            # Ring-buffer write position (already on GPU)
            # box_pos points to the *next* write slot; write there, matching
            # MemBoxStore.write_mail() semantics.
            write_pos = cached_pos[found_mask] % self.boxsize

            # delta_box.data is [N, 1, d_mail] → squeeze to [M, d_mail]
            mail_data = delta_box.data[local_positions].squeeze(1)
            mail_ts = delta_box.ts[local_positions].squeeze(1)

            # Pure GPU scatter into the pre-filled ring buffer
            assert graph.box is not None
            assert graph.box_ts is not None
            graph.box[global_positions, write_pos] = mail_data
            graph.box_ts[global_positions, write_pos] = mail_ts

            # Clean up cached attributes
            graph._box_pos_delta = None
            graph._box_delta_positions = None
        else:
            # Fallback: pre-fill was skipped (no delta nodes).
            # Nothing to do—graph.box is already complete from plan fetch.
            pass

    def apply_delta_features(
        self,
        graph: FlatGraph,
        delta: MemAndBox | None,
        mem_plan: DynamicPlan | None,
        box_plan: DynamicPlan | None,
        n_src: int,
        n_neg: int | None = None,
        layout: DynamicAssemblyLayout | None = None,
    ) -> None:
        """Step 2: Apply network delta data to positions not covered by local shared."""
        assert graph.mem is not None
        assert graph.mem_ts is not None
        assert graph.box is not None
        assert graph.box_ts is not None

        batch_n_neg = self.n_neg if n_neg is None else n_neg
        if layout is None:
            layout = self._build_assembly_layout(
                graph, mem_plan, box_plan, n_src, batch_n_neg
            )

        assert delta is not None, "delta must be provided"
        global_pos, local_pos = self._fill_from_delta(
            delta.mem, graph.mem, graph.mem_ts, layout.mem, kind="mem"
        )

        # P2P delta box now always has boxsize=1 (single mail).
        # For boxsize > 1, merge P2P mail into the full box from CPU.
        delta_box_slots = delta.box.data.shape[1] if delta.box.data.dim() == 3 else 1
        if self.boxsize > 1 and delta_box_slots == 1:
            self._apply_box_mail_delta(delta.box, graph, layout.box)
        elif (
            global_pos is not None
            and delta.box.nid.numel() > 0
            and layout.mem is layout.box
        ):
            graph.box[global_pos] = delta.box.data[local_pos]
            graph.box_ts[global_pos] = delta.box.ts[local_pos]
        else:
            self._fill_from_delta(
                delta.box, graph.box, graph.box_ts, layout.box, kind="box"
            )

    def _drain_prior_eval_writebacks(self, *, split: str) -> None:
        # Eval is much faster than train because it has no backward / optimizer step.
        # Without draining completed writer work here, the assembler can start the
        # next val/test batch's shared-memory fetch before older mem/box writebacks
        # have become visible, which makes AP timing-sensitive.
        if split == "train":
            return
        for q in (self.mem_task_q, self.box_task_q, self.mem_wr_q, self.box_wr_q):
            if q is not None:
                q.join()
