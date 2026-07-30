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

    def sort_mem_and_box(self) -> MemAndBox:
        """Sort both mem and box entries by NID in-place and return self.

        Returns:
            The same ``MemAndBox`` instance with both entries sorted.
        """
        if self.is_mem_box_sorted:
            return self

        if self.mem is not None:
            self.mem = self.mem.sort_by_nid()

            assert self.box is not None
            self.box = self.box.sort_by_nid()

        return self


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

    def _fill_from_shared(
        self,
        plan: DynamicPlan | None,
        shared_data: torch.Tensor,
        shared_ts: torch.Tensor,
        target_data: torch.Tensor,
        target_ts: torch.Tensor,
        graph: FlatGraph,
        valid_mask: torch.Tensor,
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
        assert valid_mask.dtype == torch.bool, "valid_mask must be a bool tensor"

        unique_data_gpu = shared_data[fetch_nids].to(self.device)
        unique_ts_gpu = shared_ts[fetch_nids].to(self.device)
        unique_nids_gpu = fetch_on_gpu

        query_nids = graph.nid[valid_mask]
        if query_nids.numel() == 0:
            return

        idx = torch.searchsorted(unique_nids_gpu, query_nids)
        in_bounds = idx < unique_nids_gpu.numel()
        safe_idx = idx.clamp_max(unique_nids_gpu.numel() - 1)
        found_mask = in_bounds & (unique_nids_gpu[safe_idx] == query_nids)

        if not torch.any(found_mask):
            return

        global_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)[
            found_mask
        ]
        local_positions = safe_idx[found_mask]

        target_data[global_positions] = unique_data_gpu[local_positions]
        target_ts[global_positions] = unique_ts_gpu[local_positions]

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

        fetch_nids = valid_nids.cpu().long()
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
        graph: FlatGraph,
        valid_mask: torch.Tensor,
        kind: Literal["mem", "box"],
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Fill target tensors with delta data received from P2P network strictly."""
        if delta_source is None:
            return None, None

        assert valid_mask.dtype == torch.bool, "valid_mask must be a bool tensor"
        assert delta_source.nid.dim() == 1, "delta_source.nid must be a 1D tensor"

        if delta_source.nid.numel() == 0:
            return None, None

        valid_nids = graph.nid[valid_mask]
        if valid_nids.numel() == 0:
            return None, None

        pos = torch.searchsorted(delta_source.nid, valid_nids)
        in_bound = pos < delta_source.nid.numel()
        pos_safe = pos.clamp(max=delta_source.nid.numel() - 1)
        found = in_bound & (delta_source.nid[pos_safe] == valid_nids)
        if not bool(found.all()):
            missing = valid_nids[~found]
            raise RuntimeError(
                f"Delta source is missing {missing.numel()} planned node ids"
            )

        global_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
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
    ) -> None:
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
        valid_mask = self._strict_assembly_mask(graph, n_src, batch_n_neg)

        # Fill local unique data immediately
        if mem_plan is not None:
            assert graph.mem is not None
            assert graph.mem_ts is not None
            self._fill_from_shared(
                mem_plan,
                self.shared.mem,
                self.shared.mem_ts,
                graph.mem,
                graph.mem_ts,
                graph,
                valid_mask,
            )
        if box_plan is not None:
            assert graph.box is not None
            assert graph.box_ts is not None
            self._fill_from_shared(
                box_plan,
                self.shared.box,
                self.shared.box_ts,
                graph.box,
                graph.box_ts,
                graph,
                valid_mask,
            )

            # Neighborhood-mail optimization: pre-fill box for delta nodes.
            # For boxsize > 1, delta nodes will need old ring-buffer contents
            # on GPU so that _apply_box_mail_delta can do a pure GPU scatter.
            # Pre-load them here (async H2D) instead of synchronous CPU→GPU
            # inside the delta path.
            if self.boxsize > 1:
                box_delta_mask = self._build_delta_mask(graph, box_plan, valid_mask)
                delta_nids_gpu = graph.nid[box_delta_mask]
                if delta_nids_gpu.numel() > 0:
                    delta_nids_cpu = delta_nids_gpu.cpu().long()
                    # Batch H2D: old ring-buffer contents
                    graph.box[box_delta_mask] = self.shared.box[delta_nids_cpu].to(
                        self.device
                    )
                    graph.box_ts[box_delta_mask] = self.shared.box_ts[
                        delta_nids_cpu
                    ].to(self.device)
                    # Cache ring-buffer write positions on GPU for later
                    graph._box_pos_delta = self.shared.box_pos[delta_nids_cpu].to(
                        self.device
                    )
                    graph._box_delta_mask = box_delta_mask
                else:
                    graph._box_pos_delta = None
                    graph._box_delta_mask = None

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

    def _build_delta_mask(
        self,
        graph: FlatGraph,
        plan: DynamicPlan | None,
        required_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute a boolean mask for positions that must be filled from the P2P delta.

        A position requires delta data when:
        * it is marked in ``required_mask`` (or ``graph.nid[i] != -1`` by default), AND
        * the NID is not already covered by the local shared fetch.

        Args:
            graph: The current FlatGraph whose ``nid`` tensor is queried.
            plan: The DynamicPlan holding locally fetched NIDs; if ``None`` all
                valid positions are considered remote.

        Returns:
            Boolean tensor of shape ``[graph.nid.shape[0]]`` where ``True``
            means the position must be filled from the delta.
        """
        graph_nid = graph.nid
        assert graph_nid.dim() == 1, "graph.nid must be 1D"

        base_mask = graph_nid != -1 if required_mask is None else required_mask
        if not torch.any(base_mask):
            return base_mask

        if plan is None or plan.fetch_on_gpu is None or plan.fetch_on_gpu.numel() == 0:
            return base_mask

        shared_nids = plan.fetch_on_gpu
        assert shared_nids.dim() == 1, "plan.fetch_on_gpu must be 1D"

        query_nids = graph_nid[base_mask]
        pos = torch.searchsorted(shared_nids, query_nids)

        in_bound = pos < shared_nids.numel()
        pos_safe = pos.clamp(max=shared_nids.numel() - 1)
        in_shared = in_bound & (shared_nids[pos_safe] == query_nids)

        delta_mask = base_mask.clone()
        delta_mask[torch.nonzero(base_mask, as_tuple=False).squeeze(-1)] = ~in_shared
        return delta_mask

    def _apply_box_mail_delta(
        self,
        delta_box: BoxEntry,
        graph: FlatGraph,
        box_delta_mask: torch.Tensor,
    ) -> None:
        """Insert P2P delta mail (boxsize=1) into pre-filled box on GPU.

        Relies on ``prepare_shared_features`` having already loaded the old
        ring-buffer contents and ``box_pos`` for delta nodes into
        ``graph.box`` / ``graph._box_pos_delta``.  This method performs
        only GPU-local scatter—no CPU↔GPU transfers on the critical path.
        """
        if delta_box.nid.numel() == 0:
            return

        # Use the pre-cached delta mask and box_pos from prepare_shared_features
        cached_mask = graph._box_delta_mask
        cached_pos = graph._box_pos_delta
        if cached_mask is not None and cached_pos is not None:
            # Fast path: graph.box already has old ring-buffer contents.
            valid_nids = graph.nid[cached_mask]
            if valid_nids.numel() == 0:
                return

            pos = torch.searchsorted(delta_box.nid, valid_nids)
            in_bound = pos < delta_box.nid.numel()
            pos_safe = pos.clamp(max=delta_box.nid.numel() - 1)
            found_mask = in_bound & (delta_box.nid[pos_safe] == valid_nids)

            if not found_mask.any():
                return

            global_positions = torch.nonzero(cached_mask, as_tuple=False).squeeze(-1)[
                found_mask
            ]
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
            graph._box_delta_mask = None
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
    ) -> None:
        """Step 2: Apply network delta data to positions not covered by local shared."""
        assert graph.mem is not None
        assert graph.mem_ts is not None
        assert graph.box is not None
        assert graph.box_ts is not None

        batch_n_neg = self.n_neg if n_neg is None else n_neg
        strict_mask = self._strict_assembly_mask(graph, n_src, batch_n_neg)
        mem_delta_mask = self._build_delta_mask(graph, mem_plan, strict_mask)
        box_delta_mask = self._build_delta_mask(graph, box_plan, strict_mask)

        assert delta is not None, "delta must be provided"
        global_pos, local_pos = self._fill_from_delta(
            delta.mem, graph.mem, graph.mem_ts, graph, mem_delta_mask, kind="mem"
        )

        # P2P delta box now always has boxsize=1 (single mail).
        # For boxsize > 1, merge P2P mail into the full box from CPU.
        delta_box_slots = delta.box.data.shape[1] if delta.box.data.dim() == 3 else 1
        if self.boxsize > 1 and delta_box_slots == 1:
            self._apply_box_mail_delta(delta.box, graph, box_delta_mask)
        elif (
            global_pos is not None
            and delta.box.nid.numel() > 0
            and torch.equal(mem_delta_mask, box_delta_mask)
            and delta.mem.nid.numel() == delta.box.nid.numel()
            and torch.equal(delta.mem.nid, delta.box.nid)
        ):
            graph.box[global_pos] = delta.box.data[local_pos]
            graph.box_ts[global_pos] = delta.box.ts[local_pos]
        else:
            self._fill_from_delta(
                delta.box, graph.box, graph.box_ts, graph, box_delta_mask, kind="box"
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
