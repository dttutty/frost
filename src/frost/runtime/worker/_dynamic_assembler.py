"""Dynamic assembler thread for merging local and P2P memory/mailbox data.

For each batch this thread performs a two-phase assembly of node memory and
mailbox tensors:

1. ``prepare_shared_features``: read the locally owned entries from the shared
   CPU ``MemBoxStore`` and scatter them into the FlatGraph tensors on GPU.
2. ``apply_delta_features``: once the P2P delta arrives via ``delta_queue``, fill
   the remaining (remotely owned) entries received from peer GPUs.

The assembled ``(Meta, CudaEvent)`` is forwarded to ``full_q`` for the main
training loop.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from queue import Queue
from typing import Literal, cast

import torch

from frost.aliases.worker import AssembledQ, DeltaQueue, H2DOutQ
from frost.datatypes.graph import FlatGraph
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.plan import DynamicPlan
from frost.protocols.cuda_event_proto import CudaEvent
from frost.runtime._thread_base import StreamThreadBase, put_shutdown_sentinel
from frost.shm.mem_mgr import MemBoxStore

logger = logging.getLogger(__name__)


def _debug_assert_batch_alignment() -> bool:
    flag = os.getenv("FROST_DEBUG_ASSERT_BATCH_ALIGNMENT")
    if flag is None:
        return False
    return flag.lower() in {"1", "true", "yes", "on"}


def _debug_dump_assembly_dir() -> Path | None:
    raw = os.getenv("FROST_DEBUG_DUMP_ASSEMBLY_DIR")
    if raw is None or raw == "":
        return None
    return Path(raw)


def _debug_dump_assembly_batches() -> set[int] | None:
    raw = os.getenv("FROST_DEBUG_DUMP_ASSEMBLY_BATCHES")
    if raw is None or raw.strip() == "":
        return None
    return {
        int(token.strip())
        for token in raw.split(",")
        if token.strip() != ""
    }


def _debug_dump_assembly_full() -> bool:
    flag = os.getenv("FROST_DEBUG_DUMP_ASSEMBLY_FULL")
    if flag is None:
        return False
    return flag.lower() in {"1", "true", "yes", "on"}


def _debug_assembly_checksums_path() -> Path | None:
    raw = os.getenv("FROST_DEBUG_ASSEMBLY_CHECKSUMS_PATH")
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    return Path(value)


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


class DynamicAssembler(StreamThreadBase):
    """Background thread that merges local shared memory and P2P delta into FlatGraph.

    Receives partially-prepared batches from ``h2d_out_q``, fills locally
    owned node memory / mailbox entries from ``MemBoxStore``, then waits for
    the P2P delta from ``delta_queue`` to fill remotely owned entries.  The
    completed batch is forwarded to ``full_q``.

    Args:
        h2d_out_q: Input queue from the H2D thread.
        delta_queue: Queue carrying P2P delta tuples or the sentinel ``"nothing"``.
        full_q: Output queue consumed by the main training loop.
        stop_ev: Shared stop event.
        device: Target CUDA device.
        h2d_sem: Semaphore released by write-back threads to throttle the H2D pipeline.
        mem_box_store: Shared CPU memory store for node memory and mailboxes.
    """

    def __init__(
        self,
        h2d_out_q: H2DOutQ,
        delta_queue: DeltaQueue,
        full_q: AssembledQ,
        stop_ev: threading.Event,
        device: torch.device,
        h2d_sem: threading.Semaphore,  # Optional semaphore for H2D synchronization
        mem_box_store: MemBoxStore,  # Pass shared memory instance here
        n_neg: int = 1,
        sample_4_neg: bool = True,
        use_shm_instead: bool = False,
        mem_task_q: Queue | None = None,
        box_task_q: Queue | None = None,
        mem_wr_q: Queue | None = None,
        box_wr_q: Queue | None = None,
    ) -> None:
        super().__init__(name="dy_assembler", stop_ev=stop_ev)
        self.in_q = h2d_out_q
        self.delta_queue = delta_queue
        self.full_q = full_q
        self.device = device
        self.h2d_sem = h2d_sem
        self.shared = mem_box_store
        self.stop_ev = stop_ev
        self.boxsize = mem_box_store.box.shape[1]
        self.d_mail = mem_box_store.box.shape[2]
        self.n_neg = n_neg
        self.sample_4_neg = sample_4_neg
        self.use_shm_instead = use_shm_instead
        self.mem_task_q = mem_task_q
        self.box_task_q = box_task_q
        self.mem_wr_q = mem_wr_q
        self.box_wr_q = box_wr_q
        self._debug_dump_dir = _debug_dump_assembly_dir()
        self._debug_dump_batches = _debug_dump_assembly_batches()
        self._debug_dump_full = _debug_dump_assembly_full()

    def _strict_assembly_mask(self, graph: FlatGraph, n_src: int) -> torch.Tensor:
        """Return the positions whose mem/box state must be assembled this step.

        APAN layouts sample 1-hop neighbors only for ``src`` and ``pos_dst`` but
        the current forward pass only needs dynamic state for the root slice
        ``[src | pos_dst | neg_dst]``.  Neighbor rows are intentionally left
        zero-filled on GPU.
        """
        valid_mask = graph.nid != -1
        if self.sample_4_neg:
            return valid_mask

        n_root = min(graph.nid.numel(), n_src * (2 + self.n_neg))
        root_mask = torch.zeros_like(valid_mask)
        root_mask[:n_root] = valid_mask[:n_root]
        return root_mask

    def _fill_from_shared(
        self,
        plan: DynamicPlan,
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

        table_size = shared_data.shape[0]
        if __debug__:
            max_id = fetch_nids.max().item()
            min_id = fetch_nids.min().item()

            if min_id < 0 or max_id >= table_size:
                bad_mask = (fetch_nids < 0) | (fetch_nids >= table_size)
                bad_ids = torch.unique(fetch_nids[bad_mask])
                preview = bad_ids[:20].tolist()
                raise IndexError(
                    f"plan.fetch.data contains out-of-bounds indices for shared table. "
                    f"Valid range: [0, {table_size - 1}], "
                    f"actual min={min_id}, max={max_id}. "
                    f"First {len(preview)} bad ids: {preview}"
                )

        unique_data_gpu = shared_data[fetch_nids].to(self.device)
        unique_ts_gpu = shared_ts[fetch_nids].to(self.device)
        unique_nids_gpu = fetch_on_gpu
        if __debug__:
            if unique_nids_gpu.numel() > 1:
                if not torch.all(unique_nids_gpu[1:] > unique_nids_gpu[:-1]):
                    bad_mask = unique_nids_gpu[1:] <= unique_nids_gpu[:-1]
                    bad_pos = torch.nonzero(bad_mask, as_tuple=False).squeeze(-1)
                    preview_pos = bad_pos[:20]
                    preview_vals = unique_nids_gpu[preview_pos].tolist()
                    raise ValueError(
                        "plan.fetch_on_gpu must be strictly increasing after sort "
                        f"(sorted and unique). First problematic values: {preview_vals}"
                    )

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

        if __debug__:
            table_size = shared_data.shape[0]
            max_id = valid_nids.max().item()
            min_id = valid_nids.min().item()
            if min_id < 0 or max_id >= table_size:
                bad_mask = (valid_nids < 0) | (valid_nids >= table_size)
                bad_ids = torch.unique(valid_nids[bad_mask])
                preview = bad_ids[:20].tolist()
                raise IndexError(
                    f"graph.nid contains out-of-bounds node ids for shared {kind}. "
                    f"Valid range: [0, {table_size - 1}], "
                    f"actual min={min_id}, max={max_id}. "
                    f"First {len(preview)} bad ids: {preview}"
                )

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
        if self.use_shm_instead:
            return self._fill_from_shm(
                kind=kind,
                target_data=target_data,
                target_ts=target_ts,
                graph=graph,
                valid_mask=valid_mask,
            )

        if delta_source is None:
            return None, None

        assert valid_mask.dtype == torch.bool, "valid_mask must be a bool tensor"
        assert delta_source.nid.dim() == 1, "delta_source.nid must be a 1D tensor"

        if delta_source.nid.numel() == 0:
            return None, None

        if __debug__:
            if delta_source.nid.numel() > 1:
                if not torch.all(delta_source.nid[1:] > delta_source.nid[:-1]):
                    bad_mask = delta_source.nid[1:] <= delta_source.nid[:-1]
                    bad_pos = torch.nonzero(bad_mask, as_tuple=False).squeeze(-1)
                    preview_pos = bad_pos[:20]
                    preview_vals = delta_source.nid[preview_pos].tolist()
                    raise ValueError(
                        "delta_source.nid must be strictly increasing "
                        f"(sorted and unique). First problematic values: {preview_vals}"
                    )

        valid_nids = graph.nid[valid_mask]
        if valid_nids.numel() == 0:
            return None, None

        pos = torch.searchsorted(delta_source.nid, valid_nids)
        in_bound = pos < delta_source.nid.numel()
        pos_safe = pos.clamp(max=delta_source.nid.numel() - 1)
        if __debug__:
            found_mask = in_bound & (delta_source.nid[pos_safe] == valid_nids)

            if not torch.all(found_mask):
                missing_nids = torch.unique(valid_nids[~found_mask])
                preview = missing_nids[:20].tolist()
                raise ValueError(
                    "Some graph.nid entries selected by valid_mask are missing from delta_source.nid. "
                    f"Missing count: {missing_nids.numel()}. "
                    f"First {len(preview)} missing NIDs: {preview}"
                )

        global_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        local_positions = pos_safe

        target_data[global_positions] = delta_source.data[local_positions]
        target_ts[global_positions] = delta_source.ts[local_positions]

        return global_positions, local_positions

    def prepare_shared_features(
        self,
        graph: FlatGraph,
        mem_plan: DynamicPlan,
        box_plan: DynamicPlan,
        n_src: int,
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

        valid_mask = self._strict_assembly_mask(graph, n_src)

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

            # ----- APAN optimization: pre-fill box for delta nodes -----
            # For boxsize > 1, delta nodes will need old ring-buffer contents
            # on GPU so that _apply_box_mail_delta can do a pure GPU scatter.
            # Pre-load them here (async H2D) instead of synchronous CPU→GPU
            # inside the delta path.
            if self.boxsize > 1 and not self.use_shm_instead:
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

        if __debug__:
            # searchsorted requires sorted keys
            if shared_nids.numel() > 1:
                if not torch.all(shared_nids[1:] >= shared_nids[:-1]):
                    shared_nids = torch.sort(shared_nids).values

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
    ) -> None:
        """Step 2: Apply network delta data to positions not covered by local shared."""
        assert graph.mem is not None
        assert graph.mem_ts is not None
        assert graph.box is not None
        assert graph.box_ts is not None

        strict_mask = self._strict_assembly_mask(graph, n_src)
        mem_delta_mask = self._build_delta_mask(graph, mem_plan, strict_mask)
        box_delta_mask = self._build_delta_mask(graph, box_plan, strict_mask)

        if self.use_shm_instead:
            self._fill_from_delta(
                None,
                graph.mem,
                graph.mem_ts,
                graph,
                mem_delta_mask,
                kind="mem",
            )
            self._fill_from_delta(
                None,
                graph.box,
                graph.box_ts,
                graph,
                box_delta_mask,
                kind="box",
            )
            return

        assert delta is not None, "delta must be provided when use_shm_instead is False"
        global_pos, local_pos = self._fill_from_delta(
            delta.mem, graph.mem, graph.mem_ts, graph, mem_delta_mask, kind="mem"
        )

        # P2P delta box now always has boxsize=1 (single mail).
        # For boxsize > 1 (APAN), merge P2P mail into the full box from CPU.
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

    def _maybe_dump_assembly_state(
        self,
        *,
        meta,
        graph: FlatGraph,
        mem_plan: DynamicPlan | None,
        box_plan: DynamicPlan | None,
        delta: MemAndBox | None,
        n_src: int,
    ) -> None:
        dump_dir = getattr(self, "_debug_dump_dir", None)
        dump_batches = getattr(self, "_debug_dump_batches", None)
        if dump_dir is None or dump_batches is None:
            return
        if meta.mini_batch_idx not in dump_batches:
            return

        strict_mask = self._strict_assembly_mask(graph, n_src)
        mem_delta_mask = self._build_delta_mask(graph, mem_plan, strict_mask)
        box_delta_mask = self._build_delta_mask(graph, box_plan, strict_mask)

        assert graph.mem is not None
        assert graph.mem_ts is not None
        assert graph.box is not None
        assert graph.box_ts is not None

        dump_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mini_batch_idx": int(meta.mini_batch_idx),
            "split": meta.split,
            "n_src": int(meta.n_src),
            "nid": graph.nid.detach().cpu(),
            "ts": graph.ts.detach().cpu(),
            "strict_mask": strict_mask.detach().cpu(),
            "mem_delta_mask": mem_delta_mask.detach().cpu(),
            "box_delta_mask": box_delta_mask.detach().cpu(),
            "mem_fetch_nid": (
                mem_plan.fetch.data.detach().cpu() if mem_plan is not None else None
            ),
            "box_fetch_nid": (
                box_plan.fetch.data.detach().cpu() if box_plan is not None else None
            ),
            "delta_mem_nid": (
                delta.mem.nid.detach().cpu()
                if delta is not None and delta.mem is not None
                else None
            ),
            "delta_box_nid": (
                delta.box.nid.detach().cpu()
                if delta is not None and delta.box is not None
                else None
            ),
            "delta_mem_ts": (
                delta.mem.ts.detach().cpu()
                if delta is not None and delta.mem is not None
                else None
            ),
            "delta_box_ts": (
                delta.box.ts.detach().cpu()
                if delta is not None and delta.box is not None
                else None
            ),
            "delta_mem_checksum": (
                delta.mem.data.detach().float().sum(dim=1).cpu()
                if delta is not None and delta.mem is not None
                else None
            ),
            "delta_box_checksum": (
                delta.box.data.detach().float().reshape(delta.box.data.shape[0], -1).sum(dim=1).cpu()
                if delta is not None and delta.box is not None
                else None
            ),
            "mem_ts": graph.mem_ts.detach().cpu(),
            "box_ts": graph.box_ts.detach().cpu(),
            "mem_checksum": graph.mem.detach().float().cpu().sum(dim=1),
            "box_checksum": graph.box.detach().float().cpu().reshape(graph.box.shape[0], -1).sum(dim=1),
        }
        if getattr(self, "_debug_dump_full", False):
            payload["mem_data"] = graph.mem.detach().cpu()
            payload["box_data"] = graph.box.detach().cpu()
            if delta is not None and delta.mem is not None:
                payload["delta_mem_data"] = delta.mem.data.detach().cpu()
            if delta is not None and delta.box is not None:
                payload["delta_box_data"] = delta.box.data.detach().cpu()
        torch.save(payload, dump_dir / f"batch_{meta.mini_batch_idx}.pt")

    def _maybe_append_assembly_checksums(
        self,
        *,
        meta,
        graph: FlatGraph,
        mem_plan: DynamicPlan | None,
        box_plan: DynamicPlan | None,
        delta: MemAndBox | None,
    ) -> None:
        path = _debug_assembly_checksums_path()
        if path is None:
            return

        def _sum_or_zero(t: torch.Tensor | None) -> float:
            if t is None or t.numel() == 0:
                return 0.0
            return float(t.detach().float().sum().item())

        def _abssum_or_zero(t: torch.Tensor | None) -> float:
            if t is None or t.numel() == 0:
                return 0.0
            return float(t.detach().float().abs().sum().item())

        def _len_or_zero(t: torch.Tensor | None) -> int:
            if t is None:
                return 0
            return int(t.numel())

        record = {
            "mini_batch_idx": int(meta.mini_batch_idx),
            "cur_epoch": int(meta.cur_epoch),
            "split": meta.split,
            "cur_macro_of_split": int(meta.cur_macro_of_split),
            "cur_macro_of_epoch": int(meta.cur_macro_of_epoch),
            "mem_fetch_len": _len_or_zero(
                None if mem_plan is None or mem_plan.fetch is None else mem_plan.fetch.data
            ),
            "box_fetch_len": _len_or_zero(
                None if box_plan is None or box_plan.fetch is None else box_plan.fetch.data
            ),
            "delta_mem_len": _len_or_zero(None if delta is None else delta.mem.nid),
            "delta_box_len": _len_or_zero(None if delta is None else delta.box.nid),
            "mem_fetch_nid_sum": _sum_or_zero(
                None if mem_plan is None or mem_plan.fetch is None else mem_plan.fetch.data
            ),
            "box_fetch_nid_sum": _sum_or_zero(
                None if box_plan is None or box_plan.fetch is None else box_plan.fetch.data
            ),
            "delta_mem_nid_sum": _sum_or_zero(None if delta is None else delta.mem.nid),
            "delta_box_nid_sum": _sum_or_zero(None if delta is None else delta.box.nid),
            "mem_sum": _sum_or_zero(graph.mem),
            "mem_abs_sum": _abssum_or_zero(graph.mem),
            "box_sum": _sum_or_zero(graph.box),
            "box_abs_sum": _abssum_or_zero(graph.box),
            "delta_mem_sum": _sum_or_zero(None if delta is None else delta.mem.data),
            "delta_mem_abs_sum": _abssum_or_zero(
                None if delta is None else delta.mem.data
            ),
            "delta_box_sum": _sum_or_zero(None if delta is None else delta.box.data),
            "delta_box_abs_sum": _abssum_or_zero(
                None if delta is None else delta.box.data
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

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

    def logic(self) -> None:
        """Main assembly loop: merge shared + delta data and forward to full_q.

        For each incoming batch:

        1. Waits for the H2D-ready CUDA event.
        2. Acquires two H2D semaphore slots (one for mem write-back, one for
           box write-back) to prevent the pipeline from over-running the writers.
        3. Calls ``prepare_shared_features`` to fill local data from the store.
        4. Blocks on ``delta_queue`` for the P2P result, then calls
           ``apply_delta_features`` to fill remote data.
        5. Records a CUDA event and puts ``(meta, event)`` into ``full_q``.
        """
        epoch_count = getattr(self, "epoch_count", 0)

        while not self.stop_ev.is_set():
            in_data = self.in_q.get(block=True)

            if in_data is None:
                put_shutdown_sentinel(self.full_q, None)
                break
            elif in_data == "end_of_epoch":
                self.full_q.put("end_of_epoch", block=True)
                continue

            meta, meta_ready_event = in_data
            torch.cuda.current_stream().wait_event(meta_ready_event)

            # ==========================================================
            # OPTIMIZATION: Process local shared memory FIRST
            # We do this BEFORE waiting on the blocking delta_queue
            # ==========================================================
            assert meta.flat_graph is not None
            assert meta.plan_set is not None
            assert meta.plan_set.mem_plan is not None
            assert meta.plan_set.box_plan is not None
            self._drain_prior_eval_writebacks(split=meta.split)
            wait_for_batch = getattr(self.h2d_sem, "wait_for_batch", None)
            if callable(wait_for_batch):
                wait_for_batch(meta.mini_batch_idx)
            else:
                self.h2d_sem.acquire()
                self.h2d_sem.acquire()

            self.prepare_shared_features(
                graph=meta.flat_graph,
                mem_plan=meta.plan_set.mem_plan,
                box_plan=meta.plan_set.box_plan,
                n_src=meta.n_src,
            )

            # Now block and wait for network data
            delta_tuple = self.delta_queue.get(block=True)
            if delta_tuple is None:
                put_shutdown_sentinel(self.full_q, None)
                break

            logger.debug(
                f"DynamicAssembler received delta for batch {meta.mini_batch_idx}, epoch {epoch_count}"
            )

            if delta_tuple != "nothing":
                delta = None
                if not self.use_shm_instead:
                    mem, box, works = delta_tuple
                    assert mem is not None
                    assert box is not None
                    if _debug_assert_batch_alignment():
                        expected_delta_batch_idx = meta.mini_batch_idx - 1
                        if mem.mini_batch_idx != expected_delta_batch_idx:
                            raise RuntimeError(
                                "DynamicAssembler received mem delta with unexpected batch offset: "
                                f"meta={meta.mini_batch_idx}, expected_delta={expected_delta_batch_idx}, "
                                f"actual_delta={mem.mini_batch_idx}"
                            )
                        if box.mini_batch_idx != expected_delta_batch_idx:
                            raise RuntimeError(
                                "DynamicAssembler received box delta with unexpected batch offset: "
                                f"meta={meta.mini_batch_idx}, expected_delta={expected_delta_batch_idx}, "
                                f"actual_delta={box.mini_batch_idx}"
                            )

                    # Wait for P2P network streams to finish receiving
                    for work in works:
                        work.wait()

                    box_data = box.data2recv
                    delta_mem = MemEntry(
                        nid=mem.recv_nid, data=mem.data2recv, ts=mem.ts2recv
                    )
                    delta_box = BoxEntry(
                        nid=box.recv_nid, data=box_data, ts=box.ts2recv
                    )
                    delta = MemAndBox(mem=delta_mem, box=delta_box)
                    delta = delta.sort_mem_and_box()

                # ==========================================================
                # Apply the received delta to the already partially-filled graph
                # ==========================================================
                self.apply_delta_features(
                    graph=meta.flat_graph,
                    delta=delta,
                    mem_plan=meta.plan_set.mem_plan,
                    box_plan=meta.plan_set.box_plan,
                    n_src=meta.n_src,
                )

            self._maybe_dump_assembly_state(
                meta=meta,
                graph=meta.flat_graph,
                mem_plan=meta.plan_set.mem_plan,
                box_plan=meta.plan_set.box_plan,
                delta=delta if delta_tuple != "nothing" else None,
                n_src=meta.n_src,
            )
            self._maybe_append_assembly_checksums(
                meta=meta,
                graph=meta.flat_graph,
                mem_plan=meta.plan_set.mem_plan,
                box_plan=meta.plan_set.box_plan,
                delta=delta if delta_tuple != "nothing" else None,
            )

            # Push to the next queue
            ready_event = cast(CudaEvent, torch.cuda.Event())
            ready_event.record(torch.cuda.current_stream())
            self.full_q.put((meta, ready_event), block=True)
            logger.debug(
                f"DynamicAssembler put full meta for batch {meta.mini_batch_idx}, epoch {epoch_count} into full_q"
            )
