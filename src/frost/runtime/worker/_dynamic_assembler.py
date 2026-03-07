"""Dynamic assembler thread for merging local and P2P memory/mailbox data.

For each batch this thread performs a two-phase assembly of node memory and
mailbox tensors:

1. ``prepare_shared_features``: read the locally owned entries from the shared
   CPU ``MemBoxStore`` and scatter them into the FlatGraph tensors on GPU.
2. ``apply_delta_features``: once the P2P delta arrives via ``delta_q``, fill
   the remaining (remotely owned) entries received from peer GPUs.

The assembled ``(Meta, CudaEvent)`` is forwarded to ``full_q`` for the main
training loop.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import cached_property
from typing import cast

import torch

from frost.aliases.worker import AssembledQ, DeltaQ, H2DOutQ
from frost.datatypes.graph import FlatGraph
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.plan import DynamicPlan
from frost.protocols.cuda_event_proto import CudaEvent
from frost.runtime._thread_base import StreamThreadBase
from frost.shm.mem_mgr import MemBoxStore

logger = logging.getLogger(__name__)


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
    the P2P delta from ``delta_q`` to fill remotely owned entries.  The
    completed batch is forwarded to ``full_q``.

    Args:
        h2d_out_q: Input queue from the H2D thread.
        delta_q: Queue carrying P2P delta tuples or the sentinel ``"nothing"``.
        full_q: Output queue consumed by the main training loop.
        stop_ev: Shared stop event.
        device: Target CUDA device.
        h2d_sem: Semaphore released by write-back threads to throttle the H2D pipeline.
        mem_box_store: Shared CPU memory store for node memory and mailboxes.
    """

    def __init__(
        self,
        h2d_out_q: H2DOutQ,
        delta_q: DeltaQ,
        full_q: AssembledQ,
        stop_ev: threading.Event,
        device: torch.device,
        h2d_sem: threading.Semaphore,  # Optional semaphore for H2D synchronization
        mem_box_store: MemBoxStore,  # Pass shared memory instance here
    ) -> None:
        super().__init__(name="dy_assembler", stop_ev=stop_ev)
        self.in_q = h2d_out_q
        self.delta_q = delta_q
        self.full_q = full_q
        self.device = device
        self.h2d_sem = h2d_sem
        self.shared = mem_box_store
        self.stop_ev = stop_ev
        self.boxsize = mem_box_store.box.shape[1]
        self.d_mail = mem_box_store.box.shape[2]

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

    def _fill_from_delta(
        self,
        delta_source,
        target_data: torch.Tensor,
        target_ts: torch.Tensor,
        graph: FlatGraph,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Fill target tensors with delta data received from P2P network strictly."""
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
        self, graph: FlatGraph, mem_plan: DynamicPlan, box_plan: DynamicPlan
    ) -> None:
        """Step 1: Initialize tensors and fill local shared memory data."""
        num_nodes = graph.nid.shape[0]

        # Initialize attributes with zeros only once
        if getattr(graph, "mem", None) is None:
            graph.mem = torch.zeros(
                (num_nodes, self.shared.mem.shape[1]),
                dtype=self.shared.mem.dtype,
                device=self.device,
            )
            graph.mem_ts = torch.zeros(
                num_nodes, dtype=self.shared.mem_ts.dtype, device=self.device
            )

        if getattr(graph, "box", None) is None:
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

        valid_mask = graph.nid != -1

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

    def _build_delta_mask(
        self,
        graph: FlatGraph,
        plan: DynamicPlan | None,
    ) -> torch.Tensor:
        """Compute a boolean mask for positions that must be filled from the P2P delta.

        A position requires delta data when:
        * ``graph.nid[i] != -1`` (valid node), AND
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

        base_mask = graph_nid != -1
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

    def apply_delta_features(
        self,
        graph: FlatGraph,
        delta: MemAndBox,
        mem_plan: DynamicPlan | None,
        box_plan: DynamicPlan | None,
    ) -> None:
        """Step 2: Apply network delta data to positions not covered by local shared."""
        assert graph.mem is not None
        assert graph.mem_ts is not None
        assert graph.box is not None
        assert graph.box_ts is not None

        mem_delta_mask = self._build_delta_mask(graph, mem_plan)
        box_delta_mask = self._build_delta_mask(graph, box_plan)

        global_pos, local_pos = self._fill_from_delta(
            delta.mem, graph.mem, graph.mem_ts, graph, mem_delta_mask
        )

        if (
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
                delta.box, graph.box, graph.box_ts, graph, box_delta_mask
            )

    def logic(self) -> None:
        """Main assembly loop: merge shared + delta data and forward to full_q.

        For each incoming batch:

        1. Waits for the H2D-ready CUDA event.
        2. Acquires two H2D semaphore slots (one for mem write-back, one for
           box write-back) to prevent the pipeline from over-running the writers.
        3. Calls ``prepare_shared_features`` to fill local data from the store.
        4. Blocks on ``delta_q`` for the P2P result, then calls
           ``apply_delta_features`` to fill remote data.
        5. Records a CUDA event and puts ``(meta, event)`` into ``full_q``.
        """
        epoch_count = getattr(self, "epoch_count", 0)

        while not self.stop_ev.is_set():
            in_data = self.in_q.get(block=True)

            if in_data is None:
                self.full_q.put(None, block=True)
                break
            elif in_data == "end_of_epoch":
                self.full_q.put("end_of_epoch", block=True)
                continue

            meta, meta_ready_event = in_data
            torch.cuda.current_stream().wait_event(meta_ready_event)

            # ==========================================================
            # OPTIMIZATION: Process local shared memory FIRST
            # We do this BEFORE waiting on the blocking delta_q
            # ==========================================================
            assert meta.flat_graph is not None
            assert meta.plan_set is not None
            assert meta.plan_set.mem_plan is not None
            assert meta.plan_set.box_plan is not None
            self.h2d_sem.acquire()
            self.h2d_sem.acquire()

            self.prepare_shared_features(
                graph=meta.flat_graph,
                mem_plan=meta.plan_set.mem_plan,
                box_plan=meta.plan_set.box_plan,
            )

            # Now block and wait for network data
            delta_tuple = self.delta_q.get(block=True)

            logger.debug(
                f"DynamicAssembler received delta for batch {meta.batch_idx}, epoch {epoch_count}"
            )

            if delta_tuple != "nothing":
                mem, box, works = delta_tuple

                # Wait for P2P network streams to finish receiving
                for work in works:
                    work.wait()

                # mem = mem.clone()
                # box = box.clone()

                box_data = box.data2recv

                delta_mem = MemEntry(
                    nid=mem.recv_nid, data=mem.data2recv, ts=mem.ts2recv
                )
                delta_box = BoxEntry(nid=box.recv_nid, data=box_data, ts=box.ts2recv)
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
                )

            # Push to the next queue
            ready_event = cast(CudaEvent, torch.cuda.Event())
            ready_event.record(torch.cuda.current_stream())
            self.full_q.put((meta, ready_event), block=True)
            logger.debug(
                f"DynamicAssembler put full meta for batch {meta.batch_idx}, epoch {epoch_count} into full_q"
            )
