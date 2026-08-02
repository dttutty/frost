"""Runtime coordination for publishing model-produced dynamic state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue
from typing import Callable, Iterator, Protocol, cast

import torch
import torch.nn as nn

from frost.datatypes.graph import FlatGraph, HierGraph
from frost.datatypes.mem_mail_box._data_ops import keep_last, mail_to_box
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.meta import Meta
from frost.datatypes.tasks import BoxTask, DirectWritebackRequest, MemTask
from frost.launch.plan import ExecutionPlan
from frost.models.contracts import MemoryState, NeighborhoodMailbox
from frost.runtime.protocols import CudaEvent
from frost.runtime.experimental.dynamic_state import (
    DynamicStatePolicy,
    TransportMode,
    WritebackTiming,
)
from frost.runtime.worker.p2p_syncer.utils._interaction_ops import (
    generate_interaction_mails,
)
from frost.utils.tensor_set import isin_sorted
from frost.utils.tensor_set import TensorSet


class _P2PPlan(Protocol):
    p2p: list[TensorSet]


@dataclass(frozen=True)
class _PublicationContext:
    meta: Meta
    stale_mem: MemEntry | None
    stale_box: BoxEntry | None


class DynamicStateCoordinator:
    """Observe state updates and publish them through the configured runtime path."""

    def __init__(
        self,
        *,
        plan: ExecutionPlan,
        dynamic_state_policy: DynamicStatePolicy,
        mem_task_q: Queue | None,
        box_task_q: Queue | None,
        direct_writeback: Callable[[DirectWritebackRequest], None],
    ) -> None:
        self.plan = plan
        self.dynamic_state_policy = dynamic_state_policy
        self.mem_task_q = mem_task_q
        self.box_task_q = box_task_q
        self.direct_writeback = direct_writeback
        self._active: _PublicationContext | None = None
        state = plan.dynamic_state
        self._neighborhood_mail = isinstance(state, MemoryState) and isinstance(
            state.mailbox, NeighborhoodMailbox
        )

    def attach(self, mem_updater: nn.Module | None) -> None:
        """Publish immediately after the updater, overlapping encoder compute."""
        if mem_updater is not None:
            mem_updater.register_forward_hook(self._after_state_update)

    @contextmanager
    def batch(self, meta: Meta) -> Iterator[None]:
        if self._active is not None:
            raise RuntimeError("DynamicStateCoordinator does not support nested batches.")
        meta.direct_writeback_request = None
        stale_mem: MemEntry | None = None
        stale_box: BoxEntry | None = None
        if self.plan.has_dynamic_state:
            stale_mem, stale_box = self._capture_stale_state(meta)
        self._active = _PublicationContext(meta, stale_mem, stale_box)
        try:
            yield
        finally:
            self._active = None

    def _capture_stale_state(
        self, meta: Meta
    ) -> tuple[MemEntry | None, BoxEntry | None]:
        if self.dynamic_state_policy.transport is not TransportMode.ASYNC_P2P:
            return None, None
        plan_set = meta.plan_set
        if plan_set is None or plan_set.mem_plan is None or plan_set.box_plan is None:
            raise RuntimeError("Dynamic-state batch is missing memory/mailbox plans.")
        if plan_set.mem_plan is plan_set.box_plan:
            send_nids = self._union_planned_p2p_nids(plan_set.mem_plan)
            valid = self._stale_source_mask(meta, send_nids)
            if valid is None:
                return None, None
            return (
                self._select_stale_mem_source(meta, send_nids, valid=valid),
                self._select_stale_box_source(meta, send_nids, valid=valid),
            )
        mem_send_nids = self._union_planned_p2p_nids(plan_set.mem_plan)
        box_send_nids = self._union_planned_p2p_nids(plan_set.box_plan)
        return (
            self._select_stale_mem_source(meta, mem_send_nids),
            self._select_stale_box_source(meta, box_send_nids),
        )

    def _after_state_update(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        if self._active is None:
            raise RuntimeError("Memory updater ran outside coordinator.batch().")
        if not isinstance(output, FlatGraph):
            raise TypeError(f"Memory updater must return FlatGraph, got {type(output)!r}.")
        self._publish(self._active, output)

    @torch.no_grad()
    def _publish(self, context: _PublicationContext, flat_graph: FlatGraph) -> None:
        meta = context.meta
        if (
            flat_graph.detached_mem is None
            or flat_graph.detached_ts is None
            or flat_graph.detached_nid is None
        ):
            raise RuntimeError("Memory updater did not produce detached state.")

        root_new_mem = MemEntry(
            nid=flat_graph.detached_nid,
            ts=flat_graph.detached_ts,
            data=flat_graph.detached_mem,
        )
        expected_state_rows = meta.n_src * 2
        state_shapes = {
            "nid": tuple(root_new_mem.nid.shape),
            "data": tuple(root_new_mem.data.shape),
            "ts": tuple(root_new_mem.ts.shape),
        }
        if any(shape[0] != expected_state_rows for shape in state_shapes.values()):
            raise RuntimeError(
                "Memory updater produced misaligned positive-root state for "
                f"batch={meta.mini_batch_idx}: expected_rows={expected_state_rows}, "
                f"shapes={state_shapes}, flat_nid={tuple(flat_graph.nid.shape)}, "
                f"flat_mem={None if flat_graph.mem is None else tuple(flat_graph.mem.shape)}"
            )
        box_pos_efeat = (
            flat_graph.efeat[: meta.n_src].clone()
            if flat_graph.efeat is not None
            else None
        )
        graph = None
        flat_nid_cpu = None
        if self._neighborhood_mail:
            graph = flat_graph.unflatten(
                n_src=meta.n_src,
                fanout=list(self.plan.sampling.fanout),
                n_neg=meta.n_neg,
                root_expansion=self.plan.sampling.root_expansion,
            )
            flat_nid_cpu = meta.flat_nid_cpu

        if self.dynamic_state_policy.transport is TransportMode.ASYNC_P2P:
            self._publish_async(
                context,
                root_new_mem,
                box_pos_efeat,
                graph,
                flat_nid_cpu,
            )
        elif self.dynamic_state_policy.transport in {
            TransportMode.DIRECT_CPU,
            TransportMode.GPU_LOCAL,
        }:
            self._publish_direct(context, root_new_mem, box_pos_efeat)
        else:
            raise RuntimeError("Unsupported dynamic-state runtime configuration.")

    def _publish_async(
        self,
        context: _PublicationContext,
        root_new_mem: MemEntry,
        box_pos_efeat: torch.Tensor | None,
        graph: HierGraph | None,
        flat_nid_cpu: torch.Tensor | None,
    ) -> None:
        meta = context.meta
        plan_set = meta.plan_set
        if plan_set is None or plan_set.mem_plan is None or plan_set.box_plan is None:
            raise RuntimeError("Dynamic-state batch is missing publication plans.")
        if self.mem_task_q is None or self.box_task_q is None:
            raise RuntimeError("Async dynamic-state queues are not initialized.")

        # Snapshot the mailbox input before exposing the memory entry to its
        # consumer. MemTaskHandler deduplicates MemEntry in place, so cloning
        # after mem_task_q.put() races field-by-field with that mutation and
        # can produce an entry whose nid/data/ts row counts disagree.
        box_root_new_mem = root_new_mem.clone()
        ready_event = cast(CudaEvent, torch.cuda.Event())
        ready_event.record(stream=torch.cuda.current_stream())
        self.mem_task_q.put(
            (
                MemTask(
                    mini_batch_idx=meta.mini_batch_idx,
                    root_new_mem=root_new_mem,
                    stale_mem=context.stale_mem,
                    mem_plan=plan_set.mem_plan,
                    is_last=meta.is_last_macro_of_epoch,
                ),
                ready_event,
            )
        )
        self.box_task_q.put(
            (
                BoxTask(
                    pos_efeat=box_pos_efeat,
                    root_new_mem=box_root_new_mem,
                    stale_box=context.stale_box,
                    mini_batch_idx=meta.mini_batch_idx,
                    box_plan=plan_set.box_plan,
                    is_last=meta.is_last_macro_of_epoch,
                    graph=graph,
                    flat_nid_cpu=flat_nid_cpu,
                ),
                ready_event,
            )
        )

    def _publish_direct(
        self,
        context: _PublicationContext,
        root_new_mem: MemEntry,
        box_pos_efeat: torch.Tensor | None,
    ) -> None:
        if self._neighborhood_mail:
            raise ValueError("Direct writeback does not support neighborhood mail.")
        meta = context.meta
        plan_set = meta.plan_set
        mem_filter = None
        box_filter = None
        if self.dynamic_state_policy.filter_writeback_by_plan:
            if plan_set is None or plan_set.mem_plan is None or plan_set.box_plan is None:
                raise RuntimeError("Plan-filtered writeback requires dynamic plans.")
            if (
                plan_set.mem_plan.writeback is None
                or plan_set.box_plan.writeback is None
            ):
                raise RuntimeError("Dynamic plans are missing writeback node sets.")
            mem_filter = plan_set.mem_plan.writeback.data
            box_filter = plan_set.box_plan.writeback.data
        request = DirectWritebackRequest(
            mini_batch_idx=meta.mini_batch_idx,
            mem_entry=root_new_mem.clone(),
            box_entry=mail_to_box(
                generate_interaction_mails(root_new_mem.clone(), box_pos_efeat)
            ),
            mem_filter_nids=mem_filter,
            box_filter_nids=box_filter,
        )
        if (
            self.dynamic_state_policy.writeback
            is WritebackTiming.AFTER_MEMORY_UPDATE
        ):
            self.direct_writeback(request)
        else:
            meta.direct_writeback_request = request

    @staticmethod
    def _union_planned_p2p_nids(plan: _P2PPlan) -> torch.Tensor:
        p2p = plan.p2p
        if len(p2p) == 0:
            return torch.empty(0, dtype=torch.long)
        if len(p2p) == 1:
            return p2p[0].data
        return torch.unique(torch.cat([per_dst.data for per_dst in p2p]), sorted=True)

    @staticmethod
    def _dedupe_box_entry(entry: BoxEntry) -> BoxEntry:
        if entry.nid.numel() == 0:
            entry.is_sorted = True
            entry.is_unique = True
            return entry
        entry = entry.sort_by_nid()
        if entry.is_unique:
            return entry
        keep = torch.ones(entry.nid.numel(), dtype=torch.bool, device=entry.nid.device)
        if entry.nid.numel() > 1:
            keep[:-1] = entry.nid[:-1] != entry.nid[1:]
        if keep.sum().item() != entry.nid.numel():
            return BoxEntry(
                nid=entry.nid[keep],
                data=entry.data[keep],
                ts=entry.ts[keep],
                is_sorted=True,
                is_unique=True,
            )
        entry.is_unique = True
        return entry

    @staticmethod
    def _stale_source_mask(
        meta: Meta, planned_send_nids: torch.Tensor
    ) -> torch.Tensor | None:
        graph = meta.flat_graph
        if graph is None or planned_send_nids.numel() == 0:
            return None
        valid = (graph.nid != -1) & isin_sorted(graph.nid, planned_send_nids)
        return valid if torch.any(valid) else None

    @staticmethod
    def _select_stale_mem_source(
        meta: Meta,
        planned_send_nids: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> MemEntry | None:
        graph = meta.flat_graph
        if (
            graph is None
            or graph.mem is None
            or graph.mem_ts is None
            or planned_send_nids.numel() == 0
        ):
            return None
        if valid is None:
            valid = DynamicStateCoordinator._stale_source_mask(
                meta, planned_send_nids
            )
        if valid is None:
            return None
        return keep_last(
            MemEntry(nid=graph.nid[valid], data=graph.mem[valid], ts=graph.mem_ts[valid])
        )

    @classmethod
    def _select_stale_box_source(
        cls,
        meta: Meta,
        planned_send_nids: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> BoxEntry | None:
        graph = meta.flat_graph
        if (
            graph is None
            or graph.box is None
            or graph.box_ts is None
            or planned_send_nids.numel() == 0
        ):
            return None
        if valid is None:
            valid = cls._stale_source_mask(meta, planned_send_nids)
        if valid is None:
            return None
        return cls._dedupe_box_entry(
            BoxEntry(nid=graph.nid[valid], data=graph.box[valid], ts=graph.box_ts[valid])
        )
