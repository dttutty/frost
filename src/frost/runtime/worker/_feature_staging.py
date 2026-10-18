"""Worker-local NUMA-aware staging ring for static graph features.

The graph RingBuf carries compact sampled IDs.  This module keeps the much
larger feature payload on a separate pipeline: a CPU prefetch thread gathers
the rows requested by a batch into reusable pinned slots, then H2D consumes
the slot and returns it to the ring.  Each worker owns one ring, so its pages
inherit the worker's GPU-local NUMA policy without introducing cross-worker
head/tail contention.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Literal, TypeAlias

import torch

from frost.datatypes.meta import Meta
from frost.runtime._thread_base import ThreadBase, put_shutdown_sentinel

logger = logging.getLogger(__name__)

_NODE_GATHER_WORKERS = 8
_MIN_NODE_ROWS_PER_TASK = 2048


def _timing_enabled() -> bool:
    return os.environ.get("FROST_PIPELINE_TIMING_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _next_capacity(required: int) -> int:
    if required < 1:
        return 0
    return 1 << (required - 1).bit_length()


def _allocate_staging_tensor(
    rows: int, columns: int, dtype: torch.dtype
) -> torch.Tensor:
    shape = (rows, columns)
    try:
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    except RuntimeError as exc:
        logger.warning(
            "Pinned feature staging allocation failed; using pageable NUMA-local "
            "memory instead: %s",
            exc,
        )
        return torch.empty(shape, dtype=dtype)


@dataclass
class FeatureStageSlot:
    """Reusable buffers owned by one slot of a worker's feature ring."""

    index: int
    node_values: torch.Tensor | None = None
    edge_values: torch.Tensor | None = None
    edge_positions: torch.Tensor | None = None

    def _ensure_values(
        self,
        current: torch.Tensor | None,
        *,
        rows: int,
        columns: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            current is None
            or current.dtype != dtype
            or current.shape[0] < rows
            or current.shape[1] != columns
        ):
            current = _allocate_staging_tensor(_next_capacity(rows), columns, dtype)
        return current

    def stage_node_rows(
        self,
        table: torch.Tensor,
        row_ids: torch.Tensor,
        *,
        executor: Executor | None = None,
    ) -> torch.Tensor | None:
        rows = int(row_ids.numel())
        if rows == 0:
            return None
        self.node_values = self._ensure_values(
            self.node_values,
            rows=rows,
            columns=int(table.shape[1]),
            dtype=table.dtype,
        )
        output = self.node_values[:rows]
        task_count = min(
            _NODE_GATHER_WORKERS,
            max(1, rows // _MIN_NODE_ROWS_PER_TASK),
        )
        if executor is None or task_count == 1:
            torch.index_select(table, 0, row_ids, out=output)
        else:
            boundaries = [rows * index // task_count for index in range(task_count + 1)]
            futures = [
                executor.submit(
                    torch.index_select,
                    table,
                    0,
                    row_ids[start:end],
                    out=output[start:end],
                )
                for start, end in zip(boundaries, boundaries[1:])
            ]
            for future in futures:
                future.result()
        return output

    def stage_edge_rows(
        self,
        table: torch.Tensor,
        edge_ids: torch.Tensor,
        *,
        n_src: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_mask = edge_ids != -1
        valid_mask[n_src : 3 * n_src] = False
        positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        rows = int(positions.numel())
        if rows == 0:
            raise ValueError("No valid edges found while staging edge features")
        valid_eids = edge_ids[positions]
        max_eid = int(valid_eids.max().item())
        min_eid = int(valid_eids.min().item())
        if min_eid < 0 or max_eid >= table.shape[0]:
            raise ValueError(
                "Edge feature IDs are out of bounds: "
                f"min={min_eid}, max={max_eid}, capacity={table.shape[0]}"
            )
        self.edge_values = self._ensure_values(
            self.edge_values,
            rows=rows,
            columns=int(table.shape[1]),
            dtype=table.dtype,
        )
        output = self.edge_values[:rows]
        torch.index_select(table, 0, valid_eids, out=output)
        self.edge_positions = positions
        return output, positions


class FeatureStageRing:
    """Fixed-depth free-slot ring local to one GPU worker process."""

    def __init__(self, depth: int) -> None:
        if depth < 1:
            raise ValueError(f"feature staging depth must be >= 1, got {depth}")
        self.depth = depth
        self._free: Queue[FeatureStageSlot] = Queue(maxsize=depth)
        for index in range(depth):
            self._free.put(FeatureStageSlot(index=index))

    def acquire(self) -> FeatureStageSlot:
        return self._free.get(block=True)

    def release(self, slot: FeatureStageSlot) -> None:
        self._free.put(slot, block=True)


@dataclass(frozen=True)
class StagedFeatureBatch:
    """A batch paired with one occupied feature-ring slot."""

    meta: Meta
    slot: FeatureStageSlot
    node_values: torch.Tensor | None
    edge_values: torch.Tensor | None
    edge_positions: torch.Tensor | None


FeatureStagePayload: TypeAlias = (
    StagedFeatureBatch | Meta | None | Literal["end_of_epoch"]
)


class FeaturePrefetcher(ThreadBase):
    """Gather static feature rows into a worker-local reusable ring."""

    def __init__(
        self,
        *,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
        in_q: Queue[Meta | None | Literal["end_of_epoch"]],
        out_q: Queue[FeatureStagePayload],
        ring: FeatureStageRing,
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="feature_prefetch", stop_ev=stop_ev)
        self.nfeat = nfeat
        self.efeat = efeat
        self.in_q = in_q
        self.out_q = out_q
        self.ring = ring
        self._node_gather_executor = ThreadPoolExecutor(
            max_workers=_NODE_GATHER_WORKERS,
            thread_name_prefix="feature_gather",
        )
        self._timing_enabled = _timing_enabled()
        self._ring_wait_s_curve: list[float] = []
        self._node_stage_s_curve: list[float] = []
        self._edge_stage_s_curve: list[float] = []
        self._out_q_put_s_curve: list[float] = []
        self._flat_node_rows_curve: list[int] = []
        self._valid_node_rows_curve: list[int] = []
        self._node_fetch_rows_curve: list[int] = []
        self._next_cache_rows_curve: list[int] = []

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._node_gather_executor.shutdown(wait=True, cancel_futures=True)

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            payload = self.in_q.get(block=True)
            if payload is None:
                put_shutdown_sentinel(self.out_q, None)
                break
            if payload == "end_of_epoch":
                self.out_q.put("end_of_epoch", block=True)
                continue

            meta = payload
            if meta.flat_graph is None:
                raise ValueError("feature prefetch requires a sampled flat graph")
            ring_start = time.perf_counter()
            slot = self.ring.acquire()
            ring_end = time.perf_counter()
            try:
                node_values = None
                node_stage_start = time.perf_counter()
                if self.nfeat is not None:
                    if meta.plan_set is None or meta.plan_set.nfeat_plan is None:
                        raise ValueError(
                            "node features require a static access plan before staging"
                        )
                    fetch = meta.plan_set.nfeat_plan.fetch.data
                    node_values = slot.stage_node_rows(
                        self.nfeat,
                        fetch,
                        executor=self._node_gather_executor,
                    )
                node_stage_end = time.perf_counter()

                edge_values = None
                edge_positions = None
                edge_stage_start = time.perf_counter()
                if self.efeat is not None:
                    edge_values, edge_positions = slot.stage_edge_rows(
                        self.efeat,
                        meta.flat_graph.eid,
                        n_src=meta.n_src,
                    )
                edge_stage_end = time.perf_counter()

                flat_node_rows = int(meta.flat_graph.nid.numel())
                valid_node_rows = (
                    int((meta.flat_graph.nid != -1).sum().item())
                    if self._timing_enabled
                    else 0
                )
                nfeat_plan = None if meta.plan_set is None else meta.plan_set.nfeat_plan
                node_fetch_rows = (
                    0 if nfeat_plan is None else int(nfeat_plan.fetch.data.numel())
                )
                next_cache_rows = (
                    0 if nfeat_plan is None else int(nfeat_plan.to_cache.data.numel())
                )

                put_start = time.perf_counter()
                self.out_q.put(
                    StagedFeatureBatch(
                        meta=meta,
                        slot=slot,
                        node_values=node_values,
                        edge_values=edge_values,
                        edge_positions=edge_positions,
                    ),
                    block=True,
                )
                put_end = time.perf_counter()
                if self._timing_enabled:
                    self._ring_wait_s_curve.append(ring_end - ring_start)
                    self._node_stage_s_curve.append(node_stage_end - node_stage_start)
                    self._edge_stage_s_curve.append(edge_stage_end - edge_stage_start)
                    self._out_q_put_s_curve.append(put_end - put_start)
                    self._flat_node_rows_curve.append(flat_node_rows)
                    self._valid_node_rows_curve.append(valid_node_rows)
                    self._node_fetch_rows_curve.append(node_fetch_rows)
                    self._next_cache_rows_curve.append(next_cache_rows)
            except Exception:
                self.ring.release(slot)
                raise

        if self._timing_enabled and self._node_stage_s_curve:
            timing_payload = {
                "system": "frost",
                "batch_count": len(self._node_stage_s_curve),
                "ring_wait_s_curve": [
                    round(value, 6) for value in self._ring_wait_s_curve
                ],
                "node_stage_s_curve": [
                    round(value, 6) for value in self._node_stage_s_curve
                ],
                "edge_stage_s_curve": [
                    round(value, 6) for value in self._edge_stage_s_curve
                ],
                "out_q_put_s_curve": [
                    round(value, 6) for value in self._out_q_put_s_curve
                ],
                "flat_node_rows_curve": self._flat_node_rows_curve,
                "valid_node_rows_curve": self._valid_node_rows_curve,
                "node_fetch_rows_curve": self._node_fetch_rows_curve,
                "next_cache_rows_curve": self._next_cache_rows_curve,
            }
            print(
                "APPENDIX_FEATURE_PREFETCH_TIMING_CURVE "
                f"{json.dumps(timing_payload, sort_keys=True)}"
            )
