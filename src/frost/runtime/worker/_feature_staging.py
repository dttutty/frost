"""Worker-local NUMA-aware staging ring for static graph features.

The graph RingBuf carries compact sampled IDs.  This module keeps the much
larger feature payload on a separate pipeline: a CPU prefetch thread gathers
the rows requested by a batch into reusable pinned slots, then H2D consumes
the slot and returns it to the ring.  Each worker owns one ring, so its pages
inherit the worker's GPU-local NUMA policy without introducing cross-worker
head/tail contention.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Literal, TypeAlias

import torch

from frost.datatypes.meta import Meta
from frost.runtime._thread_base import ThreadBase, put_shutdown_sentinel

logger = logging.getLogger(__name__)

_MISSING_FEATURE_ROW = torch.iinfo(torch.int64).min


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


def _encode_feature_route(
    requested_nids: torch.Tensor,
    fetched_nids: torch.Tensor,
    cached_nids: torch.Tensor | None,
) -> torch.Tensor:
    """Encode the feature source and row for every requested node on CPU.

    Non-negative entries address ``fetched_nids``. Negative entries encode a
    cache row as ``-row - 1``. ``_MISSING_FEATURE_ROW`` marks graph padding or
    nodes outside the model contract's feature scope.
    """
    if requested_nids.device.type != "cpu":
        raise ValueError("static feature routes must be built from CPU node IDs")

    route = torch.full_like(requested_nids, _MISSING_FEATURE_ROW)
    if fetched_nids.numel() > 0:
        positions = torch.searchsorted(fetched_nids, requested_nids)
        in_bounds = positions < fetched_nids.numel()
        safe_positions = positions.clamp_max(fetched_nids.numel() - 1)
        found = in_bounds & (fetched_nids[safe_positions] == requested_nids)
        route[found] = safe_positions[found]

    if cached_nids is not None and cached_nids.numel() > 0:
        positions = torch.searchsorted(cached_nids, requested_nids)
        in_bounds = positions < cached_nids.numel()
        safe_positions = positions.clamp_max(cached_nids.numel() - 1)
        found = in_bounds & (cached_nids[safe_positions] == requested_nids)
        route[found] = -safe_positions[found] - 1

    return route


@dataclass
class FeatureStageSlot:
    """Reusable buffers owned by one slot of a worker's feature ring."""

    index: int
    node_values: torch.Tensor | None = None
    node_fill_route: torch.Tensor | None = None
    node_cache_route: torch.Tensor | None = None
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
            current = _allocate_staging_tensor(
                _next_capacity(rows), columns, dtype
            )
        return current

    def stage_node_rows(
        self, table: torch.Tensor, row_ids: torch.Tensor
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
        torch.index_select(table, 0, row_ids, out=output)
        return output

    def _stage_route(
        self,
        current: torch.Tensor | None,
        route: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(route.numel())
        if current is None or current.numel() < rows:
            current = _allocate_staging_tensor(
                _next_capacity(rows), 1, torch.int64
            ).view(-1)
        output = current[:rows]
        output.copy_(route)
        return current, output

    def stage_node_routes(
        self,
        *,
        graph_nids: torch.Tensor,
        fetched_nids: torch.Tensor,
        cached_nids: torch.Tensor | None,
        to_cache_nids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage CPU-computed routes for graph fill and next-cache assembly."""
        fill_route = _encode_feature_route(
            graph_nids,
            fetched_nids,
            cached_nids,
        )
        cache_route = _encode_feature_route(
            to_cache_nids,
            fetched_nids,
            cached_nids,
        )
        if torch.any(cache_route == _MISSING_FEATURE_ROW):
            raise ValueError("cache plan contains unavailable node features")

        self.node_fill_route, staged_fill = self._stage_route(
            self.node_fill_route,
            fill_route,
        )
        self.node_cache_route, staged_cache = self._stage_route(
            self.node_cache_route,
            cache_route,
        )
        return staged_fill, staged_cache

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
    node_fill_route: torch.Tensor | None
    node_cache_route: torch.Tensor | None
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
        self._cached_nids: torch.Tensor | None = None

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            payload = self.in_q.get(block=True)
            if payload is None:
                put_shutdown_sentinel(self.out_q, None)
                break
            if payload == "end_of_epoch":
                self._cached_nids = None
                self.out_q.put("end_of_epoch", block=True)
                continue

            meta = payload
            if meta.flat_graph is None:
                raise ValueError("feature prefetch requires a sampled flat graph")
            slot = self.ring.acquire()
            try:
                node_values = None
                node_fill_route = None
                node_cache_route = None
                if self.nfeat is not None:
                    if meta.plan_set is None or meta.plan_set.nfeat_plan is None:
                        raise ValueError(
                            "node features require a static access plan before staging"
                        )
                    nfeat_plan = meta.plan_set.nfeat_plan
                    fetch = nfeat_plan.fetch.data
                    node_values = slot.stage_node_rows(self.nfeat, fetch)
                    node_fill_route, node_cache_route = slot.stage_node_routes(
                        graph_nids=meta.flat_graph.nid,
                        fetched_nids=fetch,
                        cached_nids=self._cached_nids,
                        to_cache_nids=nfeat_plan.to_cache.data,
                    )
                    self._cached_nids = nfeat_plan.to_cache.data

                edge_values = None
                edge_positions = None
                if self.efeat is not None:
                    edge_values, edge_positions = slot.stage_edge_rows(
                        self.efeat,
                        meta.flat_graph.eid,
                        n_src=meta.n_src,
                    )

                self.out_q.put(
                    StagedFeatureBatch(
                        meta=meta,
                        slot=slot,
                        node_values=node_values,
                        node_fill_route=node_fill_route,
                        node_cache_route=node_cache_route,
                        edge_values=edge_values,
                        edge_positions=edge_positions,
                    ),
                    block=True,
                )
            except Exception:
                self.ring.release(slot)
                raise
