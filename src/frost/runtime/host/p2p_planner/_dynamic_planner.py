import json
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from frost.runtime.host.types import (
    DynamicPlanInQ,
    DynamicPlanOutQ,
    MetaBundle,
    MetasDyNeed,
)
from frost.datatypes.plan import DynamicPlan
from frost.runtime._thread_base import ThreadBase, put_shutdown_sentinel
from frost.seq_sampler import _core as _native
from frost.utils.tensor_set import TensorSet


def _timing_enabled() -> bool:
    return os.environ.get("FROST_HOST_TIMING_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class _OwnershipIndex:
    """Sorted node IDs and their per-rank availability.

    ``membership[node_idx, rank]`` is true when that rank owns or has the node.
    Building this once lets routing query every receiver in one ``searchsorted``
    instead of running ``torch.isin`` for every sender/receiver pair.
    """

    nodes: torch.Tensor
    membership: torch.Tensor

    @property
    def n_gpus(self) -> int:
        return int(self.membership.shape[1])


@dataclass(frozen=True)
class _OwnedNodeIndex:
    """Sorted node IDs with exactly one selected owner per node."""

    nodes: torch.Tensor
    owners: torch.Tensor
    n_gpus: int
    dense_owners: torch.Tensor | None = None


_DENSE_OWNER_MAX_NODES = 4_000_000
_DENSE_OWNER_MAX_SPARSITY = 16
_NATIVE_ROUTE_MIN_ENTRIES = 80_000
_NATIVE_ROUTE_MIN_MAX_NODE = 65_536
_NATIVE_ROUTE_MIN_RANKS = 5


def _reference_tensor(*groups: list[TensorSet]) -> torch.Tensor:
    for group in groups:
        for tensor_set in group:
            if tensor_set.data.numel() > 0:
                return tensor_set.data
    for group in groups:
        if group:
            return group[0].data
    return torch.empty(0, dtype=torch.long)


def _coerce_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tensor.device != reference.device or tensor.dtype != reference.dtype:
        return tensor.to(device=reference.device, dtype=reference.dtype)
    return tensor


def _empty_set(reference: torch.Tensor) -> TensorSet:
    return TensorSet.empty_like(reference)


def _empty_route_matrix(n_gpus: int, reference: torch.Tensor) -> list[list[TensorSet]]:
    empty = _empty_set(reference)
    return [[empty for _ in range(n_gpus)] for _ in range(n_gpus)]


def _build_ownership_index(
    sets_by_rank: list[TensorSet], reference: torch.Tensor | None = None
) -> _OwnershipIndex:
    """Batch all per-rank sets into one sorted node-to-rank membership index."""
    n_gpus = len(sets_by_rank)
    ref = reference if reference is not None else _reference_tensor(sets_by_rank)
    nonempty = [
        (rank, _coerce_like(tensor_set.data, ref))
        for rank, tensor_set in enumerate(sets_by_rank)
        if tensor_set.data.numel() > 0
    ]
    if not nonempty:
        return _OwnershipIndex(
            nodes=torch.empty(0, dtype=ref.dtype, device=ref.device),
            membership=torch.empty((0, n_gpus), dtype=torch.bool, device=ref.device),
        )

    nodes = torch.cat([data for _, data in nonempty])
    owners = torch.cat(
        [
            torch.full((data.numel(),), rank, dtype=torch.long, device=ref.device)
            for rank, data in nonempty
        ]
    )
    order = torch.argsort(nodes, stable=True)
    sorted_nodes = nodes[order]
    sorted_owners = owners[order]
    unique_nodes, inverse = torch.unique_consecutive(sorted_nodes, return_inverse=True)
    membership = torch.zeros(
        (unique_nodes.numel(), n_gpus), dtype=torch.bool, device=ref.device
    )
    membership[inverse, sorted_owners] = True
    return _OwnershipIndex(nodes=unique_nodes, membership=membership)


def _concat_ranked_sets(
    sets_by_rank: list[TensorSet], reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Concatenate sorted sets with rank labels and prefix offsets."""
    chunks: list[torch.Tensor] = []
    rank_chunks: list[torch.Tensor] = []
    offsets = [0]
    for rank, tensor_set in enumerate(sets_by_rank):
        data = _coerce_like(tensor_set.data, reference)
        chunks.append(data)
        rank_chunks.append(
            torch.full((data.numel(),), rank, dtype=torch.long, device=reference.device)
        )
        offsets.append(offsets[-1] + data.numel())
    if not chunks or offsets[-1] == 0:
        empty = torch.empty(0, dtype=reference.dtype, device=reference.device)
        empty_ranks = torch.empty(0, dtype=torch.long, device=reference.device)
        return empty, empty_ranks, offsets
    return torch.cat(chunks), torch.cat(rank_chunks), offsets


def _lookup_membership(
    index: _OwnershipIndex, query: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(positions, valid)`` for sorted-index membership queries."""
    if query.numel() == 0 or index.nodes.numel() == 0:
        return (
            torch.zeros(query.numel(), dtype=torch.long, device=query.device),
            torch.zeros(query.numel(), dtype=torch.bool, device=query.device),
        )
    positions = torch.searchsorted(index.nodes, query)
    in_bounds = positions < index.nodes.numel()
    safe_positions = positions.clamp_max(index.nodes.numel() - 1)
    valid = in_bounds & (index.nodes[safe_positions] == query)
    return safe_positions, valid


def _route_matrix_from_entries(
    *,
    nodes: torch.Tensor,
    owners: torch.Tensor,
    receivers: torch.Tensor,
    n_gpus: int,
    reference: torch.Tensor,
) -> list[list[TensorSet]]:
    """Group routed nodes into the established ``[sender][receiver]`` layout."""
    routes = _empty_route_matrix(n_gpus, reference)
    if nodes.numel() == 0:
        return routes

    pair_keys = owners * n_gpus + receivers
    # Entries originate from receiver-major concatenations of sorted TensorSets.
    # A stable pair sort therefore preserves ascending node order inside each pair.
    order = torch.argsort(pair_keys, stable=True)
    sorted_keys = pair_keys[order]
    sorted_nodes = nodes[order]
    keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)

    offset = 0
    for key, count in zip(keys.tolist(), counts.tolist()):
        sender, receiver = divmod(int(key), n_gpus)
        routes[sender][receiver] = TensorSet(
            sorted_nodes[offset : offset + count],
            sort_now=False,
            ensure_unique=False,
        )
        offset += count
    return routes


def _compute_transfer_plan_from_index(
    needs: list[TensorSet], index: _OwnershipIndex
) -> tuple[list[TensorSet], list[list[TensorSet]]]:
    """Route all receiver needs through one batched ownership lookup."""
    n_gpus = len(needs)
    if index.n_gpus != n_gpus:
        raise ValueError(
            f"Ownership index has {index.n_gpus} ranks, expected {n_gpus}."
        )
    reference = _reference_tensor(needs) if needs else index.nodes
    query_nodes, receivers, _ = _concat_ranked_sets(needs, reference)
    positions, valid = _lookup_membership(index, query_nodes)

    if query_nodes.numel() == 0 or index.nodes.numel() == 0:
        matched = torch.zeros(
            (query_nodes.numel(), n_gpus),
            dtype=torch.bool,
            device=reference.device,
        )
    else:
        matched = index.membership[positions] & valid.unsqueeze(1)

    query_indices, owners = torch.nonzero(matched, as_tuple=True)
    routes = _route_matrix_from_entries(
        nodes=query_nodes[query_indices],
        owners=owners,
        receivers=receivers[query_indices],
        n_gpus=n_gpus,
        reference=reference,
    )

    needed_nodes = torch.zeros(
        index.nodes.numel(), dtype=torch.bool, device=reference.device
    )
    if torch.any(valid):
        needed_nodes[positions[valid]] = True
    leftovers = [
        TensorSet(
            index.nodes[index.membership[:, rank] & ~needed_nodes],
            sort_now=False,
            ensure_unique=False,
        )
        for rank in range(n_gpus)
    ]
    return leftovers, routes


def _route_from_owned_index(
    needs: list[TensorSet], index: _OwnedNodeIndex
) -> list[list[TensorSet]]:
    """Fast path for the planner's one-owner-per-node update index."""
    n_gpus = len(needs)
    if index.n_gpus != n_gpus:
        raise ValueError(
            f"Owned-node index has {index.n_gpus} ranks, expected {n_gpus}."
        )

    reference = _reference_tensor(needs) if needs else index.nodes
    query_nodes, receivers, _ = _concat_ranked_sets(needs, reference)
    if query_nodes.numel() == 0 or index.nodes.numel() == 0:
        valid = torch.zeros(query_nodes.numel(), dtype=torch.bool, device=reference.device)
        query_owners = torch.zeros(
            query_nodes.numel(), dtype=torch.long, device=reference.device
        )
    elif index.dense_owners is not None:
        in_bounds = (query_nodes >= 0) & (query_nodes < index.dense_owners.numel())
        safe_nodes = query_nodes.clamp(0, index.dense_owners.numel() - 1).long()
        query_owners = index.dense_owners[safe_nodes].long()
        valid = in_bounds & (query_owners >= 0)
    else:
        positions = torch.searchsorted(index.nodes, query_nodes)
        in_bounds = positions < index.nodes.numel()
        positions = positions.clamp_max(index.nodes.numel() - 1)
        valid = in_bounds & (index.nodes[positions] == query_nodes)
        query_owners = index.owners[positions]

    return _route_matrix_from_entries(
        nodes=query_nodes[valid],
        owners=query_owners[valid],
        receivers=receivers[valid],
        n_gpus=n_gpus,
        reference=reference,
    )


def compute_p2p_transfer_plan(
    needs: list[TensorSet],
    avails: list[TensorSet],
) -> tuple[
    list[TensorSet],  # leftovers_on_gpu
    list[list[TensorSet]],  # p2p_transfer_plan
]:
    """Sender-centric fresh P2P plan using one batched ownership lookup."""
    if len(needs) != len(avails):
        raise ValueError(
            f"needs and avails must have equal rank counts, got {len(needs)} and {len(avails)}."
        )
    reference = _reference_tensor(needs, avails)
    index = _build_ownership_index(avails, reference)
    return _compute_transfer_plan_from_index(needs, index)


def handle_remaining(
    needs: list[TensorSet],
    avails: list[TensorSet],
    gpus_distance: np.ndarray,
) -> tuple[list[TensorSet], list[list[TensorSet]]]:
    """Receiver-centric stale P2P plan: cover unfulfilled needs from stale GPU memory,
    preferring the nearest sender (greedy matching by distance).

    Args:
        needs:  per-GPU sets of nodes still needed after fresh P2P is subtracted.
        avails: per-GPU stale memory (full fetch minus fresh updates).
        gpus_distance: NxN distance matrix from Planner.

    Returns:
        remaining_needs: nodes not covered by any GPU (must be fetched from CPU).
        stale_p2p:       [src][dst] transfer plan for stale nodes.
    """
    n_gpus = len(gpus_distance)
    if gpus_distance.shape != (n_gpus, n_gpus):
        raise ValueError("gpus_distance must be a square matrix.")
    if len(needs) != n_gpus or len(avails) != n_gpus:
        raise ValueError(
            "needs, avails, and gpus_distance must describe the same rank count."
        )
    if n_gpus == 0:
        return [], []

    reference = _reference_tensor(needs, avails)
    index = _build_ownership_index(avails, reference)
    query_nodes, receivers, offsets = _concat_ranked_sets(needs, reference)
    positions, valid = _lookup_membership(index, query_nodes)
    chosen_owners = torch.full(
        (query_nodes.numel(),), -1, dtype=torch.long, device=reference.device
    )

    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() > 0:
        valid_receivers = receivers[valid_indices]
        available = index.membership[positions[valid_indices]]
        # Stable source-rank ordering preserves the old tie-break within each
        # ascending distance bucket.
        priority = torch.from_numpy(
            np.argsort(gpus_distance, axis=0, kind="stable").astype(np.int64)
        ).to(reference.device)
        candidates = priority[:, valid_receivers].T
        available_by_priority = available.gather(1, candidates)
        has_owner = torch.any(available_by_priority, dim=1)
        first_available = torch.argmax(available_by_priority.to(torch.int8), dim=1)
        selected = candidates.gather(1, first_available.unsqueeze(1)).squeeze(1)
        chosen_owners[valid_indices[has_owner]] = selected[has_owner]

    routed = chosen_owners >= 0
    routes = _route_matrix_from_entries(
        nodes=query_nodes[routed],
        owners=chosen_owners[routed],
        receivers=receivers[routed],
        n_gpus=n_gpus,
        reference=reference,
    )
    remaining = [
        TensorSet(
            query_nodes[offsets[rank] : offsets[rank + 1]][
                ~routed[offsets[rank] : offsets[rank + 1]]
            ],
            sort_now=False,
            ensure_unique=False,
        )
        for rank in range(n_gpus)
    ]
    return remaining, routes


def deduplicate_updates(
    updates: list[TensorSet], prefer_higher_rank: bool = True
) -> tuple[list[TensorSet], TensorSet]:
    """Deterministic deduplication: when prefer_higher_rank=True,
    higher-rank GPU wins conflicting node ownership."""
    owned, all_updates, _ = _deduplicate_updates_with_index(
        updates, prefer_higher_rank=prefer_higher_rank
    )
    return owned, all_updates


def _deduplicate_updates_with_index(
    updates: list[TensorSet], prefer_higher_rank: bool = True
) -> tuple[list[TensorSet], TensorSet, _OwnedNodeIndex]:
    """Resolve updates directly into a compact node-to-owner map."""
    n_gpus = len(updates)
    reference = _reference_tensor(updates)
    n_inputs = sum(update.data.numel() for update in updates)
    if n_gpus == 0 or n_inputs == 0:
        empty = _empty_set(reference)
        index = _OwnedNodeIndex(
            nodes=empty.data,
            owners=torch.empty(0, dtype=torch.long, device=reference.device),
            n_gpus=n_gpus,
        )
        return [_empty_set(reference) for _ in range(n_gpus)], empty, index

    max_node = max(int(update.data[-1]) for update in updates if update.data.numel())
    dense_span = max_node + 1
    use_dense = (
        max_node >= 0
        and dense_span <= _DENSE_OWNER_MAX_NODES
        and dense_span <= max(65_536, n_inputs * _DENSE_OWNER_MAX_SPARSITY)
    )

    dense_owners: torch.Tensor | None = None
    if use_dense:
        rank_order = range(n_gpus) if prefer_higher_rank else range(n_gpus - 1, -1, -1)
        dense_owners = torch.full(
            (dense_span,), -1, dtype=torch.int16, device=reference.device
        )
        for rank in rank_order:
            update_tensor = _coerce_like(updates[rank].data, reference)
            dense_owners[update_tensor.long()] = rank
        owned_nodes = (
            torch.nonzero(dense_owners >= 0, as_tuple=False)
            .flatten()
            .to(reference.dtype)
        )
        selected_owners = dense_owners[owned_nodes.long()].long()
    else:
        nodes, owners, _ = _concat_ranked_sets(updates, reference)
        order = torch.argsort(nodes, stable=True)
        sorted_nodes = nodes[order]
        sorted_owners = owners[order]
        if prefer_higher_rank:
            selected = torch.cat(
                (
                    sorted_nodes[1:] != sorted_nodes[:-1],
                    torch.ones(1, dtype=torch.bool, device=reference.device),
                )
            )
        else:
            selected = torch.cat(
                (
                    torch.ones(1, dtype=torch.bool, device=reference.device),
                    sorted_nodes[1:] != sorted_nodes[:-1],
                )
            )
        owned_nodes = sorted_nodes[selected]
        selected_owners = sorted_owners[selected]
    owned_index = _OwnedNodeIndex(
        nodes=owned_nodes,
        owners=selected_owners,
        n_gpus=n_gpus,
        dense_owners=dense_owners,
    )
    owned = [
        TensorSet(
            owned_nodes[selected_owners == rank],
            sort_now=False,
            ensure_unique=False,
        )
        for rank in range(n_gpus)
    ]
    all_updates = TensorSet(owned_nodes, sort_now=False, ensure_unique=False)
    return owned, all_updates, owned_index


def _deduplicate_and_route_updates(
    updates: list[TensorSet],
    needs: list[TensorSet],
    prefer_higher_rank: bool = True,
) -> tuple[list[TensorSet], TensorSet, list[list[TensorSet]]]:
    """Run the common ownership-and-routing path in one serial native pass."""
    if len(updates) != len(needs):
        raise ValueError(
            f"updates and needs must have equal rank counts, got {len(updates)} "
            f"and {len(needs)}."
        )
    n_entries = sum(
        tensor_set.data.numel()
        for group in (updates, needs)
        for tensor_set in group
    )
    max_node = max(
        (
            int(tensor_set.data[-1])
            for group in (updates, needs)
            for tensor_set in group
            if tensor_set.data.numel() > 0
        ),
        default=-1,
    )
    use_native = (
        n_entries >= _NATIVE_ROUTE_MIN_ENTRIES
        or max_node >= _NATIVE_ROUTE_MIN_MAX_NODE
        or len(updates) >= _NATIVE_ROUTE_MIN_RANKS
    )
    native_compatible = use_native and hasattr(_native, "plan_owned_routes") and all(
        tensor_set.data.device.type == "cpu"
        and tensor_set.data.dtype == torch.long
        for group in (updates, needs)
        for tensor_set in group
    )
    if native_compatible:
        owned_arrays, all_nodes_array, route_arrays = _native.plan_owned_routes(
            [tensor_set.data.numpy() for tensor_set in updates],
            [tensor_set.data.numpy() for tensor_set in needs],
            prefer_higher_rank,
        )
        owned = [
            TensorSet(torch.from_numpy(nodes), sort_now=False, ensure_unique=False)
            for nodes in owned_arrays
        ]
        all_nodes = TensorSet(
            torch.from_numpy(all_nodes_array), sort_now=False, ensure_unique=False
        )
        routes = [
            [
                TensorSet(
                    torch.from_numpy(nodes), sort_now=False, ensure_unique=False
                )
                for nodes in row
            ]
            for row in route_arrays
        ]
        return owned, all_nodes, routes

    reference = _reference_tensor(updates, needs)
    empty = _empty_set(reference)
    owned = [empty for _ in updates]
    all_nodes = empty
    rank_order = (
        range(len(updates) - 1, -1, -1)
        if prefer_higher_rank
        else range(len(updates))
    )
    for rank in rank_order:
        owned[rank] = updates[rank] - all_nodes
        all_nodes = all_nodes | updates[rank]

    routes = _empty_route_matrix(len(updates), reference)
    for sender, available in enumerate(owned):
        for receiver, need in enumerate(needs):
            routed = available & need
            if routed.data.numel() > 0:
                routes[sender][receiver] = routed
    return owned, all_nodes, routes


class DynamicPlanner(ThreadBase):
    """Per-batch P2P and writeback planner for memory/mailbox.

    Args:
        if_carry_over_route: When True, unfulfilled next-batch needs are served
            from stale GPU memory (nearest-sender greedy match) before falling
            back to CPU fetch. When False (default), all unfulfilled needs go
            straight to CPU fetch.
    """

    gpus_distance: np.ndarray
    n_gpus: int

    def __init__(
        self,
        gpus_distance: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        if_carry_over_route: bool = False,
        output_dir: str | None = None,
        plan_kind: str = "dynamic",
        persist_all_updates_to_cpu: bool = False,
        current_batch_only: bool = False,
    ) -> None:
        super().__init__(name="dynamic_planner", stop_ev=stop_ev)
        del output_dir
        self.plan_kind = plan_kind
        self.gpus_distance = gpus_distance
        self.n_gpus = len(self.gpus_distance)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.if_carry_over_route = if_carry_over_route
        self.pending: MetasDyNeed | None = None
        self.persist_all_updates_to_cpu = persist_all_updates_to_cpu
        self.current_batch_only = current_batch_only
        self._timing_enabled = _timing_enabled()
        self._plan_s_curve: list[float] = []
        self._out_q_put_s_curve: list[float] = []

    def _put_plan(self, plan: list[DynamicPlan]) -> None:
        start = time.perf_counter()
        self.out_q.put(plan)
        if self._timing_enabled:
            self._out_q_put_s_curve.append(time.perf_counter() - start)

    def _emit_timing(self) -> None:
        if not self._timing_enabled or not self._plan_s_curve:
            return
        payload = {
            "system": "frost",
            "plan_kind": self.plan_kind,
            "batch_count": len(self._plan_s_curve),
            "plan_s_curve": self._plan_s_curve,
            "out_q_put_s_curve": self._out_q_put_s_curve,
        }
        print(
            "APPENDIX_DYNAMIC_PLANNER_TIMING_CURVE "
            f"{json.dumps(payload, sort_keys=True)}"
        )

    def _plan_current_batch_only(
        self,
        all_list: list[TensorSet],
        update_list: list[TensorSet],
    ) -> list[DynamicPlan]:
        deduplicate_update_list, all_update = deduplicate_updates(update_list)
        empty_ts = TensorSet(
            torch.tensor([], dtype=all_update.data.dtype, device=all_update.data.device)
        )
        p2p: list[list[TensorSet]] = [
            [empty_ts for _ in range(self.n_gpus)] for _ in range(self.n_gpus)
        ]
        zeros = np.zeros(self.n_gpus, dtype=np.int32)
        return [
            DynamicPlan(
                p2p=p2p[src],
                fetch=all_list[src],
                writeback=deduplicate_update_list[src],
                recv_counts=zeros.copy(),
                send_counts=zeros.copy(),
            )
            for src in range(self.n_gpus)
        ]

    def logic(self) -> None:
        """Main planning loop: consume (needs, updates, raw_need_counts, metas) tuples and emit DynamicPlans.

        Uses a one-step lookahead: each incoming batch is held as ``pending`` until
        the *next* batch arrives (or an end-of-epoch / shutdown sentinel), so that
        fresh P2P transfers can be pre-computed based on what nodes the next batch
        will need.  Sentinel handling:

        - ``None``: forward ``None`` to the output queue and exit.
        - ``"end_of_epoch"``: plan the pending batch with empty next-batch sets.
        """
        if self.current_batch_only:
            while not self.stop_ev.is_set():
                in_q_data = self.in_q.get(block=True)
                if in_q_data is None:
                    put_shutdown_sentinel(self.out_q, None)
                    break
                if in_q_data == "end_of_epoch":
                    continue
                all_needs, all_pos_root, _raw_need_counts, _metas = in_q_data
                start = time.perf_counter()
                plan = self._plan_current_batch_only(all_needs, all_pos_root)
                if self._timing_enabled:
                    self._plan_s_curve.append(time.perf_counter() - start)
                self._put_plan(plan)
            self._emit_timing()
            return

        while not self.stop_ev.is_set():
            in_q_data = self.in_q.get(block=True)

            if in_q_data is None:
                put_shutdown_sentinel(self.out_q, None)
                break

            elif in_q_data == "end_of_epoch":
                assert self.pending is not None
                all_needs, all_pos_root, raw_need_counts, metas = self.pending

                if all_needs and len(all_needs) > 0:
                    ref_t = all_needs[0].data
                    empty_tensor = torch.tensor(
                        [], dtype=ref_t.dtype, device=ref_t.device
                    )
                else:
                    empty_tensor = torch.tensor([], dtype=torch.long)
                empty_ts = TensorSet(empty_tensor)

                # Last batch of the epoch: no next-batch needs or updates.
                next_needs: list[TensorSet] = [empty_ts for _ in range(self.n_gpus)]
                next_updates: list[TensorSet] = [empty_ts for _ in range(self.n_gpus)]

                start = time.perf_counter()
                plan = self._plan(
                    all_needs,
                    all_pos_root,
                    next_needs,
                    next_updates,
                    metas,
                    raw_need_counts=raw_need_counts,
                )
                if self._timing_enabled:
                    self._plan_s_curve.append(time.perf_counter() - start)
                self._put_plan(plan)
                self.pending = None
                continue

            if self.pending is None:
                all_needs, all_pos_root, raw_need_counts, metas = in_q_data
                self.pending = in_q_data
                self.fetch = all_needs  # first batch: fetch everything from CPU
            else:
                all_needs, all_pos_root, raw_need_counts, metas = self.pending
                start = time.perf_counter()
                plan = self._plan(
                    all_needs,
                    all_pos_root,
                    in_q_data[0],
                    in_q_data[1],
                    metas,
                    raw_need_counts=raw_need_counts,
                )
                if self._timing_enabled:
                    self._plan_s_curve.append(time.perf_counter() - start)
                self._put_plan(plan)
                self.pending = in_q_data
        self._emit_timing()

    def _plan(
        self,
        all_list: list[TensorSet],
        update_list: list[TensorSet],
        next_all_list: list[TensorSet],
        next_update_list: list[TensorSet],
        metas: MetaBundle,
        raw_need_counts: list[int] | None = None,
    ) -> list[DynamicPlan]:
        """Build per-GPU DynamicPlan for the current batch.

        Args:
            all_list:        per-GPU full fetch set for the current batch.
            update_list:     per-GPU nodes updated by the current batch.
            next_all_list:   per-GPU full fetch set for the next batch.
            next_update_list: per-GPU nodes that will be updated by the next batch.
            metas:           batch metadata (for reference).
        """
        n_gpus = self.n_gpus
        if raw_need_counts is None:
            raw_need_counts = [len(ts) for ts in all_list]

        # 1. Deduplicate: each updated node is owned by exactly one GPU.
        deduplicate_update_list, all_update, fresh_p2p = (
            _deduplicate_and_route_updates(update_list, next_all_list)
        )

        # 2. Union of next batch's updates — nodes that will be refreshed anyway.
        next_all_update = TensorSet.empty_like(all_update.data)
        for update in next_update_list:
            next_all_update = next_all_update | update

        # 3. Writeback policy:
        #    - Shared-memory direct-fetch mode must persist every owned update now,
        #      because later readers bypass P2P and re-read remote state from CPU.
        #    - The one-step deferred writeback optimization is only sound when stale
        #      P2P is enabled: if the next batch consumes a fresh delta but does not
        #      update that node again, a later batch may need to fetch the node from
        #      CPU. Without stale-P2P coverage the CPU copy would remain stale.
        if self.persist_all_updates_to_cpu or not self.if_carry_over_route:
            writeback_list = deduplicate_update_list
        else:
            writeback_list = [
                per_gpu_update - next_all_update
                for per_gpu_update in deduplicate_update_list
            ]

        # 4. Prefetch for next batch and optional stale P2P.
        if self.if_carry_over_route:
            # Nodes next batch needs that fresh updates can't cover → try stale GPU mem.
            stale_needs = [next_need - all_update for next_need in next_all_list]
            # Stale avails: what this GPU fetched minus its own fresh updates.
            stale_avails = [avail - all_update for avail in all_list]
            remaining_needs, stale_p2p = handle_remaining(
                stale_needs, stale_avails, self.gpus_distance
            )
            # After stale P2P, only truly unavailable nodes need CPU fetch.
            next_prefetch = remaining_needs

            # Merge fresh + stale P2P (disjoint by construction).
            ref = all_update
            empty_ts = TensorSet(
                torch.tensor([], dtype=ref.data.dtype, device=ref.data.device)
            )
            p2p: list[list[TensorSet]] = [[empty_ts] * n_gpus for _ in range(n_gpus)]
            for src in range(n_gpus):
                for dst in range(n_gpus):
                    p2p[src][dst] = fresh_p2p[src][dst] | stale_p2p[src][dst]
        else:
            # No stale handling: unfulfilled needs go to CPU fetch.
            next_prefetch = [next_need - all_update for next_need in next_all_list]
            p2p = fresh_p2p

        # 5. Build per-GPU count matrix and plans.
        p2p_cts = np.zeros((n_gpus, n_gpus), dtype=np.int32)
        for src in range(n_gpus):
            for dst in range(n_gpus):
                p2p_cts[src, dst] = len(p2p[src][dst])

        plans = []
        for src in range(n_gpus):
            plans.append(
                DynamicPlan(
                    p2p=p2p[src],
                    fetch=self.fetch[src],
                    writeback=writeback_list[src],
                    recv_counts=p2p_cts[:, src],
                    send_counts=p2p_cts[src, :],
                )
            )

        self.fetch = next_prefetch
        return plans


class NeighborhoodDynamicPlanner(DynamicPlanner):
    def _plan(
        self,
        box_fetch_list: list[TensorSet],
        box_update_list: list[TensorSet],
        next_box_fetch_list: list[TensorSet],
        next_box_update_list: list[TensorSet],
        metas: MetaBundle,
        raw_need_counts: list[int] | None = None,
    ) -> list[DynamicPlan]:
        n_gpus = self.n_gpus
        if raw_need_counts is None:
            raw_need_counts = [len(ts) for ts in box_fetch_list]

        # 1. Deduplicate: each updated node is owned by exactly one GPU.
        # box_update_list = neighborhood-mail recipients for the current batch
        deduplicate_update_list, all_update, p2p = (
            _deduplicate_and_route_updates(box_update_list, next_box_fetch_list)
        )

        # 2. Writeback all updated nodes to keep ring buffer state consistent on CPU.
        writeback_list = deduplicate_update_list

        # 3. Build per-GPU count matrix and plans.
        p2p_cts = np.zeros((n_gpus, n_gpus), dtype=np.int32)
        for src in range(n_gpus):
            for dst in range(n_gpus):
                p2p_cts[src, dst] = len(p2p[src][dst])

        plans = []
        for src in range(n_gpus):
            plans.append(
                DynamicPlan(
                    p2p=p2p[src],
                    fetch=box_fetch_list[src],
                    writeback=writeback_list[src],
                    recv_counts=p2p_cts[:, src],
                    send_counts=p2p_cts[src, :],
                )
            )

        return plans
