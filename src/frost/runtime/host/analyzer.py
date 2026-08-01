"""Plan-driven host-side access analysis.

The analyzer consumes semantic node sets from an ``ExecutionPlan``.  It has no
knowledge of concrete model names or flat-layout offsets.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch

from frost.datatypes.sampled_layout import SampledLayoutView
from frost.launch.plan import ExecutionPlan
from frost.models.contracts import (
    MemoryState,
    NeighborhoodMailbox,
    NodeSet,
    RootExpansion,
)
from frost.runtime._thread_base import ThreadBase
from frost.runtime.host.p2p_planner._dynamic_planner import (
    DynamicPlanner,
    NeighborhoodDynamicPlanner,
)
from frost.runtime.host.types import DynamicPlanInQ, DynamicPlanOutQ
from frost.runtime.host.types import AccessPlanInputs, DynamicAccessInputs
from frost.utils.tensor_set import TensorSet

_native: Any
try:
    from frost.seq_sampler import _core as _native
except ImportError:  # pragma: no cover - only possible with a stale editable build
    _native = None


_NATIVE_SCOPE_ORDER = (
    NodeSet.ALL_SAMPLED,
    NodeSet.ROOTS,
    NodeSet.POSITIVE_ROOTS,
    NodeSet.POSITIVE_NEIGHBORS,
)
_NATIVE_SCOPE_BITS = {
    scope: 1 << index for index, scope in enumerate(_NATIVE_SCOPE_ORDER)
}


def _unzip_dynamic_inputs(
    inputs: list[tuple[TensorSet, TensorSet]],
) -> tuple[list[TensorSet], list[TensorSet]]:
    if not inputs:
        return [], []
    needs, updates = zip(*inputs)
    return list(needs), list(updates)


class AccessPlanAnalyzer:
    """Translate declarative access scopes into planner inputs."""

    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan

    @property
    def _state(self) -> MemoryState | None:
        state = self.plan.dynamic_state
        return state if isinstance(state, MemoryState) else None

    @staticmethod
    def _select(layout: SampledLayoutView, node_set: NodeSet) -> TensorSet:
        return TensorSet(layout.select(node_set))

    def analyze_batch(
        self,
        layouts: list[SampledLayoutView],
        *,
        include_node_features: bool,
    ) -> AccessPlanInputs:
        """Materialize every distinct semantic node set exactly once per layout.

        Memory, mailbox, and static-feature contracts frequently select the same
        nodes.  Sharing these immutable ``TensorSet`` instances avoids repeated
        filtering, cloning, sorting, and uniquing in the host hot path.
        """

        state = self._state
        mailbox_write = self._mailbox_write_scope()
        scopes: set[NodeSet] = set()
        if state is not None:
            scopes.update(
                (
                    state.memory_read,
                    state.memory_write,
                    state.mailbox_read,
                    mailbox_write,
                )
            )
        if include_node_features:
            scopes.add(self.plan.node_feature_scope)

        ordered_scopes = sorted(scopes, key=lambda item: item.value)
        materialized: dict[NodeSet, list[TensorSet]] = {
            scope: [] for scope in ordered_scopes
        }
        raw_counts: dict[NodeSet, list[int]] = {
            scope: [] for scope in ordered_scopes
        }
        for layout in layouts:
            per_scope = self._materialize_layout_scopes(layout, ordered_scopes)
            for scope in ordered_scopes:
                tensor_set, raw_count = per_scope[scope]
                materialized[scope].append(tensor_set)
                raw_counts[scope].append(raw_count)

        memory = None
        mailbox = None
        if state is not None:
            memory = DynamicAccessInputs(
                needs=materialized[state.memory_read],
                updates=materialized[state.memory_write],
                raw_need_counts=raw_counts[state.memory_read],
            )
            mailbox = DynamicAccessInputs(
                needs=materialized[state.mailbox_read],
                updates=materialized[mailbox_write],
                raw_need_counts=raw_counts[state.mailbox_read],
            )

        node_features = (
            materialized[self.plan.node_feature_scope]
            if include_node_features
            else None
        )
        return AccessPlanInputs(
            memory=memory,
            mailbox=mailbox,
            node_features=node_features,
        )

    @staticmethod
    def _materialize_layout_scopes(
        layout: SampledLayoutView,
        scopes: list[NodeSet],
    ) -> dict[NodeSet, tuple[TensorSet, int]]:
        """Build all requested unique node sets with one native layout scan."""

        requested = sum(_NATIVE_SCOPE_BITS.get(scope, 0) for scope in scopes)
        native_fn = None
        if _native is not None and layout.nid.device.type == "cpu":
            if layout.nid.dtype == torch.int32:
                native_fn = getattr(_native, "materialize_node_scopes_32", None)
            elif layout.nid.dtype == torch.int64:
                native_fn = getattr(_native, "materialize_node_scopes_64", None)

        if native_fn is not None and layout.nid.is_contiguous():
            positive_neighbor_start = layout.n_root
            positive_neighbor_end = positive_neighbor_start
            if NodeSet.POSITIVE_NEIGHBORS in scopes:
                if layout.root_expansion is not RootExpansion.POSITIVE_ROOTS:
                    raise ValueError(
                        "positive_neighbors requires positive-root neighborhood expansion."
                    )
                fanout = layout.fanout or []
                if not fanout:
                    raise IndexError(
                        "valid_hop is unavailable for 0-hop layouts."
                    )
                positive_neighbor_end += 2 * layout.n_src * fanout[0]

            arrays, counts = native_fn(
                layout.nid.numpy(),
                layout.n_root,
                2 * layout.n_src,
                positive_neighbor_start,
                positive_neighbor_end,
                requested,
            )
            result: dict[NodeSet, tuple[TensorSet, int]] = {}
            for index, scope in enumerate(_NATIVE_SCOPE_ORDER):
                if scope in scopes:
                    result[scope] = (
                        TensorSet(
                            torch.from_numpy(arrays[index]),
                            sort_now=False,
                            ensure_unique=False,
                        ),
                        int(counts[index]),
                    )
            if NodeSet.NONE in scopes:
                result[NodeSet.NONE] = (
                    TensorSet.empty_like(layout.nid),
                    0,
                )
            return result

        result = {}
        for scope in scopes:
            selected = layout.select(scope)
            result[scope] = (TensorSet(selected), int(selected.numel()))
        return result

    def can_share_memory_mailbox_planner(self) -> bool:
        """Whether memory and mailbox have identical dynamic routing semantics."""

        state = self._state
        if state is None or isinstance(state.mailbox, NeighborhoodMailbox):
            return False
        return (
            state.memory_read is state.mailbox_read
            and state.memory_write is self._mailbox_write_scope()
        )

    def get_mem_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        state = self._state
        if state is None:
            return [], []
        return _unzip_dynamic_inputs(
            [
                (
                    self._select(layout, state.memory_read),
                    self._select(layout, state.memory_write),
                )
                for layout in layouts
            ]
        )

    def get_mem_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]:
        state = self._state
        if state is None:
            return []
        return [int(layout.select(state.memory_read).numel()) for layout in layouts]

    def _mailbox_write_scope(self) -> NodeSet:
        state = self._state
        if state is None:
            return NodeSet.NONE
        mailbox = state.mailbox
        return getattr(mailbox, "recipients", state.memory_write)

    def get_mail_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        state = self._state
        if state is None:
            return [], []
        write_scope = self._mailbox_write_scope()
        return _unzip_dynamic_inputs(
            [
                (
                    self._select(layout, state.mailbox_read),
                    self._select(layout, write_scope),
                )
                for layout in layouts
            ]
        )

    def get_mail_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]:
        state = self._state
        if state is None:
            return []
        return [int(layout.select(state.mailbox_read).numel()) for layout in layouts]

    def get_feat_plan_inputs(self, layouts: list[SampledLayoutView]) -> list[TensorSet]:
        scope = self.plan.node_feature_scope
        return [self._select(layout, scope) for layout in layouts]

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        if_carry_over_route: bool,
        output_dir: str | None,
        persist_all_updates_to_cpu: bool = False,
        current_batch_only: bool = False,
    ) -> ThreadBase | None:
        state = self._state
        if state is None:
            return None
        planner_type = (
            NeighborhoodDynamicPlanner
            if isinstance(state.mailbox, NeighborhoodMailbox)
            else DynamicPlanner
        )
        return planner_type(
            gpus_distance=gpu_dist,
            in_q=in_q,
            out_q=out_q,
            stop_ev=stop_ev,
            if_carry_over_route=if_carry_over_route,
            output_dir=output_dir,
            plan_kind="mail",
            persist_all_updates_to_cpu=persist_all_updates_to_cpu,
            current_batch_only=current_batch_only,
        )
