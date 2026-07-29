"""Plan-driven host-side access analysis.

The analyzer consumes semantic node sets from an ``ExecutionPlan``.  It has no
knowledge of concrete model names or flat-layout offsets.
"""

from __future__ import annotations

import threading

import numpy as np

from frost.datatypes.sampled_layout import SampledLayoutView
from frost.launch.plan import ExecutionPlan
from frost.models.contracts import MemoryState, NeighborhoodMailbox, NodeSet
from frost.runtime._thread_base import ThreadBase
from frost.runtime.host.p2p_planner._dynamic_planner import (
    DynamicPlanner,
    NeighborhoodDynamicPlanner,
)
from frost.runtime.host.types import DynamicPlanInQ, DynamicPlanOutQ
from frost.utils.tensor_set import TensorSet


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
        scope = self.plan.features.node_scope
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
