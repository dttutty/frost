"""Planner-input factories for host-side memory, mailbox, and feature planning.

Each analyzer defines the per-batch inputs for:
- ``get_mem_plan_inputs()``
- ``create_mail_planner()``
- ``get_mail_plan_inputs()``
- ``get_feat_plan_inputs()``

Model-specific fetch/update semantics are documented on the concrete analyzer
classes below.
"""

import logging
import threading

import numpy as np

from frost.aliases.host import DynamicPlanInQ, DynamicPlanOutQ
from frost.datatypes.sampled_layout import SampledLayoutView
from frost.runtime._thread_base import ThreadBase
from frost.runtime.host.p2p_planner._dynamic_planner import (
    APANDynamicPlanner,
    DynamicPlanner,
)
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


def _unzip_dynamic_inputs(
    inputs: list[tuple[TensorSet, TensorSet]],
) -> tuple[list[TensorSet], list[TensorSet]]:
    if not inputs:
        return [], []
    needs, updates = zip(*inputs)
    return list(needs), list(updates)


class StaticAnalyzer:
    """Planner for memory-free models (e.g. TGAT, GraphMixer, DyGFormer).

    Only node features need to be fetched; there is no memory or mailbox state.
    Mail planner methods are stubs (never called when has_mem=False).
    """

    def get_mem_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return [], []

    def get_mem_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]:
        return []

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        if_carry_over_route: bool,
        output_dir: str | None,
        persist_all_updates_to_cpu: bool = False,
    ) -> None:
        return None

    def get_mail_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return [], []

    def get_mail_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]:
        return []

    def _feat_plan_input(self, layout: SampledLayoutView) -> TensorSet:
        return TensorSet(layout.valid_nid)

    def get_feat_plan_inputs(self, layouts: list[SampledLayoutView]) -> list[TensorSet]:
        return [self._feat_plan_input(layout) for layout in layouts]


class MemoryAnalyzer(StaticAnalyzer):
    """Shared planner-input policy for memory models without postman delivery.

    Applies to TGN/JODIE-style models where all sampled valid nodes may be read
    in the current batch, but only positive roots (``src + pos_dst``) mutate
    state.

    Planner-input semantics:
    - memory fetch: all sampled valid nodes
    - memory update: positive roots only
    - mailbox fetch: all sampled valid nodes
    - mailbox update: positive roots only
    - node-feature fetch: all sampled valid nodes

    Uses ``DynamicPlanner`` for mailbox planning.
    """

    def _mem_plan_input(self, layout: SampledLayoutView) -> tuple[TensorSet, TensorSet]:
        pos_root = TensorSet(layout.pos_roots)
        all_nodes_set = TensorSet(layout.valid_nid)
        return all_nodes_set, pos_root

    def _mail_plan_input(
        self, layout: SampledLayoutView
    ) -> tuple[TensorSet, TensorSet]:
        pos_root = TensorSet(layout.pos_roots)
        all_nodes_set = TensorSet(layout.valid_nid)
        return all_nodes_set, pos_root

    def get_mem_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return _unzip_dynamic_inputs(
            [self._mem_plan_input(layout) for layout in layouts]
        )

    def get_mem_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]:
        return [int(layout.valid_nid.numel()) for layout in layouts]

    def get_mail_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return _unzip_dynamic_inputs(
            [self._mail_plan_input(layout) for layout in layouts]
        )

    def get_mail_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]:
        return [int(layout.valid_nid.numel()) for layout in layouts]

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
    ) -> ThreadBase:
        return DynamicPlanner(
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


class APANMemoryAnalyzer(MemoryAnalyzer):
    """Planner-input policy for APAN (1-hop, ``sample_4_neg=False``).

    APAN narrows reads to roots and delegates mailbox writes to the runtime
    ``Postman.deliver()`` path.

    Planner-input semantics:
    - memory fetch: all roots (``src + pos_dst + neg_dst``)
    - memory update: positive roots only
    - mailbox fetch: all roots
    - mailbox update: valid 1-hop neighbors of ``src + pos_dst`` only
    - node-feature fetch: all roots only, excluding sampled neighbors

    The mailbox update set must stay aligned with the runtime write path:
    padding entries in ``layout.apan_pos_ngh`` are dropped, and root nodes are
    never written through the postman mailbox path. Uses
    ``APANDynamicPlanner`` for mailbox planning.
    """

    def _validate_layout(self, layout: SampledLayoutView) -> None:
        assert layout.fanout is not None, "APAN requires fanout for neighbor sampling"
        assert len(layout.fanout) == 1, "APAN only supports 1-hop sampling"
        assert not layout.sample_4_neg, "APAN expects sample_4_neg=False layouts"

    def _mem_plan_input(self, layout: SampledLayoutView) -> tuple[TensorSet, TensorSet]:

        self._validate_layout(layout)
        roots = TensorSet(layout.roots)
        pos_root = TensorSet(layout.pos_roots)
        return roots, pos_root

    def _mail_plan_input(
        self, layout: SampledLayoutView
    ) -> tuple[TensorSet, TensorSet]:
        """Build APAN mailbox planner inputs for one batch.

        For each batch:
        - fetch mailbox state for all roots
        - merge the fetched state with the previous P2P delta at runtime
        - write one mail entry to each valid neighbor of ``src + pos_dst``
        - send the current batch's updates so the next batch can prefetch them

        ``box_update`` is returned explicitly, even though it can be inferred
        from the layout, because the planner interface expects a
        ``(fetch, update)`` pair.
        """
        self._validate_layout(layout)
        roots = TensorSet(layout.roots)
        box_update = TensorSet(layout.filter_valid(layout.apan_pos_ngh))
        return roots, box_update

    def _feat_plan_input(self, layout: SampledLayoutView) -> TensorSet:
        self._validate_layout(layout)
        return TensorSet(layout.roots)

    def get_mem_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]:
        return [int(layout.roots.numel()) for layout in layouts]

    def get_mail_plan_raw_need_counts(
        self, layouts: list[SampledLayoutView]
    ) -> list[int]:
        return [int(layout.roots.numel()) for layout in layouts]

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
    ) -> ThreadBase:
        return APANDynamicPlanner(
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
