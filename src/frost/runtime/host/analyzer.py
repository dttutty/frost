"""Planner-input factories for each model family.

Each class provides:
- ``get_mem_plan_inputs()``: compute per-batch memory planner inputs.
- ``create_mail_planner()``: factory that returns the correct DynamicPlanner thread.
- ``get_mail_plan_inputs()``: compute per-batch mailbox planner inputs.
- ``get_feat_plan_inputs()``: compute per-batch node-feature planner inputs.

Legacy Access-field semantics, translated from the removed ``Access`` dataclass:
- ``mem_needed``: For TGN and JODIE, all sampled nodes. For APAN,
  ``src + pos_dst + neg_dst`` only, excluding neighbors.
- ``mem_update``: For TGN, JODIE, and APAN, only ``src + pos_dst``.
- ``box_needed``: For TGN and JODIE, all sampled nodes. For APAN,
  ``src + pos_dst + neg_dst`` plus sampled neighbors when mailbox state must be
  forwarded to peer GPUs.
- ``box_update``: For TGN and JODIE, only ``src + pos_dst``. For APAN,
  ``src + pos_dst`` plus the sampled neighbors of ``src`` and ``pos_dst``.
- ``required_nfeat``: For TGN and JODIE, all sampled nodes. For APAN,
  ``src + pos_dst + neg_dst`` only, excluding neighbors.

The current APAN mailbox planner remains aligned with the runtime mailbox write
path, so its concrete mail inputs are defined by ``APANAnalyzer`` below.
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


class _RootMailboxAnalyzer:
    """Shared planner-input policy for models where only positive roots update state."""

    def __init__(self, n_neg: int):
        self.n_neg = n_neg

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

    def _feat_plan_input(self, layout: SampledLayoutView) -> TensorSet:
        return TensorSet(layout.valid_nid)

    def get_mem_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return _unzip_dynamic_inputs(
            [self._mem_plan_input(layout) for layout in layouts]
        )

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
        output_dir: str | None,
    ) -> ThreadBase:
        return DynamicPlanner(
            gpus_distance=gpu_dist,
            in_q=in_q,
            out_q=out_q,
            stop_ev=stop_ev,
            handle_stale=handle_stale,
            output_dir=output_dir,
            plan_kind="mail",
        )

    def get_mail_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return _unzip_dynamic_inputs(
            [self._mail_plan_input(layout) for layout in layouts]
        )

    def get_feat_plan_inputs(self, layouts: list[SampledLayoutView]) -> list[TensorSet]:
        return [self._feat_plan_input(layout) for layout in layouts]


class APANAnalyzer:
    """Combined analyzer + planner for the APAN model (1-hop, sample_4_neg=False).

    Uses ``APANDynamicPlanner`` for mail planning. The box update set includes
    only the valid 1-hop neighbors of src+dst, matching what
    ``Postman.deliver()`` produces at runtime.  Root nodes (src, dst) are
    **not** written to the mailbox by the postman; their state is updated
    directly through the memory updater.
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg

    def _validate_layout(self, layout: SampledLayoutView) -> None:
        assert layout.fanout is not None, "APAN requires fanout for neighbor sampling"
        assert len(layout.fanout) == 1, "APAN only supports 1-hop sampling"
        assert not layout.sample_4_neg, "APAN expects sample_4_neg=False layouts"

    def _mem_plan_input(self, layout: SampledLayoutView) -> tuple[TensorSet, TensorSet]:
        self._validate_layout(layout)

        pos_root = TensorSet(layout.pos_roots)
        all_root = TensorSet(layout.roots)
        return all_root, pos_root

    def _mail_plan_input(
        self, layout: SampledLayoutView
    ) -> tuple[TensorSet, TensorSet]:
        self._validate_layout(layout)
        pos_root = TensorSet(layout.pos_roots)
        box_update = TensorSet(layout.filter_valid(layout.apan_pos_ngh))
        return pos_root, box_update

    def _feat_plan_input(self, layout: SampledLayoutView) -> TensorSet:
        self._validate_layout(layout)
        return TensorSet(layout.roots)

    def get_mem_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return _unzip_dynamic_inputs(
            [self._mem_plan_input(layout) for layout in layouts]
        )

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
        output_dir: str | None,
    ) -> ThreadBase:
        return APANDynamicPlanner(
            gpus_distance=gpu_dist,
            in_q=in_q,
            out_q=out_q,
            stop_ev=stop_ev,
            handle_stale=handle_stale,
            output_dir=output_dir,
            plan_kind="mail",
        )

    def get_mail_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return _unzip_dynamic_inputs(
            [self._mail_plan_input(layout) for layout in layouts]
        )

    def get_feat_plan_inputs(self, layouts: list[SampledLayoutView]) -> list[TensorSet]:
        return [self._feat_plan_input(layout) for layout in layouts]


class NoMemAnalyzer:
    """Planner for memory-free models (e.g. TGAT, GraphMixer, DyGFormer).

    Only node features need to be fetched; there is no memory or mailbox state.
    Mail planner methods are stubs (never called when has_mem=False).
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg

    def get_mem_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return [], []

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
        output_dir: str | None,
    ) -> None:
        return None

    def get_mail_plan_inputs(
        self, layouts: list[SampledLayoutView]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return [], []

    def get_feat_plan_inputs(self, layouts: list[SampledLayoutView]) -> list[TensorSet]:
        return [TensorSet(layout.valid_nid) for layout in layouts]


class TGNAnalyzer(_RootMailboxAnalyzer):
    """Combined analyzer + planner for TGN.

    Uses ``DynamicPlanner`` for mail planning. All sampled nodes need
    memory/mailbox/nfeat; only src+dst update memory and mailbox.
    """

    pass


class JODIEAnalyzer(_RootMailboxAnalyzer):
    """Combined analyzer + planner for JODIE (0-hop memory model).

    Same planning logic as TGN.
    """

    pass
