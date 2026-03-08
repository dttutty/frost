"""Combined access-counting + mail-planner factories for each model family.

Each class provides:
- ``count_access()``: compute per-batch Access (which nodes need mem/box/nfeat).
- ``create_mail_planner()``: factory that returns the correct DynamicPlanner thread.
- ``get_mail_plan_inputs()``: extract (needs, updates) from Access for dispatch.
"""

import logging
import threading
from typing import cast

import numpy as np
import torch

from frost.aliases.host import DynamicPlanInQ, DynamicPlanOutQ
from frost.datatypes.access import Access
from frost.runtime._thread_base import ThreadBase
from frost.runtime.host.p2p_planner._dynamic_planner import (
    APANDynamicPlanner,
    DynamicPlanner,
)
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


class APANAnalyzer:
    """Combined analyzer + planner for the APAN model (1-hop, sample_4_neg=False).

    Uses ``APANDynamicPlanner`` for mail planning. The box update set includes
    root nodes (src+dst, self-delivery) plus their valid 1-hop neighbors,
    matching what ``Postman.deliver()`` produces at runtime.
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ) -> Access:
        assert valid_all is not None, "APAN requires valid_all for access counting"
        assert fanout is not None, "APAN requires fanout for neighbor sampling"
        assert len(fanout) == 1, "APAN only supports 1-hop sampling"
        n_src = src.shape[0]
        ngh_start = n_src * (2 + self.n_neg)
        pos_ngh_raw = valid_all[ngh_start : (ngh_start + fanout[0] * n_src * 2)]
        pos_ngh = pos_ngh_raw[pos_ngh_raw != -1]

        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)
        box_update_ts = pos_root | TensorSet(pos_ngh)

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            all_nodes_in_graph=all_nodes_set,
            pos_seed_need=pos_root,
            box_update=box_update_ts,
            required_nfeat=all_nodes_set,
        )

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
    ) -> ThreadBase:
        return APANDynamicPlanner(
            gpus_distance=gpu_dist,
            in_q=in_q,
            out_q=out_q,
            stop_ev=stop_ev,
            handle_stale=handle_stale,
        )

    def get_mail_plan_inputs(
        self, access_list: list[Access]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        # APAN: needs = pos_seed_need (src+dst need box read for assembly)
        needs = [cast(TensorSet, a.pos_seed_need) for a in access_list]
        updates = [cast(TensorSet, a.box_update) for a in access_list]
        return needs, updates


class NoMemAnalyzer:
    """Planner for memory-free models (e.g. TGAT, GraphMixer, DyGFormer).

    Only node features need to be fetched; there is no memory or mailbox state.
    Mail planner methods are stubs (never called when has_mem=False).
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ) -> Access:
        all_nid = all_nid.clone()
        all_nodes_set = TensorSet(all_nid)
        return Access(required_nfeat=all_nodes_set)

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
    ) -> None:
        return None

    def get_mail_plan_inputs(
        self, access_list: list[Access]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        return [], []


class TGNAnalyzer:
    """Combined analyzer + planner for TGN.

    Uses ``DynamicPlanner`` for mail planning. All sampled nodes need
    memory/mailbox/nfeat; only src+dst update memory and mailbox.
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ) -> Access:
        all_nid = all_nid.clone()
        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            all_nodes_in_graph=all_nodes_set,
            pos_seed_need=pos_root,
            box_update=pos_root,
            required_nfeat=all_nodes_set,
        )

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
    ) -> ThreadBase:
        return DynamicPlanner(
            gpus_distance=gpu_dist,
            in_q=in_q,
            out_q=out_q,
            stop_ev=stop_ev,
            handle_stale=handle_stale,
        )

    def get_mail_plan_inputs(
        self, access_list: list[Access]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        # TGN: needs = all_nodes_in_graph (all nodes need box read)
        needs = [cast(TensorSet, a.all_nodes_in_graph) for a in access_list]
        updates = [cast(TensorSet, a.box_update) for a in access_list]
        return needs, updates


class JODIEAnalyzer:
    """Combined analyzer + planner for JODIE (0-hop memory model).

    Same planning logic as TGN.
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ) -> Access:
        all_nid = all_nid.clone()
        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            all_nodes_in_graph=all_nodes_set,
            pos_seed_need=pos_root,
            box_update=pos_root,
            required_nfeat=all_nodes_set,
        )

    def create_mail_planner(
        self,
        gpu_dist: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool,
    ) -> ThreadBase:
        return DynamicPlanner(
            gpus_distance=gpu_dist,
            in_q=in_q,
            out_q=out_q,
            stop_ev=stop_ev,
            handle_stale=handle_stale,
        )

    def get_mail_plan_inputs(
        self, access_list: list[Access]
    ) -> tuple[list[TensorSet], list[TensorSet]]:
        needs = [cast(TensorSet, a.all_nodes_in_graph) for a in access_list]
        updates = [cast(TensorSet, a.box_update) for a in access_list]
        return needs, updates
