import logging
import threading
from queue import Queue as Q
from typing import cast

import numpy as np

from frost.aliases.host import (
    AccessBundle,
    DynamicPlanInQ,
    DynamicPlanOutQ,
    MetaBundle,
    StaticPlanInQ,
    StaticPlanOutQ,
)
from frost.datatypes.access import Access
from frost.protocols.analyzer_proto import AnalyzerProto
from frost.runtime.host.p2p_planner._dynamic_planner import DynamicPlanner
from frost.runtime.host.p2p_planner._static_planner import StaticPlanner
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


def build_gpu_distance_matrix(gpu_topo: list) -> np.ndarray:
    """Build an NxN inter-GPU distance matrix from a (possibly nested) topology list.

    A flat list like ``[0, 1, 2]`` means all GPUs are equidistant (distance 1).
    A nested list like ``[[0, 1], [2, 3]]`` encodes NVLink groups: GPUs within the
    same inner list are closer (distance 1) than GPUs across groups (distance 2).

    Args:
        gpu_topo: flat or arbitrarily nested list of GPU IDs, e.g. ``[0, 1]``
                  or ``[[0, 1], [2, 3]]``.

    Returns:
        np.ndarray of shape (max_gpu_id + 1, max_gpu_id + 1).
    """

    def _all_ids(node: int | list) -> set[int]:
        if isinstance(node, int):
            return {node}
        return set().union(*(_all_ids(child) for child in node))

    def _max_depth(node: int | list) -> int:
        if isinstance(node, int):
            return 0
        return 1 + max((_max_depth(child) for child in node), default=0)

    all_ids = _all_ids(gpu_topo)
    if not all_ids:
        return np.array([])

    n = max(all_ids) + 1
    dist = np.zeros((n, n), dtype=int)
    max_depth = _max_depth(gpu_topo)

    def _fill(node: int | list, depth: int) -> list[int]:
        if isinstance(node, int):
            return [node]
        edge_dist = max_depth - depth
        child_groups = [_fill(child, depth + 1) for child in node]
        for i in range(len(child_groups)):
            for j in range(i + 1, len(child_groups)):
                for a in child_groups[i]:
                    for b in child_groups[j]:
                        dist[a, b] = edge_dist
                        dist[b, a] = edge_dist
        return [gpu for group in child_groups for gpu in group]

    _fill(gpu_topo, 0)
    return dist


class PlanRouter:
    """Owns and routes to sub-planner threads (DynamicPlanner × 2, StaticPlanner).

    Not a thread itself — the main thread calls ``dispatch()`` inline after each
    macro-batch, keeping the routing logic in the hot path while the actual plan
    computation runs in parallel sub-threads.

    The ``planner`` object (from Hydra config) provides both the mail planner
    thread factory and the field-extraction logic, so PlanRouter doesn't need
    model-specific branching.

    Args:
        gpu_topo:     flat or nested list of GPU IDs encoding NVLink topology.
        planner:      combined analyzer/planner object with ``create_mail_planner()``
                      and ``get_mail_plan_inputs()`` methods.
        handle_stale: forwarded to DynamicPlanner instances.
    """

    mem_plan_in_q: DynamicPlanInQ | None
    mail_plan_in_q: DynamicPlanInQ | None
    nfeat_plan_in_q: StaticPlanInQ | None

    def __init__(
        self,
        gpu_topo: list,
        planner: AnalyzerProto,
        mem_plan_out_q: DynamicPlanOutQ | None,
        mail_plan_out_q: DynamicPlanOutQ | None,
        nfeat_plan_out_q: StaticPlanOutQ | None,
        stop_ev: threading.Event,
        handle_stale: bool = False,
    ) -> None:
        self.gpu_dist = build_gpu_distance_matrix(gpu_topo)
        self.n_gpus = len(self.gpu_dist)
        self.stop_ev = stop_ev
        self.planner = planner

        self.mem_plan_in_q: DynamicPlanInQ | None = None
        if mem_plan_out_q:
            self.mem_plan_in_q = Q()
            self.mem_planner = DynamicPlanner(
                gpus_distance=self.gpu_dist,
                in_q=self.mem_plan_in_q,
                out_q=mem_plan_out_q,
                stop_ev=stop_ev,
                handle_stale=handle_stale,
            )

        self.mail_plan_in_q: DynamicPlanInQ | None = None
        if mail_plan_out_q:
            self.mail_plan_in_q = Q()
            self.mail_planner = planner.create_mail_planner(
                gpu_dist=self.gpu_dist,
                in_q=self.mail_plan_in_q,
                out_q=mail_plan_out_q,
                stop_ev=stop_ev,
                handle_stale=handle_stale,
            )

        self.nfeat_plan_in_q: StaticPlanInQ | None = None
        if nfeat_plan_out_q:
            self.nfeat_plan_in_q = Q()
            self.nfeat_planner = StaticPlanner(
                gpus_distance=self.gpu_dist,
                in_q=self.nfeat_plan_in_q,
                out_q=nfeat_plan_out_q,
                stop_ev=stop_ev,
            )

        if mem_plan_out_q:
            self.mem_planner.start()
        if mail_plan_out_q:
            self.mail_planner.start()
        if nfeat_plan_out_q:
            self.nfeat_planner.start()

    def dispatch(self, metas: MetaBundle, access_list: AccessBundle) -> None:
        """Route one macro-batch to sub-planners. Called inline from the main thread."""
        assert len(metas) == self.n_gpus

        if self.mem_plan_in_q is not None:
            mem_needs = [
                cast(TensorSet, cast(Access, io).required_mem) for io in access_list
            ]
            mem_updates = [
                cast(TensorSet, cast(Access, io).mem_update) for io in access_list
            ]
            self.mem_plan_in_q.put((mem_needs, mem_updates, metas))

        if self.mail_plan_in_q is not None:
            access_objs = [cast(Access, io) for io in access_list]
            box_needs, box_updates = self.planner.get_mail_plan_inputs(access_objs)
            self.mail_plan_in_q.put((box_needs, box_updates, metas))

        if self.nfeat_plan_in_q is not None:
            nfeat_needs = [
                cast(TensorSet, cast(Access, io).required_nfeat) for io in access_list
            ]
            self.nfeat_plan_in_q.put(nfeat_needs)

    def signal_epoch_end(self) -> None:
        """Forward the end-of-epoch sentinel to all sub-planners."""
        if self.mem_plan_in_q is not None:
            self.mem_plan_in_q.put("end_of_epoch")
        if self.mail_plan_in_q is not None:
            self.mail_plan_in_q.put("end_of_epoch")
        if self.nfeat_plan_in_q is not None:
            self.nfeat_plan_in_q.put("end_of_epoch")

    def shutdown(self) -> None:
        """Send shutdown sentinels and join all sub-planner threads."""
        if self.mem_plan_in_q is not None:
            self.mem_plan_in_q.put(None)
        if self.mail_plan_in_q is not None:
            self.mail_plan_in_q.put(None)
        if self.nfeat_plan_in_q is not None:
            self.nfeat_plan_in_q.put(None)

        if self.mem_plan_in_q is not None:
            self.mem_planner.join()
        if self.mail_plan_in_q is not None:
            self.mail_planner.join()
        if self.nfeat_plan_in_q is not None:
            self.nfeat_planner.join()
