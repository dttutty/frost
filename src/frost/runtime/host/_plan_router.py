import json
import logging
import os
import threading
import time
from queue import Queue as Q
from typing import Generic, TypeVar

import numpy as np

from frost.runtime.experimental.dynamic_state import (
    DynamicStatePolicy,
    PlannerMode,
    resolve_dynamic_state_policy,
)
from frost.runtime.host.types import (
    DynamicPlanInQ,
    DynamicPlanOutQ,
    LayoutBundle,
    MetaBundle,
    StaticPlanInQ,
    StaticPlanOutQ,
)
from frost.runtime._thread_base import join_thread, put_shutdown_sentinel
from frost.runtime.host.p2p_planner._dynamic_planner import DynamicPlanner
from frost.runtime.host.p2p_planner._static_planner import StaticPlanner
from frost.runtime.host.protocols import AnalyzerProto

logger = logging.getLogger(__name__)

_PlanPayload = TypeVar("_PlanPayload")


class _FanoutQueue(Generic[_PlanPayload]):
    """Minimal queue sink that publishes one immutable plan to two consumers."""

    def __init__(self, *queues: Q[_PlanPayload]) -> None:
        self.queues = queues

    def put(self, item: _PlanPayload, block: bool = True) -> None:
        for queue in self.queues:
            queue.put(item, block=block)

    def put_nowait(self, item: _PlanPayload) -> None:
        for queue in self.queues:
            put_shutdown_sentinel(queue, item)


def _timing_enabled() -> bool:
    return os.environ.get("FROST_HOST_TIMING_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    """Owns and routes to contract-grouped dynamic/static planner threads.

    Not a thread itself — the main thread calls ``dispatch()`` inline after each
    macro-batch, keeping the routing logic in the hot path while the actual plan
    computation runs in parallel sub-threads.

    The analyzer object (from Hydra config) provides both the mail planner
    thread factory and the planner-input extraction logic, so PlanRouter does
    not need model-specific branching.

    Args:
        gpu_topo:     flat or nested list of GPU IDs encoding NVLink topology.
        analyzer:     combined analyzer/planner object that materializes one
                      macro-batch's semantic access scopes and creates any
                      distinct mailbox planner.
        if_carry_over_route: forwarded to DynamicPlanner instances.
    """

    mem_plan_in_q: DynamicPlanInQ | None
    mail_plan_in_q: DynamicPlanInQ | None
    nfeat_plan_in_q: StaticPlanInQ | None
    mem_planner: threading.Thread | None
    mail_planner: threading.Thread | None
    nfeat_planner: threading.Thread | None

    def __init__(
        self,
        gpu_topo: list,
        analyzer: AnalyzerProto,
        mem_plan_out_q: DynamicPlanOutQ | None,
        mail_plan_out_q: DynamicPlanOutQ | None,
        nfeat_plan_out_q: StaticPlanOutQ | None,
        stop_ev: threading.Event,
        if_carry_over_route: bool = False,
        output_dir: str | None = None,
        persist_all_updates_to_cpu: bool = False,
        dynamic_state_policy: DynamicStatePolicy | None = None,
    ) -> None:
        self.gpu_dist = build_gpu_distance_matrix(gpu_topo)
        self.n_gpus = len(self.gpu_dist)
        self.stop_ev = stop_ev
        self.analyzer = analyzer
        self.dynamic_state_policy = (
            dynamic_state_policy
            if dynamic_state_policy is not None
            else resolve_dynamic_state_policy("baseline")
        )
        self.mem_planner = None
        self.mail_planner = None
        self.nfeat_planner = None
        self._timing_enabled = _timing_enabled()
        self._mem_inputs_s_curve: list[float] = []
        self._mem_q_put_s_curve: list[float] = []
        self._mail_inputs_s_curve: list[float] = []
        self._mail_q_put_s_curve: list[float] = []
        self._nfeat_inputs_s_curve: list[float] = []
        self._nfeat_q_put_s_curve: list[float] = []
        self._access_analysis_s_curve: list[float] = []
        self._shared_mem_mail_planner = False

        self.mem_plan_in_q: DynamicPlanInQ | None = None
        can_share_dynamic_planner = (
            mem_plan_out_q is not None
            and mail_plan_out_q is not None
            and analyzer.can_share_memory_mailbox_planner()
        )
        if can_share_dynamic_planner:
            assert mem_plan_out_q is not None
            assert mail_plan_out_q is not None
            self.mem_plan_in_q = Q()
            self.mail_plan_in_q = self.mem_plan_in_q
            shared_out = _FanoutQueue(mem_plan_out_q, mail_plan_out_q)
            self.mem_planner = DynamicPlanner(
                gpus_distance=self.gpu_dist,
                in_q=self.mem_plan_in_q,
                out_q=shared_out,  # type: ignore[arg-type]
                stop_ev=stop_ev,
                if_carry_over_route=if_carry_over_route,
                output_dir=output_dir,
                plan_kind="mem_mail",
                persist_all_updates_to_cpu=persist_all_updates_to_cpu,
                current_batch_only=(
                    self.dynamic_state_policy.planner is PlannerMode.CURRENT_BATCH
                ),
            )
            self._shared_mem_mail_planner = True
        elif mem_plan_out_q:
            self.mem_plan_in_q = Q()
            self.mem_planner = DynamicPlanner(
                gpus_distance=self.gpu_dist,
                in_q=self.mem_plan_in_q,
                out_q=mem_plan_out_q,
                stop_ev=stop_ev,
                if_carry_over_route=if_carry_over_route,
                output_dir=output_dir,
                plan_kind="mem",
                persist_all_updates_to_cpu=persist_all_updates_to_cpu,
                current_batch_only=(
                    self.dynamic_state_policy.planner is PlannerMode.CURRENT_BATCH
                ),
            )

        if not can_share_dynamic_planner:
            self.mail_plan_in_q = None
        if mail_plan_out_q and not can_share_dynamic_planner:
            self.mail_plan_in_q = Q()
            self.mail_planner = analyzer.create_mail_planner(
                gpu_dist=self.gpu_dist,
                in_q=self.mail_plan_in_q,
                out_q=mail_plan_out_q,
                stop_ev=stop_ev,
                if_carry_over_route=if_carry_over_route,
                output_dir=output_dir,
                persist_all_updates_to_cpu=persist_all_updates_to_cpu,
                current_batch_only=(
                    self.dynamic_state_policy.planner is PlannerMode.CURRENT_BATCH
                ),
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

        if self.mem_planner is not None:
            self.mem_planner.start()
        if self.mail_planner is not None:
            self.mail_planner.start()
        if self.nfeat_planner is not None:
            self.nfeat_planner.start()

    def dispatch(self, metas: MetaBundle, layouts: LayoutBundle) -> None:
        """Route one macro-batch to sub-planners. Called inline from the main thread."""
        assert len(metas) == self.n_gpus

        analysis_start = time.perf_counter()
        inputs = self.analyzer.analyze_batch(
            layouts,
            include_node_features=self.nfeat_plan_in_q is not None,
        )
        analysis_end = time.perf_counter()
        if self._timing_enabled:
            self._access_analysis_s_curve.append(analysis_end - analysis_start)

        if self.mem_plan_in_q is not None:
            start = time.perf_counter()
            if inputs.memory is None:
                raise RuntimeError("memory planner requires memory access inputs")
            inputs_end = time.perf_counter()
            self.mem_plan_in_q.put(
                (
                    inputs.memory.needs,
                    inputs.memory.updates,
                    inputs.memory.raw_need_counts,
                    metas,
                )
            )
            if self._timing_enabled:
                self._mem_inputs_s_curve.append(inputs_end - start)
                self._mem_q_put_s_curve.append(time.perf_counter() - inputs_end)

        if self.mail_plan_in_q is not None and not self._shared_mem_mail_planner:
            start = time.perf_counter()
            if inputs.mailbox is None:
                raise RuntimeError("mailbox planner requires mailbox access inputs")
            inputs_end = time.perf_counter()
            self.mail_plan_in_q.put(
                (
                    inputs.mailbox.needs,
                    inputs.mailbox.updates,
                    inputs.mailbox.raw_need_counts,
                    metas,
                )
            )
            if self._timing_enabled:
                self._mail_inputs_s_curve.append(inputs_end - start)
                self._mail_q_put_s_curve.append(time.perf_counter() - inputs_end)

        if self.nfeat_plan_in_q is not None:
            start = time.perf_counter()
            if inputs.node_features is None:
                raise RuntimeError("node-feature planner requires static access inputs")
            inputs_end = time.perf_counter()
            self.nfeat_plan_in_q.put(inputs.node_features)
            if self._timing_enabled:
                self._nfeat_inputs_s_curve.append(inputs_end - start)
                self._nfeat_q_put_s_curve.append(time.perf_counter() - inputs_end)

    def signal_epoch_end(self) -> None:
        """Forward the end-of-epoch sentinel to all sub-planners."""
        if self.mem_plan_in_q is not None:
            self.mem_plan_in_q.put("end_of_epoch")
        if self.mail_plan_in_q is not None and not self._shared_mem_mail_planner:
            self.mail_plan_in_q.put("end_of_epoch")
        if self.nfeat_plan_in_q is not None:
            self.nfeat_plan_in_q.put("end_of_epoch")

    def shutdown(self, join_timeout_s: float = 5.0) -> None:
        """Send shutdown sentinels and join all sub-planner threads."""
        if self.mem_plan_in_q is not None:
            put_shutdown_sentinel(self.mem_plan_in_q, None)
        if self.mail_plan_in_q is not None and not self._shared_mem_mail_planner:
            put_shutdown_sentinel(self.mail_plan_in_q, None)
        if self.nfeat_plan_in_q is not None:
            put_shutdown_sentinel(self.nfeat_plan_in_q, None)

        join_thread(self.mem_planner, join_timeout_s, logger)
        join_thread(self.mail_planner, join_timeout_s, logger)
        join_thread(self.nfeat_planner, join_timeout_s, logger)
        if self._timing_enabled:
            payload = {
                "system": "frost",
                "mem_inputs_s_curve": self._mem_inputs_s_curve,
                "mem_q_put_s_curve": self._mem_q_put_s_curve,
                "mail_inputs_s_curve": self._mail_inputs_s_curve,
                "mail_q_put_s_curve": self._mail_q_put_s_curve,
                "nfeat_inputs_s_curve": self._nfeat_inputs_s_curve,
                "nfeat_q_put_s_curve": self._nfeat_q_put_s_curve,
                "access_analysis_s_curve": self._access_analysis_s_curve,
                "shared_mem_mail_planner": self._shared_mem_mail_planner,
            }
            print(
                "APPENDIX_PLAN_ROUTER_TIMING_CURVE "
                f"{json.dumps(payload, sort_keys=True)}"
            )
