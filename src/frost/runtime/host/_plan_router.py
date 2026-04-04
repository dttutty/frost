import logging
import os
import re
import subprocess
import threading
from collections import defaultdict, deque
from queue import Queue as Q

import numpy as np

from frost.aliases.host import (
    DynamicPlanInQ,
    DynamicPlanOutQ,
    LayoutBundle,
    MetaBundle,
    StaticPlanInQ,
    StaticPlanOutQ,
)
from frost.protocols.analyzer_proto import AnalyzerProto
from frost.runtime._thread_base import join_thread, put_shutdown_sentinel
from frost.runtime.host.p2p_planner._dynamic_planner import DynamicPlanner
from frost.runtime.host.p2p_planner._static_planner import StaticPlanner

logger = logging.getLogger(__name__)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_GPU_LABEL_RE = re.compile(r"^GPU(\d+)$")


def _run_nvidia_smi(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return proc.stdout


def _query_gpu_identity_map() -> dict[str, int]:
    out = _run_nvidia_smi(
        ["--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"]
    )
    if out is None:
        return {}

    mapping: dict[str, int] = {}
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        idx_s, uuid, pci_bus_id = [part.strip() for part in line.split(",", maxsplit=2)]
        idx = int(idx_s)
        mapping[idx_s] = idx
        mapping[uuid] = idx
        mapping[pci_bus_id] = idx
    return mapping


def _resolve_visible_gpu_ids(
    n_gpu: int, cuda_visible_devices: str | None = None
) -> list[int]:
    if n_gpu <= 0:
        return []
    if not cuda_visible_devices:
        return list(range(n_gpu))

    mapping = _query_gpu_identity_map()
    resolved: list[int] = []
    for token in (part.strip() for part in cuda_visible_devices.split(",")):
        if not token:
            continue
        if token in mapping:
            resolved.append(mapping[token])
            continue
        if token.isdigit():
            resolved.append(int(token))
            continue
        logger.warning(
            "Unsupported CUDA_VISIBLE_DEVICES token %r; falling back to logical GPU order.",
            token,
        )
        return list(range(n_gpu))

    if len(resolved) < n_gpu:
        logger.warning(
            "CUDA_VISIBLE_DEVICES resolved to %s, but %d worker GPUs are required; "
            "falling back to logical GPU order.",
            resolved,
            n_gpu,
        )
        return list(range(n_gpu))
    return resolved[:n_gpu]


def _parse_nvidia_smi_topology(
    topo_output: str,
) -> tuple[dict[int, dict[int, str]], dict[int, int | None]]:
    lines = [_ANSI_ESCAPE_RE.sub("", line).rstrip() for line in topo_output.splitlines()]
    content: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Legend:"):
            break
        content.append(stripped)
    if not content:
        raise ValueError("nvidia-smi topo output is empty")

    header_tokens = [
        tok.strip() for tok in re.split(r"\t+", content[0]) if tok.strip()
    ]
    gpu_headers = []
    for token in header_tokens:
        match = _GPU_LABEL_RE.match(token)
        if match is None:
            break
        gpu_headers.append(int(match.group(1)))
    if not gpu_headers:
        raise ValueError("Failed to parse GPU headers from nvidia-smi topo output")

    links: dict[int, dict[int, str]] = {}
    numa_affinity: dict[int, int | None] = {}
    for line in content[1:]:
        tokens = [tok.strip() for tok in re.split(r"\t+", line) if tok.strip()]
        if not tokens:
            continue
        row_match = _GPU_LABEL_RE.match(tokens[0])
        if row_match is None:
            continue
        src = int(row_match.group(1))
        if len(tokens) < 1 + len(gpu_headers):
            raise ValueError(f"Malformed nvidia-smi topo row: {line!r}")
        peer_labels = tokens[1 : 1 + len(gpu_headers)]
        links[src] = {dst: label for dst, label in zip(gpu_headers, peer_labels)}
        trailing = tokens[1 + len(gpu_headers) :]
        numa_token = trailing[1] if len(trailing) >= 2 else None
        numa_affinity[src] = (
            int(numa_token)
            if numa_token is not None and re.fullmatch(r"-?\d+", numa_token)
            else None
        )
    return links, numa_affinity


def detect_gpu_topology(
    n_gpu: int, cuda_visible_devices: str | None = None
) -> list[int | list]:
    """Detect visible-GPU topology as nested logical GPU groups.

    The returned nested list uses logical worker GPU indices ``0..n_gpu-1`` and
    encodes three planner distance levels when possible:
    NVLink-connected peers < same-NUMA non-NVLink peers < cross-NUMA peers.
    """
    if n_gpu <= 0:
        return []

    topo_output = _run_nvidia_smi(["topo", "-m"])
    if topo_output is None:
        logger.warning(
            "nvidia-smi topo -m is unavailable; falling back to a flat GPU topology."
        )
        return list(range(n_gpu))

    try:
        links, numa_affinity = _parse_nvidia_smi_topology(topo_output)
    except ValueError as exc:
        logger.warning("Failed to parse nvidia-smi topo -m output: %s", exc)
        return list(range(n_gpu))

    visible_physical = _resolve_visible_gpu_ids(n_gpu, cuda_visible_devices)
    logical_ids = list(range(n_gpu))

    missing = [pid for pid in visible_physical if pid not in links]
    if missing:
        logger.warning(
            "Visible GPU ids %s are missing from nvidia-smi topo -m output; "
            "falling back to a flat GPU topology.",
            missing,
        )
        return list(range(n_gpu))

    logical_adj: dict[int, set[int]] = {idx: set() for idx in logical_ids}
    for i in logical_ids:
        for j in range(i + 1, n_gpu):
            src = visible_physical[i]
            dst = visible_physical[j]
            link = links[src].get(dst, "")
            if link.upper().startswith("NV"):
                logical_adj[i].add(j)
                logical_adj[j].add(i)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in logical_ids:
        if start in seen:
            continue
        comp: list[int] = []
        queue: deque[int] = deque([start])
        seen.add(start)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in sorted(logical_adj[cur]):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(sorted(comp))

    logical_numa: dict[int, int | None] = {
        logical: numa_affinity.get(physical)
        for logical, physical in enumerate(visible_physical)
    }
    known_numas = {numa for numa in logical_numa.values() if numa is not None}
    if not known_numas:
        logical_numa = {logical: 0 for logical in logical_ids}

    numa_groups: dict[int | None, list[list[int]]] = defaultdict(list)
    for comp in components:
        numa_groups[logical_numa[comp[0]]].append(comp)

    ordered_numas = sorted(
        numa_groups,
        key=lambda numa: min(min(comp) for comp in numa_groups[numa]),
    )
    return [
        [sorted(comp) for comp in sorted(numa_groups[numa], key=lambda comp: comp[0])]
        for numa in ordered_numas
    ]


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

    The analyzer object (from Hydra config) provides both the mail planner
    thread factory and the planner-input extraction logic, so PlanRouter does
    not need model-specific branching.

    Args:
        gpu_topo:     flat or nested list of GPU IDs encoding NVLink topology.
        analyzer:     combined analyzer/planner object with ``create_mail_planner()``
                      plus ``get_*_plan_inputs()`` methods.
        handle_stale: forwarded to DynamicPlanner instances.
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
        handle_stale: bool = False,
        output_dir: str | None = None,
        persist_all_updates_to_cpu: bool = False,
    ) -> None:
        self.gpu_dist = build_gpu_distance_matrix(gpu_topo)
        self.n_gpus = len(self.gpu_dist)
        self.stop_ev = stop_ev
        self.analyzer = analyzer
        self.mem_planner = None
        self.mail_planner = None
        self.nfeat_planner = None

        self.mem_plan_in_q: DynamicPlanInQ | None = None
        if mem_plan_out_q:
            self.mem_plan_in_q = Q()
            self.mem_planner = DynamicPlanner(
                gpus_distance=self.gpu_dist,
                in_q=self.mem_plan_in_q,
                out_q=mem_plan_out_q,
                stop_ev=stop_ev,
                handle_stale=handle_stale,
                output_dir=output_dir,
                plan_kind="mem",
                persist_all_updates_to_cpu=persist_all_updates_to_cpu,
            )

        self.mail_plan_in_q: DynamicPlanInQ | None = None
        if mail_plan_out_q:
            self.mail_plan_in_q = Q()
            self.mail_planner = analyzer.create_mail_planner(
                gpu_dist=self.gpu_dist,
                in_q=self.mail_plan_in_q,
                out_q=mail_plan_out_q,
                stop_ev=stop_ev,
                handle_stale=handle_stale,
                output_dir=output_dir,
                persist_all_updates_to_cpu=persist_all_updates_to_cpu,
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

        if self.mem_plan_in_q is not None:
            mem_needs, mem_updates = self.analyzer.get_mem_plan_inputs(layouts)
            self.mem_plan_in_q.put((mem_needs, mem_updates, metas))

        if self.mail_plan_in_q is not None:
            box_needs, box_updates = self.analyzer.get_mail_plan_inputs(layouts)
            self.mail_plan_in_q.put((box_needs, box_updates, metas))

        if self.nfeat_plan_in_q is not None:
            nfeat_needs = self.analyzer.get_feat_plan_inputs(layouts)
            self.nfeat_plan_in_q.put(nfeat_needs)

    def signal_epoch_end(self) -> None:
        """Forward the end-of-epoch sentinel to all sub-planners."""
        if self.mem_plan_in_q is not None:
            self.mem_plan_in_q.put("end_of_epoch")
        if self.mail_plan_in_q is not None:
            self.mail_plan_in_q.put("end_of_epoch")
        if self.nfeat_plan_in_q is not None:
            self.nfeat_plan_in_q.put("end_of_epoch")

    def shutdown(self, join_timeout_s: float = 5.0) -> None:
        """Send shutdown sentinels and join all sub-planner threads."""
        if self.mem_plan_in_q is not None:
            put_shutdown_sentinel(self.mem_plan_in_q, None)
        if self.mail_plan_in_q is not None:
            put_shutdown_sentinel(self.mail_plan_in_q, None)
        if self.nfeat_plan_in_q is not None:
            put_shutdown_sentinel(self.nfeat_plan_in_q, None)

        join_thread(self.mem_planner, join_timeout_s, logger)
        join_thread(self.mail_planner, join_timeout_s, logger)
        join_thread(self.nfeat_planner, join_timeout_s, logger)
