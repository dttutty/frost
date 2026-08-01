"""GPU topology discovery shared by launch policy and runtime planning."""

from __future__ import annotations

import logging
import re
import subprocess
from collections import defaultdict, deque


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
    lines = [
        _ANSI_ESCAPE_RE.sub("", line).rstrip() for line in topo_output.splitlines()
    ]
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

    header_tokens = [tok.strip() for tok in re.split(r"\t+", content[0]) if tok.strip()]
    gpu_headers = []
    for token in header_tokens:
        match = _GPU_LABEL_RE.match(token)
        if match is None:
            break
        gpu_headers.append(int(match.group(1)))
    if not gpu_headers:
        raise ValueError("Failed to parse GPU headers from nvidia-smi topo output")
    try:
        numa_column = header_tokens.index("NUMA Affinity")
    except ValueError:
        numa_column = -1

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
        numa_token = (
            tokens[1 + numa_column]
            if numa_column >= 0 and len(tokens) > 1 + numa_column
            else None
        )
        numa_affinity[src] = (
            int(numa_token)
            if numa_token is not None and re.fullmatch(r"-?\d+", numa_token)
            else None
        )
    return links, numa_affinity


def detect_visible_gpu_numa_nodes(
    n_gpu: int, cuda_visible_devices: str | None = None
) -> list[int | None]:
    """Return physical NUMA nodes for logical visible GPUs."""
    if n_gpu <= 0:
        return []
    topo_output = _run_nvidia_smi(["topo", "-m"])
    if topo_output is None:
        return [None] * n_gpu
    try:
        _, numa_affinity = _parse_nvidia_smi_topology(topo_output)
    except ValueError:
        return [None] * n_gpu
    visible_physical = _resolve_visible_gpu_ids(n_gpu, cuda_visible_devices)
    return [numa_affinity.get(physical) for physical in visible_physical]


def detect_gpu_topology(
    n_gpu: int, cuda_visible_devices: str | None = None
) -> list[int | list]:
    """Detect visible GPU groups ordered by NVLink and NUMA locality."""
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

    logical_numa = {
        logical: numa_affinity.get(physical)
        for logical, physical in enumerate(visible_physical)
    }
    if not {numa for numa in logical_numa.values() if numa is not None}:
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
