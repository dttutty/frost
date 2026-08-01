"""Default CPU and NUMA-memory placement for local FROST ranks."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path

from frost.launch.gpu_topology import detect_visible_gpu_numa_nodes


@dataclass(frozen=True)
class AffinityStatus:
    node: int | None
    cpu_count: int
    memory_bound: bool
    reason: str

    def describe(self) -> str:
        if self.node is None:
            return f"NUMA affinity not applied: {self.reason}"
        memory = "CPU+memory" if self.memory_bound else "CPU only"
        return (
            f"NUMA affinity: {memory} bound to node {self.node} "
            f"({self.cpu_count} allowed logical CPUs); {self.reason}"
        )


def parse_cpu_list(value: str) -> set[int]:
    """Parse Linux cpulist syntax such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for raw_part in value.strip().split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" not in part:
            cpus.add(int(part))
            continue
        start_s, end_s = part.split("-", maxsplit=1)
        start, end = int(start_s), int(end_s)
        if start > end:
            raise ValueError(f"Invalid descending CPU range {part!r}.")
        cpus.update(range(start, end + 1))
    if not cpus:
        raise ValueError("CPU list is empty.")
    return cpus


def resolve_numa_node(
    policy: str | int,
    *,
    n_gpu: int,
    cuda_visible_devices: str | None,
) -> tuple[int | None, str]:
    """Resolve ``auto``/``off``/explicit-node launch policy."""
    normalized = str(policy).strip().lower()
    if normalized in {"off", "none", "false"}:
        return None, "disabled by numa_policy"
    if normalized != "auto":
        try:
            node = int(normalized)
        except ValueError as exc:
            raise ValueError(
                "numa_policy must be 'auto', 'off', or a non-negative node ID."
            ) from exc
        if node < 0:
            raise ValueError("Explicit NUMA node ID must be non-negative.")
        return node, "explicit numa_policy"

    nodes = detect_visible_gpu_numa_nodes(n_gpu, cuda_visible_devices)
    known = [node for node in nodes if node is not None and node >= 0]
    if len(known) != n_gpu:
        return None, f"GPU NUMA topology is incomplete: {nodes}"
    unique = set(known)
    if len(unique) != 1:
        return None, f"visible GPUs span NUMA nodes {sorted(unique)}"
    return known[0], "auto-selected common visible-GPU NUMA node"


def _bind_memory_to_node(node: int) -> None:
    library = ctypes.util.find_library("numa") or "libnuma.so.1"
    numa = ctypes.CDLL(library, use_errno=True)
    numa.numa_available.restype = ctypes.c_int
    if numa.numa_available() < 0:
        raise OSError("libnuma reports NUMA support unavailable")

    numa.numa_allocate_nodemask.restype = ctypes.c_void_p
    numa.numa_bitmask_clearall.argtypes = [ctypes.c_void_p]
    numa.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    numa.numa_set_membind.argtypes = [ctypes.c_void_p]
    numa.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    mask = numa.numa_allocate_nodemask()
    if not mask:
        raise OSError("libnuma failed to allocate a node mask")
    try:
        numa.numa_bitmask_clearall(mask)
        numa.numa_bitmask_setbit(mask, node)
        ctypes.set_errno(0)
        numa.numa_set_membind(mask)
        error = ctypes.get_errno()
        if error:
            raise OSError(error, os.strerror(error))
    finally:
        numa.numa_bitmask_free(mask)


def configure_process_affinity(
    policy: str | int,
    *,
    n_gpu: int,
    cuda_visible_devices: str | None = None,
    sysfs_root: Path = Path("/sys/devices/system/node"),
) -> AffinityStatus:
    """Bind this rank before it creates threads or shared-memory allocations."""
    node, reason = resolve_numa_node(
        policy,
        n_gpu=n_gpu,
        cuda_visible_devices=cuda_visible_devices,
    )
    if node is None:
        return AffinityStatus(None, 0, False, reason)
    if not hasattr(os, "sched_setaffinity"):
        return AffinityStatus(None, 0, False, "CPU affinity is unsupported")

    cpulist_path = sysfs_root / f"node{node}" / "cpulist"
    try:
        node_cpus = parse_cpu_list(cpulist_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return AffinityStatus(None, 0, False, f"cannot read {cpulist_path}: {exc}")
    allowed = set(os.sched_getaffinity(0))
    selected = node_cpus & allowed
    if not selected:
        return AffinityStatus(
            None,
            0,
            False,
            f"NUMA node {node} has no CPUs in the process cpuset",
        )
    os.sched_setaffinity(0, selected)
    try:
        _bind_memory_to_node(node)
    except (OSError, AttributeError) as exc:
        return AffinityStatus(node, len(selected), False, f"memory bind unavailable: {exc}")
    return AffinityStatus(node, len(selected), True, reason)
