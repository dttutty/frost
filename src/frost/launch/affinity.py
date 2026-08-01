"""Default CPU and NUMA-memory placement for local FROST ranks."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping

from frost.launch.gpu_topology import detect_visible_gpu_numa_nodes


@dataclass(frozen=True)
class AffinityStatus:
    node: int | None
    cpu_count: int
    cpu_bound: bool
    memory_bound: bool
    reason: str

    def describe(self) -> str:
        if self.node is None:
            return f"NUMA affinity not applied: {self.reason}"
        if self.cpu_bound and self.memory_bound:
            placement = "CPU+memory"
        elif self.cpu_bound:
            placement = "CPU only"
        elif self.memory_bound:
            placement = "memory only"
        else:
            placement = "neither CPU nor memory"
        return (
            f"NUMA affinity: {placement} bound to node {self.node} "
            f"({self.cpu_count} allowed logical CPUs); {self.reason}"
        )


@dataclass(frozen=True)
class NumaRuntimePlan:
    """Feature- and topology-aware placement for post-dataset allocations."""

    localize_workers: bool
    graph_ring_nodes: tuple[int | None, ...]
    shared_state_node: int | None
    reason: str


@dataclass(frozen=True)
class ThreadingStatus:
    """Resolved CPU thread-pool environment for one local rank."""

    target_threads: int | None
    omp_threads: str | None
    mkl_threads: str | None
    omp_source: str
    mkl_source: str
    reason: str

    def describe(self) -> str:
        target = "inherit" if self.target_threads is None else str(self.target_threads)
        return (
            f"CPU threading: target={target}, "
            f"OMP_NUM_THREADS={self.omp_threads or 'unset'} ({self.omp_source}), "
            f"MKL_NUM_THREADS={self.mkl_threads or 'unset'} ({self.mkl_source}); "
            f"{self.reason}"
        )


THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
AUTO_THREAD_ENV_MARKER = "FROST_AUTO_THREAD_ENV"
DEFAULT_MAX_AUTO_THREADS = 32


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


def mark_auto_thread_environment(
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Mark thread variables that were absent before ``torchrun`` starts.

    ``torch.distributed.run`` otherwise injects ``OMP_NUM_THREADS=1`` before
    the child executes :mod:`run`, which makes that launcher default
    indistinguishable from a user override.  The maintained ``frost-train``
    entry point calls this helper in the parent.  A temporary value suppresses
    torchrun's warning; each child replaces marked variables with its
    role-aware value before importing Torch.
    """
    target = os.environ if environ is None else environ
    managed = tuple(key for key in THREAD_ENV_KEYS if key not in target)
    for key in managed:
        target[key] = "1"
    if managed:
        target[AUTO_THREAD_ENV_MARKER] = ",".join(managed)
    else:
        target.pop(AUTO_THREAD_ENV_MARKER, None)
    return managed


def _marked_auto_thread_variables(
    environ: MutableMapping[str, str],
) -> set[str]:
    raw = environ.get(AUTO_THREAD_ENV_MARKER, "")
    return {
        value.strip()
        for value in raw.split(",")
        if value.strip() in THREAD_ENV_KEYS
    }


def _read_node_cpus(node: int, *, node_sysfs_root: Path) -> set[int] | None:
    try:
        return parse_cpu_list(
            (node_sysfs_root / f"node{node}" / "cpulist").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return None


def count_physical_cores(
    cpus: set[int],
    *,
    cpu_sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> tuple[int, bool]:
    """Count physical cores in a logical CPU set using Linux topology IDs.

    The boolean reports whether every CPU had readable topology.  Falling back
    to logical CPUs keeps launch functional on restricted containers, while
    the status reason makes the less conservative estimate visible.
    """
    if not cpus:
        return 0, True
    physical: set[tuple[int, int]] = set()
    for cpu in cpus:
        topology = cpu_sysfs_root / f"cpu{cpu}" / "topology"
        try:
            package = int(
                (topology / "physical_package_id").read_text(encoding="utf-8")
            )
            core = int((topology / "core_id").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return len(cpus), False
        physical.add((package, core))
    return len(physical), True


def _parse_thread_policy(policy: str | int) -> int | None:
    normalized = str(policy).strip().lower()
    if normalized == "auto":
        return None
    if normalized in {"inherit", "off", "none"}:
        return 0
    try:
        threads = int(normalized)
    except ValueError as exc:
        raise ValueError(
            "cpu_threads must be 'auto', 'inherit', or a positive integer."
        ) from exc
    if threads < 1:
        raise ValueError("cpu_threads must be a positive integer.")
    return threads


def resolve_auto_process_threads(
    *,
    n_gpu: int,
    local_rank: int,
    numa_policy: str | int,
    cuda_visible_devices: str | None,
    max_threads: int = DEFAULT_MAX_AUTO_THREADS,
    node_sysfs_root: Path = Path("/sys/devices/system/node"),
    cpu_sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> tuple[int, str]:
    """Choose a topology-aware thread count for one host or GPU rank."""
    if n_gpu < 1:
        raise ValueError("n_gpu must be positive when resolving CPU threads.")
    if not 0 <= local_rank <= n_gpu:
        raise ValueError(f"local_rank must be in 0..{n_gpu}, got {local_rank}.")
    if max_threads < 1:
        raise ValueError("max_threads must be positive.")

    allowed = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set(range(os.cpu_count() or 1))
    )
    if local_rank == n_gpu:
        physical, exact = count_physical_cores(
            allowed, cpu_sysfs_root=cpu_sysfs_root
        )
        threads = max(1, min(max_threads, physical))
        qualifier = "physical" if exact else "logical fallback"
        return threads, f"host rank uses up to {max_threads} of {physical} {qualifier} cores"

    normalized_numa = str(numa_policy).strip().lower()
    explicit_node: int | None = None
    if normalized_numa not in {"auto", "off", "none", "false"}:
        try:
            explicit_node = int(normalized_numa)
        except ValueError:
            explicit_node = None

    gpu_nodes = detect_visible_gpu_numa_nodes(n_gpu, cuda_visible_devices)
    node = explicit_node
    workers_on_node = n_gpu
    if node is None and len(gpu_nodes) == n_gpu:
        candidate = gpu_nodes[local_rank]
        if candidate is not None and candidate >= 0:
            node = int(candidate)
            workers_on_node = sum(value == candidate for value in gpu_nodes)
    elif node is not None:
        workers_on_node = n_gpu

    candidate_cpus = allowed
    if node is not None:
        node_cpus = _read_node_cpus(node, node_sysfs_root=node_sysfs_root)
        if node_cpus is not None and node_cpus & allowed:
            candidate_cpus = node_cpus & allowed
        else:
            node = None
            workers_on_node = n_gpu

    physical, exact = count_physical_cores(
        candidate_cpus, cpu_sysfs_root=cpu_sysfs_root
    )
    threads = max(1, min(max_threads, physical // max(1, workers_on_node)))
    qualifier = "physical" if exact else "logical fallback"
    location = f"NUMA node {node}" if node is not None else "allowed CPU set"
    return (
        threads,
        f"GPU worker shares {physical} {qualifier} cores on {location} with "
        f"{workers_on_node} worker(s), capped at {max_threads}",
    )


def resolve_common_worker_threads(
    *,
    n_gpu: int,
    numa_policy: str | int = "auto",
    cuda_visible_devices: str | None = None,
    max_threads: int = DEFAULT_MAX_AUTO_THREADS,
    node_sysfs_root: Path = Path("/sys/devices/system/node"),
    cpu_sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> int:
    """Return one conservative automatic value usable by external runtimes."""
    return min(
        resolve_auto_process_threads(
            n_gpu=n_gpu,
            local_rank=rank,
            numa_policy=numa_policy,
            cuda_visible_devices=cuda_visible_devices,
            max_threads=max_threads,
            node_sysfs_root=node_sysfs_root,
            cpu_sysfs_root=cpu_sysfs_root,
        )[0]
        for rank in range(n_gpu)
    )


def configure_process_threads(
    policy: str | int,
    *,
    n_gpu: int,
    local_rank: int,
    numa_policy: str | int,
    cuda_visible_devices: str | None,
    environ: MutableMapping[str, str] | None = None,
    max_threads: int = DEFAULT_MAX_AUTO_THREADS,
    node_sysfs_root: Path = Path("/sys/devices/system/node"),
    cpu_sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> ThreadingStatus:
    """Apply role-aware OMP/MKL defaults without replacing user values."""
    target_env = os.environ if environ is None else environ
    managed = _marked_auto_thread_variables(target_env)
    parsed = _parse_thread_policy(policy)
    if parsed == 0:
        target_threads = None
        reason = "cpu_threads=inherit leaves thread-pool environment unchanged"
    elif parsed is not None:
        target_threads = parsed
        reason = f"explicit cpu_threads={parsed}"
    else:
        target_threads, reason = resolve_auto_process_threads(
            n_gpu=n_gpu,
            local_rank=local_rank,
            numa_policy=numa_policy,
            cuda_visible_devices=cuda_visible_devices,
            max_threads=max_threads,
            node_sysfs_root=node_sysfs_root,
            cpu_sysfs_root=cpu_sysfs_root,
        )

    sources: dict[str, str] = {}
    for key in THREAD_ENV_KEYS:
        if key in target_env and key not in managed:
            sources[key] = "environment"
        elif target_threads is None:
            sources[key] = "inherited" if key in target_env else "unset"
        else:
            target_env[key] = str(target_threads)
            sources[key] = "auto" if parsed is None else "config"
    target_env.pop(AUTO_THREAD_ENV_MARKER, None)
    return ThreadingStatus(
        target_threads=target_threads,
        omp_threads=target_env.get("OMP_NUM_THREADS"),
        mkl_threads=target_env.get("MKL_NUM_THREADS"),
        omp_source=sources["OMP_NUM_THREADS"],
        mkl_source=sources["MKL_NUM_THREADS"],
        reason=reason,
    )


def resolve_numa_node(
    policy: str | int,
    *,
    n_gpu: int,
    cuda_visible_devices: str | None,
    local_rank: int | None = None,
) -> tuple[int | None, str]:
    """Resolve ``auto``/``off``/explicit-node launch policy for one local rank.

    Under ``auto``, GPU workers follow the NUMA node of their own visible GPU,
    including when all visible GPUs share one node.  The host rank deliberately
    keeps the launcher's full CPU cpuset and default first-touch memory policy:
    sampling and planning are CPU-heavy and benefit from physical cores across
    sockets, while worker-side feature gathering and H2D staging must stay local
    to the target GPU.  ``local_rank=None`` retains the topology-only resolution
    used by callers that do not yet know the rank.
    """
    normalized = str(policy).strip().lower()
    if normalized in {"off", "none", "false"}:
        return None, "disabled by numa_policy"
    if normalized != "auto":
        try:
            explicit_node = int(normalized)
        except ValueError as exc:
            raise ValueError(
                "numa_policy must be 'auto', 'off', or a non-negative node ID."
            ) from exc
        if explicit_node < 0:
            raise ValueError("Explicit NUMA node ID must be non-negative.")
        return explicit_node, "explicit numa_policy"

    nodes = detect_visible_gpu_numa_nodes(n_gpu, cuda_visible_devices)
    if local_rank is not None:
        if local_rank == n_gpu:
            return (
                None,
                "auto host rank keeps the full CPU cpuset and first-touch memory policy",
            )
        if local_rank < 0 or local_rank >= n_gpu:
            return None, f"local rank {local_rank} is outside 0..{n_gpu}"
        known = [node for node in nodes if node is not None and node >= 0]
        if len(known) != n_gpu:
            return None, f"GPU NUMA topology is incomplete: {nodes}"
        gpu_node = nodes[local_rank]
        if gpu_node is None or gpu_node < 0:
            return None, f"visible GPU {local_rank} has no known NUMA node"
        return gpu_node, f"auto-selected visible GPU {local_rank} NUMA node"

    known = [node for node in nodes if node is not None and node >= 0]
    if len(known) != n_gpu:
        return None, f"GPU NUMA topology is incomplete: {nodes}"
    unique = set(known)
    if len(unique) != 1:
        return None, f"visible GPUs span NUMA nodes {sorted(unique)}"
    return known[0], "auto-selected common visible-GPU NUMA node"


def resolve_gpu_shm_numa_nodes(
    policy: str | int,
    *,
    n_gpu: int,
    cuda_visible_devices: str | None,
) -> list[int | None]:
    """Return per-GPU region placement for host-created shared memory.

    Only ``auto`` applies per-region placement. ``off`` leaves pages on the
    default first-touch policy, while an explicit node already binds the host
    process and therefore needs no region-level override.
    """
    if str(policy).strip().lower() != "auto":
        return [None] * n_gpu
    nodes = detect_visible_gpu_numa_nodes(n_gpu, cuda_visible_devices)
    if len(nodes) != n_gpu or any(node is None or node < 0 for node in nodes):
        return [None] * n_gpu
    return [int(node) for node in nodes if node is not None]


def resolve_shared_gpu_numa_node(
    policy: str | int,
    *,
    n_gpu: int,
    cuda_visible_devices: str | None,
) -> int | None:
    """Return one NUMA node for storage consumed by every visible GPU.

    A shared feature or dynamic-state table can be bound safely only when all
    visible GPUs have the same local node.  Per-GPU rings use
    :func:`resolve_gpu_shm_numa_nodes` instead.  Explicit policies rely on the
    host process-wide bind so region-level placement is unnecessary.
    """
    if str(policy).strip().lower() != "auto":
        return None
    nodes = detect_visible_gpu_numa_nodes(n_gpu, cuda_visible_devices)
    if len(nodes) != n_gpu or any(node is None or node < 0 for node in nodes):
        return None
    known = {int(node) for node in nodes if node is not None}
    return next(iter(known)) if len(known) == 1 else None


def resolve_numa_runtime_plan(
    policy: str | int,
    *,
    n_gpu: int,
    cuda_visible_devices: str | None,
    has_static_features: bool,
) -> NumaRuntimePlan:
    """Compile allocation placement after dataset feature metadata is known.

    Same-node featureless workloads keep the original broad CPU/first-touch
    behavior because their bottleneck is host sampling rather than static
    feature gather.  Static features make worker CPU and H2D locality critical.
    GPUs spanning multiple NUMA nodes also require per-worker/ring locality even
    without static features.
    """
    normalized = str(policy).strip().lower()
    if normalized != "auto":
        return NumaRuntimePlan(
            localize_workers=False,
            graph_ring_nodes=tuple([None] * n_gpu),
            shared_state_node=None,
            reason=f"runtime placement inherited from numa_policy={policy}",
        )

    nodes = detect_visible_gpu_numa_nodes(n_gpu, cuda_visible_devices)
    if len(nodes) != n_gpu or any(node is None or node < 0 for node in nodes):
        return NumaRuntimePlan(
            localize_workers=False,
            graph_ring_nodes=tuple([None] * n_gpu),
            shared_state_node=None,
            reason=f"GPU NUMA topology is incomplete: {nodes}",
        )

    known_nodes = tuple(int(node) for node in nodes if node is not None)
    unique_nodes = set(known_nodes)
    spans_nodes = len(unique_nodes) > 1
    localize = has_static_features or spans_nodes
    if not localize:
        return NumaRuntimePlan(
            localize_workers=False,
            graph_ring_nodes=tuple([None] * n_gpu),
            shared_state_node=None,
            reason=(
                "same-node featureless workload keeps broad worker CPU and "
                "first-touch storage placement"
            ),
        )

    shared_state_node = (
        next(iter(unique_nodes)) if has_static_features and not spans_nodes else None
    )
    return NumaRuntimePlan(
        localize_workers=True,
        graph_ring_nodes=known_nodes,
        shared_state_node=shared_state_node,
        reason=(
            "static feature gather requires GPU-local workers and storage"
            if has_static_features
            else "visible GPUs span NUMA nodes"
        ),
    )


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
    local_rank: int | None = None,
    cuda_visible_devices: str | None = None,
    sysfs_root: Path = Path("/sys/devices/system/node"),
    apply_auto_worker: bool = True,
) -> AffinityStatus:
    """Bind this rank before it creates threads or shared-memory allocations."""
    node, reason = resolve_numa_node(
        policy,
        n_gpu=n_gpu,
        cuda_visible_devices=cuda_visible_devices,
        local_rank=local_rank,
    )
    if node is None:
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else 0
        )
        return AffinityStatus(None, cpu_count, False, False, reason)
    auto_worker = (
        str(policy).strip().lower() == "auto"
        and local_rank is not None
        and 0 <= local_rank < n_gpu
    )
    if auto_worker and not apply_auto_worker:
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else 0
        )
        return AffinityStatus(
            None,
            cpu_count,
            False,
            False,
            "auto runtime plan keeps the worker on the full CPU cpuset",
        )
    if not hasattr(os, "sched_setaffinity"):
        return AffinityStatus(
            None, 0, False, False, "CPU affinity is unsupported"
        )

    allowed = set(os.sched_getaffinity(0))
    cpulist_path = sysfs_root / f"node{node}" / "cpulist"
    try:
        node_cpus = parse_cpu_list(cpulist_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return AffinityStatus(
            None, 0, False, False, f"cannot read {cpulist_path}: {exc}"
        )
    selected = node_cpus & allowed
    if not selected:
        return AffinityStatus(
            None,
            0,
            False,
            False,
            f"NUMA node {node} has no CPUs in the process cpuset",
        )
    os.sched_setaffinity(0, selected)
    try:
        _bind_memory_to_node(node)
    except (OSError, AttributeError) as exc:
        return AffinityStatus(
            node, len(selected), True, False, f"memory bind unavailable: {exc}"
        )
    return AffinityStatus(node, len(selected), True, True, reason)
