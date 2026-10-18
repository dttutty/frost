from __future__ import annotations

import os
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - optional production dependency
    psutil = None  # type: ignore[assignment]


_CLOCK_TICKS_PER_SEC = max(int(os.sysconf("SC_CLK_TCK")), 1)


def logical_cpu_count() -> int:
    count = psutil.cpu_count(logical=True) if psutil is not None else os.cpu_count()
    return max(int(count or 1), 1)


def _get_non_negative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _get_optional_non_negative_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _discover_sibling_rank_pids() -> list[int]:
    """Return all sibling rank PIDs launched under the same torchrun parent.

    We rely on the launcher parent rather than distributed collectives so the
    benchmark can include both worker ranks and the host process without adding
    any extra synchronization requirements to the host runtime.
    """

    pid = os.getpid()
    if psutil is None:  # pragma: no cover - guarded by benchmark flag
        return [pid]
    try:
        proc = psutil.Process(pid)
        parent = proc.parent()
        if parent is None:
            return [pid]
        child_pids = sorted({child.pid for child in parent.children(recursive=False)})
        return child_pids or [pid]
    except (psutil.Error, OSError, ValueError):
        return [pid]


def _sample_total_cpu_time_s(pid_list: list[int]) -> tuple[float, list[int]]:
    """Return summed user+system CPU time over tracked processes."""

    total = 0.0
    live_pids: list[int] = []
    for pid in pid_list:
        if psutil is not None:
            try:
                proc = psutil.Process(pid)
                cpu_times = proc.cpu_times()
                total += float(cpu_times.user) + float(cpu_times.system)
            except psutil.Error:
                continue
        else:
            try:
                text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
                rparen = text.rfind(")")
                if rparen == -1:
                    continue
                tail = text[rparen + 2 :].split()
                utime_ticks = int(tail[11])
                stime_ticks = int(tail[12])
                total += (utime_ticks + stime_ticks) / _CLOCK_TICKS_PER_SEC
            except (OSError, ValueError):
                continue
        live_pids.append(pid)
    return total, live_pids
