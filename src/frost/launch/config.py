from __future__ import annotations

import math
from typing import Any, Literal

from omegaconf import DictConfig

DYGLIB_ALIGNED_FEATURE_DIM = 172
DEFAULT_NO_MEMORY_QUEUE_SIZE = 4


def set_nested_attr(root: object, path: tuple[str, ...], value: Any) -> None:
    target: Any = root
    for attr in path[:-1]:
        target = getattr(target, attr)
    setattr(target, path[-1], value)


def stable_mode_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.get("stable_mode", False))


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc


def _coerce_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(f"{name} must be a float, got {value!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a float, got {value!r}") from exc


def resolve_no_memory_queue_size(configured_size: object | None) -> int:
    if configured_size is None:
        return DEFAULT_NO_MEMORY_QUEUE_SIZE
    resolved = _coerce_int(configured_size, name="no_memory_queue_size")
    if resolved < 1:
        raise ValueError("no_memory_queue_size must be >= 1.")
    return resolved


def resolve_time_scaling_factor(configured_factor: object | None) -> float:
    if configured_factor is None:
        return 0.0
    resolved = _coerce_float(configured_factor, name="time_scaling_factor")
    if not math.isfinite(resolved):
        raise ValueError("time_scaling_factor must be finite.")
    return resolved


def resolve_eval_metric(value: object) -> Literal["ap", "mrr"]:
    metric = str(value).strip().lower()
    if metric not in {"ap", "mrr"}:
        raise ValueError(f"Unsupported eval_metric={metric!r}; expected 'ap' or 'mrr'.")
    return metric  # type: ignore[return-value]
