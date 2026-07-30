from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from omegaconf import DictConfig


logger = logging.getLogger(__name__)

DYGLIB_ALIGNED_FEATURE_DIM = 172
DEFAULT_NO_MEMORY_QUEUE_SIZE = 4
REPO_ROOT = Path(__file__).resolve().parents[3]
DYGLIB_PROCESSED_DATA_ROOT = REPO_ROOT / "third_party" / "DyGLib" / "processed_data"


def normalize_fanout(raw_fanout: object) -> list[int]:
    if raw_fanout is None:
        return []
    if isinstance(raw_fanout, int):
        return [raw_fanout]
    if isinstance(raw_fanout, Iterable) and not isinstance(raw_fanout, str | bytes):
        return [int(n_ngh) for n_ngh in raw_fanout]
    raise TypeError(f"Unsupported fanout config type: {type(raw_fanout)!r}")


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


def resolve_dummy_feature_dim(
    dataset: str,
    requested_dim: object,
    *,
    model_name: str,
    feature_name: str,
    feature_file_name: str,
    processed_data_root: Path | None = None,
) -> int:
    if requested_dim is not None:
        return _coerce_int(requested_dim, name=f"{model_name} dummy_{feature_name}_dim")

    dataset_key = dataset.strip().lower()
    data_root = processed_data_root or DYGLIB_PROCESSED_DATA_ROOT
    feature_path = (
        data_root / dataset_key / feature_file_name.format(dataset_key=dataset_key)
    )
    raw_feature_dim: int | None = None
    if feature_path.exists():
        try:
            raw_features = np.load(feature_path, mmap_mode="r")
            if raw_features.ndim == 2:
                raw_feature_dim = int(raw_features.shape[1])
        except Exception as exc:
            logger.warning(
                "Failed to inspect DyGLib %s features at %s: %s",
                feature_name,
                feature_path,
                exc,
            )

    if raw_feature_dim is not None:
        resolved_dim = max(raw_feature_dim, DYGLIB_ALIGNED_FEATURE_DIM)
        logger.info(
            "Resolved %s dummy_%s_dim=%d for dataset=%s from DyGLib %s "
            "features (raw=%d, aligned floor=%d).",
            model_name,
            feature_name,
            resolved_dim,
            dataset,
            feature_name,
            raw_feature_dim,
            DYGLIB_ALIGNED_FEATURE_DIM,
        )
        return resolved_dim

    logger.info(
        "Resolved %s dummy_%s_dim=%d for dataset=%s using DyGLib's default "
        "aligned feature width.",
        model_name,
        feature_name,
        DYGLIB_ALIGNED_FEATURE_DIM,
        dataset,
    )
    return DYGLIB_ALIGNED_FEATURE_DIM


def resolve_dygformer_dummy_edge_dim(
    dataset: str, requested_dim: object, *, processed_data_root: Path | None = None
) -> int:
    return resolve_dummy_feature_dim(
        dataset,
        requested_dim,
        model_name="DyGFormer",
        feature_name="edge",
        feature_file_name="ml_{dataset_key}.npy",
        processed_data_root=processed_data_root,
    )


def resolve_dygformer_dummy_node_dim(
    dataset: str, requested_dim: object, *, processed_data_root: Path | None = None
) -> int:
    return resolve_dummy_feature_dim(
        dataset,
        requested_dim,
        model_name="DyGFormer",
        feature_name="node",
        feature_file_name="ml_{dataset_key}_node.npy",
        processed_data_root=processed_data_root,
    )


def resolve_tgat_dummy_edge_dim(
    dataset: str, requested_dim: object, *, processed_data_root: Path | None = None
) -> int:
    return resolve_dummy_feature_dim(
        dataset,
        requested_dim,
        model_name="TGAT",
        feature_name="edge",
        feature_file_name="ml_{dataset_key}.npy",
        processed_data_root=processed_data_root,
    )


def resolve_tgat_dummy_node_dim(
    dataset: str, requested_dim: object, *, processed_data_root: Path | None = None
) -> int:
    return resolve_dummy_feature_dim(
        dataset,
        requested_dim,
        model_name="TGAT",
        feature_name="node",
        feature_file_name="ml_{dataset_key}_node.npy",
        processed_data_root=processed_data_root,
    )


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


def resolve_dynamic_state_fetch_mode(
    cfg: DictConfig, *, has_mem: bool, fanout: list[int]
) -> bool:
    del has_mem, fanout
    return bool(cfg.get("__debug_use_shm_instead", False))


def resolve_eval_metric(value: object) -> Literal["ap", "mrr"]:
    metric = str(value).strip().lower()
    if metric not in {"ap", "mrr"}:
        raise ValueError(f"Unsupported eval_metric={metric!r}; expected 'ap' or 'mrr'.")
    return metric  # type: ignore[return-value]


def string_keyed_dict(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must resolve to a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must all be strings")
    return {str(key): item for key, item in value.items()}
