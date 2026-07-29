from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from pathlib import Path

from frost.data.paths import (
    is_raw_tgb_dataset,
    resolve_artifact_dataset_dir,
    resolve_child_directory,
)
from frost.data.sources import load_runtime_edge_frame


DEFAULT_SPLIT: tuple[float, ...] = (7.0, 1.5, 1.5)


def _normalize_split(split: Sequence[int | float]) -> tuple[float, ...]:
    if len(split) < 2:
        raise ValueError("split must contain at least a train and one eval partition.")

    normalized: list[float] = []
    for weight in split:
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("split weights must be real numbers.")
        if weight <= 0:
            raise ValueError("split weights must be positive.")
        normalized.append(float(weight))
    return tuple(normalized)


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as fh:
        # Subtract the header row. The dataset edge tables are never empty here.
        return max(sum(1 for _ in fh) - 1, 0)


def _resolve_num_edges(dataset_or_num_edges: str | int, data_dir: Path | None) -> int:
    if isinstance(dataset_or_num_edges, bool):
        raise TypeError("dataset_or_num_edges must be a dataset name or an integer.")

    if isinstance(dataset_or_num_edges, int):
        if dataset_or_num_edges <= 0:
            raise ValueError("num_edges must be positive.")
        return dataset_or_num_edges

    if isinstance(dataset_or_num_edges, str):
        if data_dir is None and is_raw_tgb_dataset(dataset_or_num_edges):
            return len(load_runtime_edge_frame(dataset_or_num_edges))
        dataset_dir = (
            resolve_child_directory(data_dir, dataset_or_num_edges)
            if data_dir is not None
            else resolve_artifact_dataset_dir(dataset_or_num_edges)
        )
        csv_path = dataset_dir / "edges.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Could not find edges.csv under {dataset_dir}.")
        num_edges = _count_csv_rows(csv_path)
        if num_edges <= 0:
            raise ValueError(f"Dataset {dataset_or_num_edges} has no edge rows.")
        return num_edges

    raise TypeError("dataset_or_num_edges must be a dataset name or an integer.")


def get_max_macro_batch_size(
    dataset_or_num_edges: str | int,
    split: Sequence[int | float] = DEFAULT_SPLIT,
    data_dir: str | Path | None = None,
) -> int:
    """Return a conservative upper bound for ``macro_batch_size``.

    The returned value ensures ``macro_batch_size`` does not exceed the
    smallest non-train split implied by ``split``. This matches the current
    host-side balancing constraint, which requires each evaluation split to
    contain at least one full macro-batch.

    Args:
        dataset_or_num_edges: Runtime dataset name or a known ``num_edges``.
        split: Relative train/eval split weights. Default is ``(7, 1.5, 1.5)``.
        data_dir: Dataset root used when ``dataset_or_num_edges`` is a name.

    Returns:
        The maximum allowed macro batch size.
    """
    normalized_split = _normalize_split(split)
    total_weight = sum(normalized_split)
    min_eval_weight = min(normalized_split[1:])
    resolved_data_dir = None if data_dir is None else Path(data_dir)
    num_edges = _resolve_num_edges(dataset_or_num_edges, resolved_data_dir)
    min_eval_edges = (num_edges * min_eval_weight) // total_weight
    return int(min_eval_edges)
