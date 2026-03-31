from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


DEFAULT_SPLIT = (8, 1, 1)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "DATA"


def _normalize_split(split: Sequence[int]) -> tuple[int, ...]:
    if len(split) < 2:
        raise ValueError("split must contain at least a train and one eval partition.")

    normalized: list[int] = []
    for weight in split:
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise TypeError("split weights must be integers.")
        if weight <= 0:
            raise ValueError("split weights must be positive.")
        normalized.append(weight)
    return tuple(normalized)


def _resolve_dataset_dir(dataset: str, data_dir: Path) -> Path:
    direct = data_dir / dataset
    if direct.is_dir():
        return direct

    lower = data_dir / dataset.lower()
    if lower.is_dir():
        return lower

    if data_dir.is_dir():
        for child in data_dir.iterdir():
            if child.is_dir() and child.name.lower() == dataset.lower():
                return child

    available = sorted(child.name for child in data_dir.iterdir() if child.is_dir())
    raise FileNotFoundError(
        f"Could not resolve dataset directory for {dataset} under {data_dir}. "
        f"Available datasets: {available}"
    )


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as fh:
        # Subtract the header row. The dataset edge tables are never empty here.
        return max(sum(1 for _ in fh) - 1, 0)


def _resolve_num_edges(dataset_or_num_edges: str | int, data_dir: Path) -> int:
    if isinstance(dataset_or_num_edges, bool):
        raise TypeError("dataset_or_num_edges must be a dataset name or an integer.")

    if isinstance(dataset_or_num_edges, int):
        if dataset_or_num_edges <= 0:
            raise ValueError("num_edges must be positive.")
        return dataset_or_num_edges

    if isinstance(dataset_or_num_edges, str):
        dataset_dir = _resolve_dataset_dir(dataset_or_num_edges, data_dir)
        csv_path = dataset_dir / "edges.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Could not find edges.csv under {dataset_dir}.")
        num_edges = _count_csv_rows(csv_path)
        if num_edges <= 0:
            raise ValueError(f"Dataset {dataset_or_num_edges} has no edge rows.")
        return num_edges

    raise TypeError("dataset_or_num_edges must be a dataset name or an integer.")


def get_max_batch_size(
    dataset_or_num_edges: str | int,
    split: Sequence[int] = DEFAULT_SPLIT,
    n_gpu: int = 2,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> int:
    """Return a conservative upper bound for per-worker ``batch_size``.

    The returned value ensures ``batch_size * n_gpu`` does not exceed the
    smallest non-train split implied by ``split``. This matches the current
    host-side balancing constraint, which requires each evaluation split to
    contain at least one full multi-GPU macro-batch.

    Args:
        dataset_or_num_edges: Dataset name under ``DATA/`` or a known ``num_edges``.
        split: Relative train/eval split weights. Default is ``(8, 1, 1)``.
        n_gpu: Number of worker GPUs. Default is ``2``.
        data_dir: Dataset root used when ``dataset_or_num_edges`` is a name.

    Returns:
        The maximum allowed per-worker batch size. A return value of ``0``
        means the requested split is too small for the given ``n_gpu``.
    """
    if isinstance(n_gpu, bool) or not isinstance(n_gpu, int):
        raise TypeError("n_gpu must be an integer.")
    if n_gpu <= 0:
        raise ValueError("n_gpu must be positive.")

    normalized_split = _normalize_split(split)
    total_weight = sum(normalized_split)
    min_eval_weight = min(normalized_split[1:])
    num_edges = _resolve_num_edges(dataset_or_num_edges, Path(data_dir))
    min_eval_edges = (num_edges * min_eval_weight) // total_weight
    return min_eval_edges // n_gpu
