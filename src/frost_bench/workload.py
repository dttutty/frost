from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from frost.data.paths import is_raw_tgb_dataset, resolve_artifact_dataset_dir
from frost.data.sources import load_runtime_edge_frame


@lru_cache(maxsize=None)
def train_edge_end(dataset: str, *, data_root: Path | None = None) -> int:
    if is_raw_tgb_dataset(dataset):
        frame = load_runtime_edge_frame(dataset)
        split = frame["default_split"].to_numpy(copy=False)
        validation_rows = (split > 0).nonzero()[0]
        if len(validation_rows):
            return int(validation_rows[0])
        raise RuntimeError(f"Could not find validation boundary in dataset {dataset}")

    dataset_dir = resolve_artifact_dataset_dir(dataset, data_root=data_root)
    edges_path = next(
        (
            path
            for path in (dataset_dir / "edges.csv", dataset_dir / "edges.csv.zst")
            if path.is_file()
        ),
        None,
    )
    if edges_path is None:
        raise FileNotFoundError(f"Could not find an edge table under {dataset_dir}")
    columns = pd.read_csv(edges_path, nrows=0).columns
    split_column = "default_split" if "default_split" in columns else "ext_roll"
    offset = 0
    for chunk in pd.read_csv(
        edges_path, usecols=[split_column], chunksize=5_000_000
    ):
        values = chunk[split_column].to_numpy(copy=False)
        boundary = np.flatnonzero(values > 0)
        if len(boundary):
            return offset + int(boundary[0])
        offset += len(chunk)
    raise RuntimeError(f"Could not find validation boundary in {edges_path}")


def steps_per_epoch(dataset: str, batch_size: int) -> int:
    return math.ceil(train_edge_end(dataset) / batch_size)


def intervals_per_epoch(dataset: str, batch_size: int) -> int:
    return max(steps_per_epoch(dataset, batch_size) - 1, 0)


# Backend-specific names keep call sites explicit while sharing the arithmetic.
frost_steps_per_epoch = steps_per_epoch
frost_intervals_per_epoch = intervals_per_epoch


def tgl_steps_per_epoch(dataset: str, per_gpu_batch_size: int, n_gpu: int) -> int:
    return math.ceil(train_edge_end(dataset) / (per_gpu_batch_size * n_gpu))


def compute_frost_epochs(
    dataset: str, macro_batch_size: int, warmup: int, target: int
) -> int:
    intervals = intervals_per_epoch(dataset, macro_batch_size)
    if intervals <= 0:
        raise RuntimeError(
            "Not enough train steps per epoch for FROST measurement: "
            f"{steps_per_epoch(dataset, macro_batch_size)}"
        )
    return max(1, math.ceil((warmup + target) / intervals))


def compute_tgl_max_train_steps(
    dataset: str,
    per_gpu_batch_size: int,
    n_gpu: int,
    warmup: int,
    target: int,
) -> int:
    epoch_steps = tgl_steps_per_epoch(dataset, per_gpu_batch_size, n_gpu)
    required_intervals = warmup + target
    steps = 1
    while True:
        full_epochs, partial_steps = divmod(steps, epoch_steps)
        intervals = full_epochs * max(epoch_steps - 1, 0)
        if partial_steps > 0:
            intervals += max(partial_steps - 1, 0)
        if intervals >= required_intervals:
            return steps + 2
        steps += 1


def compute_tgl_epochs(
    dataset: str,
    per_gpu_batch_size: int,
    n_gpu: int,
    max_train_steps: int,
) -> int:
    epoch_steps = tgl_steps_per_epoch(dataset, per_gpu_batch_size, n_gpu)
    return max(1, math.ceil(max_train_steps / epoch_steps))


def compute_dyglib_max_train_batches(
    dataset: str, batch_size: int, warmup: int, target: int
) -> int:
    epoch_steps = math.ceil(train_edge_end(dataset) / batch_size)
    measured_intervals = max(epoch_steps - warmup - 1, 0)
    if measured_intervals <= 0:
        raise RuntimeError(
            f"Not enough train batches per epoch for DyGLib measurement: {epoch_steps}"
        )
    return max(1, math.ceil(target / measured_intervals)) * epoch_steps + 2
