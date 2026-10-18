"""Dataset-source facade shared by runtime and benchmark utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from frost.data.paths import (
    is_raw_tgb_dataset,
    resolve_artifact_dataset_dir,
    resolve_tgb_raw_dataset_dir,
)
from frost.data.tgb import load_tgb_frame


def load_runtime_edge_frame(
    dataset: str, *, artifact_data_root: str | Path | None = None
) -> pd.DataFrame:
    """Load or construct the canonical runtime edge frame for a dataset."""
    if is_raw_tgb_dataset(dataset):
        return load_tgb_frame(resolve_tgb_raw_dataset_dir(dataset), dataset)
    dataset_dir = resolve_artifact_dataset_dir(dataset, data_root=artifact_data_root)
    path = dataset_dir / "edges.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find canonical edge table: {path}")
    return pd.read_csv(path)
