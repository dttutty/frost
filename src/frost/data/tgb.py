"""Runtime adapters for official TGB datasets."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from frost.data.preparation import build_temporal_csr


@dataclass(frozen=True)
class TGBRuntimeGraph:
    frame: pd.DataFrame
    arrays: dict[str, np.ndarray]
    num_nodes: int


_DATASET_BASE_NAMES = {
    "TGBL_WIKI": "tgbl-wiki",
    "TGBL_COIN": "tgbl-coin",
    "TGBL_COMMENT": "tgbl-comment",
    "TGBL_FLIGHT": "tgbl-flight",
    "TGBL_REVIEW": "tgbl-review",
}


def _dataset_base_name(dataset: str) -> str:
    normalized = dataset.strip().upper()
    try:
        return _DATASET_BASE_NAMES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw TGB dataset: {dataset!r}") from exc


def _processed_frame_path(raw_dir: Path, dataset: str) -> Path:
    return raw_dir / f"ml_{_dataset_base_name(dataset)}.pkl"


def _tgbl_wiki_edgelist_path(raw_dir: Path) -> Path:
    candidates = (
        raw_dir / "tgbl-wiki_edgelist_v2.csv",
        raw_dir / "tgbl-wiki_edgelist.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing official TGB wiki edge list; checked {[str(path) for path in candidates]}"
    )


def load_tgbl_wiki_frame(raw_dir: Path) -> pd.DataFrame:
    """Build FROST's canonical edge frame from official tgbl-wiki-v2."""
    source = raw_dir / "ml_tgbl-wiki.pkl"
    raw_frame: pd.DataFrame | None = None
    if source.is_file():
        try:
            raw_frame = pd.read_pickle(source)
        except (ModuleNotFoundError, NotImplementedError):
            # Pickles produced by newer NumPy/Pandas releases are not always
            # readable in baseline environments. The official CSV is the
            # portable source of truth, so fall back to it when available.
            if not any(
                path.is_file()
                for path in (
                    raw_dir / "tgbl-wiki_edgelist_v2.csv",
                    raw_dir / "tgbl-wiki_edgelist.csv",
                )
            ):
                raise
    if raw_frame is not None:
        required_columns = {"u", "i", "ts", "idx"}
        missing_columns = required_columns.difference(raw_frame.columns)
        if missing_columns:
            raise ValueError(f"{source} is missing columns {sorted(missing_columns)}.")
        src = raw_frame["u"].to_numpy(dtype=np.int64, copy=True)
        dst = raw_frame["i"].to_numpy(dtype=np.int64, copy=True)
        ts = raw_frame["ts"].to_numpy(dtype=np.int64, copy=True)
    else:
        edge_list = _tgbl_wiki_edgelist_path(raw_dir)
        raw_edges = pd.read_csv(
            edge_list,
            skiprows=1,
            header=None,
            usecols=[0, 1, 2],
        )
        if raw_edges.empty:
            raise ValueError(f"Official TGB edge table is empty: {edge_list}")
        src = raw_edges.iloc[:, 0].to_numpy(dtype=np.int64, copy=True)
        raw_dst = raw_edges.iloc[:, 1].to_numpy(dtype=np.int64, copy=True)
        dst = raw_dst + int(src.max()) + 1
        ts = raw_edges.iloc[:, 2].to_numpy(dtype=np.int64, copy=True)
    if len(src) == 0:
        raise ValueError(f"Official TGB edge table is empty: {source}")

    val_time, test_time = np.quantile(ts, [0.70, 0.85])
    split = np.zeros(len(src), dtype=np.int64)
    split[np.logical_and(ts > val_time, ts <= test_time)] = 1
    split[ts > test_time] = 2
    return pd.DataFrame(
        {
            "eid": np.arange(1, len(src) + 1, dtype=np.int64),
            "src": src,
            "dst": dst,
            "time": ts,
            "default_split": split,
        }
    )


def load_tgb_frame(raw_dir: Path, dataset: str) -> pd.DataFrame:
    """Load the exact event IDs produced by TGB's official preprocessing.

    The large homogeneous datasets use string node identifiers and dataset-specific
    feature transforms. Reusing TGB's processed pickle is both faster and guarantees
    that official negative-candidate dictionaries address the same integer node IDs.
    Wiki retains its portable CSV fallback for freshly extracted archives.
    """
    normalized = dataset.strip().upper()
    if normalized == "TGBL_WIKI":
        return load_tgbl_wiki_frame(raw_dir)

    source = _processed_frame_path(raw_dir, normalized)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing TGB processed edge table {source}. Run the official TGB "
            "dataset preprocessing once before launching FROST."
        )
    raw_frame = pd.read_pickle(source)
    required_columns = {"u", "i", "ts", "w"}
    missing_columns = required_columns.difference(raw_frame.columns)
    if missing_columns:
        raise ValueError(f"{source} is missing columns {sorted(missing_columns)}.")
    src = raw_frame["u"].to_numpy(dtype=np.int64, copy=False)
    dst = raw_frame["i"].to_numpy(dtype=np.int64, copy=False)
    ts = raw_frame["ts"].to_numpy(dtype=np.int64, copy=False)
    val_time, test_time = np.quantile(ts, [0.70, 0.85])
    split = np.zeros(len(src), dtype=np.int8)
    split[np.logical_and(ts > val_time, ts <= test_time)] = 1
    split[ts > test_time] = 2
    return pd.DataFrame(
        {
            "eid": np.arange(1, len(src) + 1, dtype=np.int64),
            "src": src,
            "dst": dst,
            "time": ts,
            "default_split": split,
        }
    )


def load_tgb_runtime_graph(raw_dir: Path, dataset: str) -> TGBRuntimeGraph:
    """Build a bidirectional temporal CSR from official TGB-preprocessed events."""
    frame = load_tgb_frame(raw_dir, dataset)
    src = frame["src"].to_numpy(dtype=np.int64, copy=False)
    dst = frame["dst"].to_numpy(dtype=np.int64, copy=False)
    num_nodes = int(max(src.max(), dst.max())) + 1
    arrays = build_temporal_csr(frame, add_reverse=True, num_nodes=num_nodes)
    return TGBRuntimeGraph(frame=frame, arrays=arrays, num_nodes=num_nodes)


def load_tgbl_wiki_runtime_graph(raw_dir: Path) -> TGBRuntimeGraph:
    """Build FROST's frame and bidirectional CSR from official tgbl-wiki-v2."""
    frame = load_tgbl_wiki_frame(raw_dir)
    src = frame["src"].to_numpy(dtype=np.int64, copy=False)
    dst = frame["dst"].to_numpy(dtype=np.int64, copy=False)
    num_nodes = int(max(src.max(), dst.max())) + 1
    arrays = build_temporal_csr(frame, add_reverse=True, num_nodes=num_nodes)
    return TGBRuntimeGraph(frame=frame, arrays=arrays, num_nodes=num_nodes)


def load_tgbl_wiki_edge_features(raw_dir: Path) -> np.ndarray:
    """Load official edge features without adding FROST's padding row."""
    source = raw_dir / "ml_tgbl-wiki_edge.pkl"
    if source.is_file():
        with source.open("rb") as handle:
            features = np.asarray(pickle.load(handle))
    else:
        edge_list = _tgbl_wiki_edgelist_path(raw_dir)
        with edge_list.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            first_row = next(reader, None)
        if first_row is None or len(first_row) <= 4:
            raise ValueError(
                f"Official TGB wiki edge list has no features: {edge_list}"
            )
        features = pd.read_csv(
            edge_list,
            skiprows=1,
            header=None,
            usecols=range(4, len(first_row)),
        ).to_numpy()
    if features.ndim != 2:
        raise ValueError(
            f"Expected a 2D edge feature array in {source}, got {features.shape}"
        )
    return features


def load_tgb_features(
    raw_dir: Path, dataset: str, feat_type: str
) -> np.ndarray | None:
    """Load the feature matrix consumed by TGB's official TGN baseline."""
    normalized = dataset.strip().upper()
    if feat_type not in {"node", "edge"}:
        raise ValueError(f"Unsupported TGB feature type: {feat_type!r}")
    if normalized == "TGBL_WIKI":
        return None if feat_type == "node" else load_tgbl_wiki_edge_features(raw_dir)
    if feat_type == "node" and normalized != "TGBL_FLIGHT":
        return None

    base_name = _dataset_base_name(normalized)
    suffix = "node" if feat_type == "node" else "edge"
    source = raw_dir / f"ml_{base_name}_{suffix}.pkl"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing TGB processed {feat_type} features {source}. Run the official "
            "TGB dataset preprocessing once before launching FROST."
        )
    with source.open("rb") as handle:
        features = np.asarray(pickle.load(handle))

    if feat_type == "edge" and normalized in {
        "TGBL_COIN",
        "TGBL_REVIEW",
        "TGBL_COMMENT",
    }:
        frame = pd.read_pickle(_processed_frame_path(raw_dir, normalized))
        weights = frame["w"].to_numpy(dtype=np.float64, copy=False).reshape(-1, 1)
        if normalized in {"TGBL_COIN", "TGBL_REVIEW"}:
            features = weights
        else:
            features = np.concatenate((features, weights), axis=1)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D {feat_type} features in {source}: {features.shape}")
    return features
