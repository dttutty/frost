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
class TGBWikiRuntimeGraph:
    frame: pd.DataFrame
    arrays: dict[str, np.ndarray]
    num_nodes: int


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
    if source.is_file():
        raw_frame = pd.read_pickle(source)
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


def load_tgbl_wiki_runtime_graph(raw_dir: Path) -> TGBWikiRuntimeGraph:
    """Build FROST's frame and bidirectional CSR from official tgbl-wiki-v2."""
    frame = load_tgbl_wiki_frame(raw_dir)
    src = frame["src"].to_numpy(dtype=np.int64, copy=False)
    dst = frame["dst"].to_numpy(dtype=np.int64, copy=False)
    num_nodes = int(max(src.max(), dst.max())) + 1
    arrays = build_temporal_csr(frame, add_reverse=True, num_nodes=num_nodes)
    return TGBWikiRuntimeGraph(frame=frame, arrays=arrays, num_nodes=num_nodes)


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


def build_tgbl_wiki_node_features(frame: pd.DataFrame, num_nodes: int) -> np.ndarray:
    """Build the user/page role feature used by the existing FROST conversion."""
    features = np.zeros((num_nodes, 1), dtype=np.bool_)
    features[np.unique(frame["dst"].to_numpy(dtype=np.int64, copy=False))] = True
    return features
