#!/usr/bin/env python3
"""Convert the official TGB tgbl-wiki-v2 files into FROST's DATA layout."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from frost.data import build_temporal_csr


def prepare(raw_dir: Path, output_dir: Path) -> None:
    source_df = raw_dir / "ml_tgbl-wiki.pkl"
    source_edge_feat = raw_dir / "ml_tgbl-wiki_edge.pkl"
    val_ns = raw_dir / "tgbl-wiki_val_ns_v2.pkl"
    test_ns = raw_dir / "tgbl-wiki_test_ns_v2.pkl"
    required = [source_df, source_edge_feat, val_ns, test_ns]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official TGB files: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    managed_outputs = [
        output_dir / "edges.csv",
        output_dir / "full_graph_with_reverse_edges.npz",
        output_dir / "edge_features.npy",
        output_dir / "node_features.npy",
        output_dir / "tgbl-wiki_val_ns_v2.pkl",
        output_dir / "tgbl-wiki_test_ns_v2.pkl",
        output_dir / "tgb_metadata.json",
    ]
    existing = [str(path) for path in managed_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing conversion. Remove these managed "
            f"outputs first if regeneration is intended: {existing}"
        )

    raw_df = pd.read_pickle(source_df)
    required_columns = {"u", "i", "ts", "idx"}
    missing_columns = required_columns.difference(raw_df.columns)
    if missing_columns:
        raise ValueError(f"{source_df} is missing columns {sorted(missing_columns)}.")

    src = raw_df["u"].to_numpy(dtype=np.int64, copy=True)
    dst = raw_df["i"].to_numpy(dtype=np.int64, copy=True)
    ts = raw_df["ts"].to_numpy(dtype=np.int64, copy=True)
    # FROST reserves edge-feature row zero, so interaction EIDs are one-based.
    eid = np.arange(1, len(raw_df) + 1, dtype=np.int64)

    val_time, test_time = np.quantile(ts, [0.70, 0.85])
    split = np.zeros(len(raw_df), dtype=np.int64)
    split[np.logical_and(ts > val_time, ts <= test_time)] = 1
    split[ts > test_time] = 2
    edges = pd.DataFrame(
        {
            "eid": eid,
            "src": src,
            "dst": dst,
            "time": ts,
            "default_split": split,
        }
    )

    n_nodes = int(max(src.max(), dst.max())) + 1
    graph = build_temporal_csr(edges, add_reverse=True, num_nodes=n_nodes)

    with source_edge_feat.open("rb") as f:
        raw_edge_features = np.asarray(pickle.load(f))
    if raw_edge_features.shape[0] != len(edges):
        raise ValueError(
            "Official edge-feature rows do not match the edge list: "
            f"{raw_edge_features.shape[0]} != {len(edges)}"
        )
    edge_features = np.zeros(
        (len(edges) + 1, raw_edge_features.shape[1]),
        dtype=raw_edge_features.dtype,
    )
    edge_features[1:] = raw_edge_features

    # A lightweight bipartite role feature: users=False, pages=True.
    node_features = np.zeros((n_nodes, 1), dtype=np.bool_)
    node_features[np.unique(dst)] = True

    edges.to_csv(output_dir / "edges.csv", index=False)
    np.savez(output_dir / "full_graph_with_reverse_edges.npz", **graph)
    np.save(output_dir / "edge_features.npy", edge_features)
    np.save(output_dir / "node_features.npy", node_features)

    for source in (val_ns, test_ns):
        (output_dir / source.name).symlink_to(source.resolve())

    split_counts = np.bincount(split, minlength=3)
    metadata = {
        "dataset": "tgbl-wiki",
        "version": 2,
        "num_nodes": n_nodes,
        "num_edges": len(edges),
        "edge_feature_dim": int(edge_features.shape[1]),
        "train_edges": int(split_counts[0]),
        "val_edges": int(split_counts[1]),
        "test_edges": int(split_counts[2]),
        "negative_capacity": 999,
        "raw_dir": str(raw_dir.resolve()),
    }
    (output_dir / "tgb_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path.home() / "datasets" / "tgb" / "raw" / "tgbl_wiki",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DATA") / "TGBL_WIKI",
    )
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
