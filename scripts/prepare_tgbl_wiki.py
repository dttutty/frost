#!/usr/bin/env python3
"""Optionally export official tgbl-wiki-v2 into FROST's artifact layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frost.data.paths import resolve_tgb_raw_dataset_dir
from frost.data.tgb import (
    build_tgbl_wiki_node_features,
    load_tgbl_wiki_edge_features,
    load_tgbl_wiki_runtime_graph,
)


def prepare(raw_dir: Path, output_dir: Path) -> None:
    val_ns = raw_dir / "tgbl-wiki_val_ns_v2.pkl"
    test_ns = raw_dir / "tgbl-wiki_test_ns_v2.pkl"
    required = [
        raw_dir / "ml_tgbl-wiki.pkl",
        raw_dir / "ml_tgbl-wiki_edge.pkl",
        val_ns,
        test_ns,
    ]
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

    runtime_graph = load_tgbl_wiki_runtime_graph(raw_dir)
    edges = runtime_graph.frame
    graph = runtime_graph.arrays
    n_nodes = runtime_graph.num_nodes
    raw_edge_features = load_tgbl_wiki_edge_features(raw_dir)
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
    node_features = build_tgbl_wiki_node_features(edges, n_nodes)

    edges.to_csv(output_dir / "edges.csv", index=False)
    np.savez(output_dir / "full_graph_with_reverse_edges.npz", **graph)
    np.save(output_dir / "edge_features.npy", edge_features)
    np.save(output_dir / "node_features.npy", node_features)

    for source in (val_ns, test_ns):
        (output_dir / source.name).symlink_to(source.resolve())

    split_counts = np.bincount(
        edges["default_split"].to_numpy(dtype=np.int64, copy=False), minlength=3
    )
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
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir or resolve_tgb_raw_dataset_dir("TGBL_WIKI")
    prepare(raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
