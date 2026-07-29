"""Build and validate the on-disk dataset artifacts consumed by FROST."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from frost.data.graph_policy import (
    check_graph_direction,
    resolve_graph_artifact,
)


CANONICAL_EDGE_COLUMNS = ("src", "dst", "time", "eid")
LEGACY_EDGE_COLUMNS = ("u", "i", "ts", "idx")
GRAPH_KEYS = ("indptr", "indices", "ts", "eid")


def _integer_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(frame[name], errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"edge column {name!r} must be numeric") from exc

    if not np.isfinite(numeric).all():
        raise ValueError(f"edge column {name!r} contains a non-finite value")
    values = numeric.astype(np.int64)
    if not np.array_equal(numeric, values):
        raise ValueError(f"edge column {name!r} must contain integers")
    return np.ascontiguousarray(values)


def resolve_edge_columns(frame: pd.DataFrame) -> tuple[str, str, str, str]:
    """Return source, destination, timestamp, and edge-id column names."""
    columns = set(frame.columns)
    if set(CANONICAL_EDGE_COLUMNS).issubset(columns):
        return CANONICAL_EDGE_COLUMNS
    if set(LEGACY_EDGE_COLUMNS).issubset(columns):
        return LEGACY_EDGE_COLUMNS
    raise ValueError(
        "edges.csv must contain either canonical columns "
        f"{CANONICAL_EDGE_COLUMNS} or legacy columns {LEGACY_EDGE_COLUMNS}; "
        f"found {list(frame.columns)}"
    )


def load_edges(path: Path) -> pd.DataFrame:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"missing edge table: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"edge table is empty: {path}")
    resolve_edge_columns(frame)
    return frame


def _edge_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_col, dst_col, ts_col, eid_col = resolve_edge_columns(frame)
    src = _integer_column(frame, src_col)
    dst = _integer_column(frame, dst_col)
    ts = _integer_column(frame, ts_col)
    eid = _integer_column(frame, eid_col)
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("node ids must be non-negative")
    if (eid < 0).any():
        raise ValueError("edge ids must be non-negative")
    return src, dst, ts, eid


def build_temporal_csr(
    frame: pd.DataFrame,
    *,
    add_reverse: bool,
    num_nodes: int | None = None,
) -> dict[str, np.ndarray]:
    """Build FROST's deterministically sorted temporal CSR representation."""
    if frame.empty:
        raise ValueError("cannot build a graph from an empty edge table")
    src, dst, ts, eid = _edge_arrays(frame)
    inferred_num_nodes = int(max(src.max(), dst.max())) + 1
    if num_nodes is None:
        num_nodes = inferred_num_nodes
    elif num_nodes < inferred_num_nodes:
        raise ValueError(
            f"num_nodes={num_nodes} is smaller than inferred size {inferred_num_nodes}"
        )

    if add_reverse:
        graph_src = np.concatenate((src, dst))
        graph_dst = np.concatenate((dst, src))
        graph_ts = np.concatenate((ts, ts))
        graph_eid = np.concatenate((eid, eid))
    else:
        graph_src = src.copy()
        graph_dst = dst.copy()
        graph_ts = ts.copy()
        graph_eid = eid.copy()

    # Primary order is source, followed by timestamp, eid, and destination.
    order = np.lexsort((graph_dst, graph_eid, graph_ts, graph_src))
    sorted_src = graph_src[order]
    counts = np.bincount(sorted_src, minlength=num_nodes)
    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return {
        "indptr": indptr,
        "indices": np.ascontiguousarray(graph_dst[order], dtype=np.int64),
        "ts": np.ascontiguousarray(graph_ts[order], dtype=np.int64),
        "eid": np.ascontiguousarray(graph_eid[order], dtype=np.int64),
    }


def _require_writable_target(path: Path, *, force: bool) -> Path:
    path = path.expanduser()
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_temporal_csr(
    edges_path: Path,
    output_path: Path,
    *,
    add_reverse: bool,
    force: bool = False,
) -> dict[str, int | str | bool]:
    frame = load_edges(edges_path)
    graph = build_temporal_csr(frame, add_reverse=add_reverse)
    output_path = _require_writable_target(output_path, force=force)
    if output_path.suffix != ".npz":
        raise ValueError(f"graph output must use the .npz extension: {output_path}")
    np.savez(
        output_path,
        indptr=graph["indptr"],
        indices=graph["indices"],
        ts=graph["ts"],
        eid=graph["eid"],
    )
    return {
        "output": str(output_path.resolve()),
        "num_nodes": len(graph["indptr"]) - 1,
        "num_edges": len(frame),
        "stored_adjacencies": len(graph["indices"]),
        "add_reverse": add_reverse,
    }


def generate_node_roles(
    edges_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> dict[str, int | str]:
    """Generate a boolean destination-role feature for a bipartite dataset."""
    frame = load_edges(edges_path)
    src, dst, _, _ = _edge_arrays(frame)
    src_nodes = np.unique(src)
    dst_nodes = np.unique(dst)
    overlap = np.intersect1d(src_nodes, dst_nodes, assume_unique=True)
    if overlap.size:
        preview = overlap[:5].tolist()
        raise ValueError(
            "node-role generation requires disjoint source/destination partitions; "
            f"overlap starts with {preview}"
        )

    num_nodes = int(max(src.max(), dst.max())) + 1
    roles = np.zeros((num_nodes, 1), dtype=np.bool_)
    roles[dst_nodes, 0] = True
    output_path = _require_writable_target(output_path, force=force)
    if output_path.suffix != ".npy":
        raise ValueError(f"node-role output must use the .npy extension: {output_path}")
    np.save(output_path, roles)
    return {
        "output": str(output_path.resolve()),
        "num_nodes": num_nodes,
        "source_nodes": len(src_nodes),
        "destination_nodes": len(dst_nodes),
    }


def convert_tensor_to_numpy(
    source_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source_path = source_path.expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"missing tensor file: {source_path}")
    output_path = _require_writable_target(output_path, force=force)
    if output_path.suffix != ".npy":
        raise ValueError(f"feature output must use the .npy extension: {output_path}")

    value = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"expected {source_path} to contain one torch.Tensor, got {type(value).__name__}"
        )
    array = value.detach().cpu().numpy()
    np.save(output_path, array)
    return {
        "output": str(output_path.resolve()),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _validate_array_rows(path: Path, expected_rows: int) -> None:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim == 0:
        raise ValueError(f"expected an array with a row dimension: {path}")
    if array.shape[0] != expected_rows:
        raise ValueError(
            f"row mismatch for {path}: expected {expected_rows}, got {array.shape[0]}"
        )


def validate_dataset(
    dataset_dir: Path, *, graph_direction: str = "auto"
) -> dict[str, Any]:
    """Validate a canonical dataset directory against FROST's runtime contract."""
    dataset_dir = dataset_dir.expanduser()
    frame = load_edges(dataset_dir / "edges.csv")
    src_col, dst_col, ts_col, eid_col = resolve_edge_columns(frame)
    if (src_col, dst_col, ts_col, eid_col) != CANONICAL_EDGE_COLUMNS:
        raise ValueError("published datasets must use canonical FROST edge columns")
    required_columns = set(CANONICAL_EDGE_COLUMNS) | {"default_split"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"edges.csv is missing columns: {sorted(missing_columns)}")

    graph_path = resolve_graph_artifact(dataset_dir)
    with np.load(graph_path, allow_pickle=False) as archive:
        missing_keys = set(GRAPH_KEYS).difference(archive.files)
        if missing_keys:
            raise ValueError(
                f"graph artifact is missing arrays: {sorted(missing_keys)}"
            )
        actual_direction = check_graph_direction(
            dataset=dataset_dir.name,
            requested=graph_direction,
            n_edges=len(frame),
            n_adjacencies=len(archive["indices"]),
        )
        expected = build_temporal_csr(
            frame, add_reverse=actual_direction == "bidirectional"
        )
        for key in GRAPH_KEYS:
            actual = archive[key]
            if not np.array_equal(actual, expected[key]):
                raise ValueError(
                    f"graph array {key!r} does not match the canonical edge table"
                )

    num_nodes = len(expected["indptr"]) - 1
    num_edges = len(frame)
    optional_shapes: dict[str, list[int]] = {}
    row_contracts = {
        "edge_features.npy": num_edges + 1,
        "node_features.npy": num_nodes,
        "node_role.npy": num_nodes,
    }
    for filename, expected_rows in row_contracts.items():
        path = dataset_dir / filename
        if path.is_file():
            _validate_array_rows(path, expected_rows)
            optional_shapes[filename] = list(
                np.load(path, mmap_mode="r", allow_pickle=False).shape
            )

    labels_path = dataset_dir / "ban_labels.csv"
    if labels_path.is_file():
        labels = pd.read_csv(labels_path)
        if len(labels) != num_edges:
            raise ValueError(
                f"row mismatch for {labels_path}: expected {num_edges}, got {len(labels)}"
            )
        optional_shapes[labels_path.name] = list(labels.shape)

    return {
        "dataset": dataset_dir.name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "stored_adjacencies": len(expected["indices"]),
        "graph_direction": actual_direction,
        "graph_artifact": graph_path.name,
        "optional_artifacts": optional_shapes,
    }
