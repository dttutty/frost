"""Build and validate the on-disk dataset artifacts consumed by FROST."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from frost.data.graph_policy import (
    check_graph_direction,
    infer_graph_direction,
    resolve_graph_artifact,
)


CANONICAL_EDGE_COLUMNS = ("src", "dst", "time", "eid")
LEGACY_EDGE_COLUMNS = ("u", "i", "ts", "idx")
GRAPH_KEYS = ("indptr", "indices", "ts", "eid")
SplitStrategy = Literal["timestamp", "event_order"]


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


def assign_temporal_splits(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    strategy: SplitStrategy = "timestamp",
) -> np.ndarray:
    """Assign deterministic train/validation/test IDs to ordered events.

    ``timestamp`` keeps every tied timestamp in one split. ``event_order`` uses
    exact row-count cutoffs and therefore treats ``(time, eid)`` as the event
    order; it is provided only for protocols that explicitly define that
    ordering.
    """
    if frame.empty:
        raise ValueError("cannot split an empty edge table")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio and val_ratio must be positive and sum to less than 1"
        )
    _, _, ts_col, _ = resolve_edge_columns(frame)
    times = pd.to_numeric(frame[ts_col], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("timestamps must be finite")
    if np.any(times[1:] < times[:-1]):
        raise ValueError("edges must be sorted by non-decreasing timestamp")

    n_edges = len(frame)
    if n_edges < 3:
        raise ValueError("at least three events are required for three splits")

    if strategy == "event_order":
        train_end = int(np.floor(n_edges * train_ratio))
        val_end = int(np.floor(n_edges * (train_ratio + val_ratio)))
        if not (0 < train_end < val_end < n_edges):
            raise ValueError("split ratios produce an empty split")
        split = np.full(n_edges, 2, dtype=np.int8)
        split[:train_end] = 0
        split[train_end:val_end] = 1
        return split
    if strategy != "timestamp":
        raise ValueError(
            f"unsupported split strategy {strategy!r}; expected timestamp or event_order"
        )

    boundaries = np.flatnonzero(times[1:] > times[:-1]) + 1
    if boundaries.size < 2:
        raise ValueError(
            "timestamp splitting requires at least three distinct timestamps; "
            "use event_order only if (time, eid) is the protocol's explicit total order"
        )
    train_target = n_edges * train_ratio
    validation_target = n_edges * (train_ratio + val_ratio)
    train_candidates = boundaries[:-1]
    train_end = int(
        train_candidates[np.argmin(np.abs(train_candidates - train_target))]
    )
    validation_candidates = boundaries[boundaries > train_end]
    val_end = int(
        validation_candidates[
            np.argmin(np.abs(validation_candidates - validation_target))
        ]
    )
    split = np.full(n_edges, 2, dtype=np.int8)
    split[:train_end] = 0
    split[train_end:val_end] = 1
    return split


def write_temporal_splits(
    edges_path: Path,
    output_path: Path,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    strategy: SplitStrategy = "timestamp",
    force: bool = False,
) -> dict[str, Any]:
    """Write an edge table with a regenerated ``default_split`` column."""
    frame = load_edges(edges_path)
    split = assign_temporal_splits(
        frame,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        strategy=strategy,
    )
    frame["default_split"] = split
    output_path = _require_writable_target(output_path, force=force)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    counts = np.bincount(split, minlength=3)
    return {
        "output": str(output_path.resolve()),
        "strategy": strategy,
        "train": int(counts[0]),
        "validation": int(counts[1]),
        "test": int(counts[2]),
    }


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
    if np.issubdtype(array.dtype, np.number):
        rows_per_chunk = max(1, 1_000_000 // max(int(np.prod(array.shape[1:])), 1))
        for start in range(0, array.shape[0], rows_per_chunk):
            if not np.isfinite(array[start : start + rows_per_chunk]).all():
                raise ValueError(f"feature array contains non-finite values: {path}")


def _validate_edge_table(
    frame: pd.DataFrame, *, allow_tied_split_timestamps: bool
) -> dict[str, Any]:
    src, dst, times, eids = _edge_arrays(frame)
    expected_eids = np.arange(1, len(frame) + 1, dtype=np.int64)
    if not np.array_equal(eids, expected_eids):
        raise ValueError("edges.csv must use contiguous eid values 1..N")
    if np.any(times[1:] < times[:-1]):
        raise ValueError("edges.csv timestamps must be non-decreasing")

    split_numeric = pd.to_numeric(frame["default_split"], errors="raise").to_numpy()
    if not np.isfinite(split_numeric).all():
        raise ValueError("default_split contains non-finite values")
    split = split_numeric.astype(np.int8)
    if not np.array_equal(split_numeric, split) or not np.isin(split, [0, 1, 2]).all():
        raise ValueError("default_split must contain only integer values 0, 1, and 2")
    if np.any(split[1:] < split[:-1]):
        raise ValueError("default_split must be non-decreasing")
    counts = np.bincount(split, minlength=3)
    if np.any(counts == 0):
        raise ValueError("train, validation, and test splits must all be non-empty")

    split_ranges = []
    for split_id in range(3):
        split_times = times[split == split_id]
        split_ranges.append((int(split_times.min()), int(split_times.max())))
    tied_boundaries = [
        split_ranges[0][1] == split_ranges[1][0],
        split_ranges[1][1] == split_ranges[2][0],
    ]
    if any(tied_boundaries) and not allow_tied_split_timestamps:
        raise ValueError(
            "default_split divides one or more tied timestamps; regenerate with "
            "the timestamp split strategy or explicitly validate an event-order protocol"
        )
    return {
        "split_counts": counts.astype(int).tolist(),
        "split_time_ranges": split_ranges,
        "tied_split_boundaries": tied_boundaries,
        "num_src_nodes": int(np.unique(src).size),
        "num_dst_nodes": int(np.unique(dst).size),
    }


def _validate_precomputed_negatives(
    dataset_dir: Path, frame: pd.DataFrame
) -> list[int] | None:
    path = dataset_dir / "neg_pre_sampling_1_result.csv"
    if not path.is_file():
        return None
    negatives = pd.read_csv(path)
    if list(negatives.columns) != ["eid", "neg_dst"]:
        raise ValueError(f"{path} must use canonical columns ['eid', 'neg_dst']")
    if len(negatives) != len(frame):
        raise ValueError(
            f"row mismatch for {path}: expected {len(frame)}, got {len(negatives)}"
        )
    expected_eids = np.arange(1, len(frame) + 1, dtype=np.int64)
    neg_eids = negatives["eid"].to_numpy(dtype=np.int64)
    if not np.array_equal(neg_eids, expected_eids):
        raise ValueError(f"{path} must use contiguous eid values 1..N")
    neg_dst = negatives["neg_dst"].to_numpy(dtype=np.int64)
    positive_dst = frame["dst"].to_numpy(dtype=np.int64)
    valid_dst = np.unique(positive_dst)
    if not np.isin(neg_dst, valid_dst).all():
        examples = neg_dst[~np.isin(neg_dst, valid_dst)][:5].tolist()
        raise ValueError(
            f"{path} contains destinations outside the observed dst partition: {examples}"
        )
    if np.any(neg_dst == positive_dst):
        examples = expected_eids[neg_dst == positive_dst][:5].tolist()
        raise ValueError(f"{path} collides with positive edges at eids {examples}")
    return [len(negatives), 2]


def write_dataset_manifest(
    dataset_dir: Path,
    *,
    upstream_project: str,
    upstream_dataset: str,
    source_url: str,
    event_direction: str,
    split_strategy: SplitStrategy,
    negative_seed: int = 0,
    upstream_license: str = "not_specified_by_upstream_record",
    force: bool = False,
) -> dict[str, Any]:
    """Describe an existing release directory's provenance and protocols."""
    dataset_dir = dataset_dir.expanduser()
    frame = load_edges(dataset_dir / "edges.csv")
    edge_summary = _validate_edge_table(
        frame, allow_tied_split_timestamps=split_strategy == "event_order"
    )
    graph_path = resolve_graph_artifact(dataset_dir)
    with np.load(graph_path, allow_pickle=False) as archive:
        sampling_adjacency = infer_graph_direction(
            n_edges=len(frame), n_adjacencies=len(archive["indices"])
        )

    artifacts: dict[str, Any] = {}
    for filename in ("edge_features.npy",):
        path = dataset_dir / filename
        if path.is_file():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            artifacts[filename] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
    manifest = {
        "schema_version": 1,
        "dataset": dataset_dir.name,
        "upstream": {
            "project": upstream_project,
            "dataset": upstream_dataset,
            "url": source_url,
            "license": upstream_license,
        },
        "edge_schema": ["eid", "src", "dst", "time", "default_split"],
        "split": {
            "strategy": split_strategy,
            "ratios_requested": [0.70, 0.15, 0.15],
            "counts": edge_summary["split_counts"],
            "time_ranges": edge_summary["split_time_ranges"],
            "tied_timestamp_boundaries": edge_summary["tied_split_boundaries"],
        },
        "graph": {
            "event_direction": event_direction,
            "sampling_adjacency": sampling_adjacency,
            "artifact": graph_path.name,
        },
        "negative_sampling": {
            "use": "evaluation_only",
            "strategy": "fixed_random_destination_corruption",
            "candidate_set": "observed_dst_nodes",
            "generator": "python_random_v1",
            "seed": negative_seed,
            "artifact": "neg_pre_sampling_1_result.csv",
        },
        "artifacts": artifacts,
    }
    output_path = _require_writable_target(dataset_dir / "manifest.json", force=force)
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(output_path)
    return manifest


def validate_dataset(
    dataset_dir: Path,
    *,
    graph_direction: str = "auto",
    allow_tied_split_timestamps: bool = False,
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
    edge_summary = _validate_edge_table(
        frame, allow_tied_split_timestamps=allow_tied_split_timestamps
    )

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
    }
    for filename, expected_rows in row_contracts.items():
        path = dataset_dir / filename
        if path.is_file():
            _validate_array_rows(path, expected_rows)
            optional_shapes[filename] = list(
                np.load(path, mmap_mode="r", allow_pickle=False).shape
            )

    negative_shape = _validate_precomputed_negatives(dataset_dir, frame)
    if negative_shape is not None:
        optional_shapes["neg_pre_sampling_1_result.csv"] = negative_shape

    manifest_path = dataset_dir / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dataset") != dataset_dir.name:
            raise ValueError(
                f"manifest dataset must equal directory name {dataset_dir.name!r}"
            )
        if manifest.get("edge_schema") != [
            "eid",
            "src",
            "dst",
            "time",
            "default_split",
        ]:
            raise ValueError("manifest edge_schema does not match edges.csv")
        manifest_split = manifest.get("split", {})
        if manifest_split.get("counts") != edge_summary["split_counts"]:
            raise ValueError("manifest split counts do not match edges.csv")
        manifest_graph = manifest.get("graph", {})
        if manifest_graph.get("artifact") != graph_path.name:
            raise ValueError("manifest graph artifact does not match the dataset")
        if manifest_graph.get("sampling_adjacency") != actual_direction:
            raise ValueError("manifest sampling adjacency does not match the graph")
        if negative_shape is not None:
            negative_policy = manifest.get("negative_sampling", {})
            if negative_policy.get("candidate_set") != "observed_dst_nodes":
                raise ValueError(
                    "manifest negative candidate_set must be observed_dst_nodes"
                )
            if negative_policy.get("use") != "evaluation_only":
                raise ValueError("fixed negative artifact must be evaluation_only")

    return {
        "dataset": dataset_dir.name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "stored_adjacencies": len(expected["indices"]),
        "graph_direction": actual_direction,
        "graph_artifact": graph_path.name,
        "optional_artifacts": optional_shapes,
        "manifest": manifest,
        **edge_summary,
    }
