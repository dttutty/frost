"""Reproducible import of the DGB Zenodo snapshot into FROST artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frost.data.graph_policy import graph_filename
from frost.data.preparation import (
    SplitStrategy,
    assign_temporal_splits,
    write_temporal_csr,
)

DGB_ZENODO_RECORD = "7213796"
DGB_ZENODO_URL = f"https://zenodo.org/records/{DGB_ZENODO_RECORD}"
TIMESTAMP_INDEX_DATASETS = {"canparl", "untrade", "unvote"}
EDGE_FEATURE_DTYPES: dict[str, np.dtype[Any]] = {
    "canparl": np.dtype(np.int16),
    "contacts": np.dtype(np.int8),
    "flights": np.dtype(np.int16),
    "untrade": np.dtype(np.int32),
    "unvote": np.dtype(np.int16),
    "uslegis": np.dtype(np.int16),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_single(source_dir: Path, pattern: str, *, required: bool) -> Path | None:
    matches = sorted(source_dir.glob(pattern))
    if len(matches) > 1:
        raise ValueError(f"expected at most one {pattern} in {source_dir}: {matches}")
    if not matches:
        if required:
            raise FileNotFoundError(f"missing {pattern} in {source_dir}")
        return None
    return matches[0]


def _resolve_edge_feature(source_dir: Path) -> Path | None:
    matches = [
        path
        for path in sorted(source_dir.glob("ml_*.npy"))
        if not path.stem.endswith("_node") and not path.stem.endswith("_node_all_zero")
    ]
    if len(matches) > 1:
        raise ValueError(
            f"expected at most one DGB edge feature array in {source_dir}: {matches}"
        )
    return matches[0] if matches else None


def _integerize_timestamps(values: pd.Series) -> tuple[np.ndarray, int]:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("DGB timestamps must be finite")
    for multiplier in (1, 1_000, 1_000_000, 1_000_000_000):
        scaled = numeric * multiplier
        rounded = np.rint(scaled)
        if np.allclose(scaled, rounded, rtol=0, atol=1e-6):
            return rounded.astype(np.int64), multiplier
    raise ValueError(
        "timestamps cannot be represented exactly enough with a decimal scale up to 1e9"
    )


def _normalize_timestamp_indices(times: np.ndarray) -> np.ndarray:
    if np.any(times[1:] < times[:-1]):
        raise ValueError("DGB events must be sorted before timestamp normalization")
    changes = np.empty(len(times), dtype=np.bool_)
    changes[0] = True
    changes[1:] = times[1:] != times[:-1]
    return np.cumsum(changes, dtype=np.int64) - 1


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def import_dgb_snapshot(
    source_dir: Path,
    output_dir: Path,
    *,
    dataset: str,
    graph_direction: str,
    split_strategy: SplitStrategy = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    negative_seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Import one extracted DGB ``ml_*`` directory into a FROST dataset.

    The Zenodo record already contains the DGB baseline's ``ml_*.csv`` and
    ``ml_*.npy`` products. This function makes every additional FROST
    transformation explicit and records it in ``manifest.json``.
    """
    # Import lazily so importing frost.data.paths from the negative-sampler
    # package does not cycle through frost.data.__init__ -> frost.data.dgb.
    from frost.neg_sampler import generate_precomputed_negative_file

    source_dir = source_dir.expanduser()
    output_dir = output_dir.expanduser()
    if graph_direction not in {"directed", "bidirectional"}:
        raise ValueError("graph_direction must be directed or bidirectional")
    edge_source = _resolve_single(source_dir, "ml_*.csv", required=True)
    assert edge_source is not None
    feature_source = _resolve_edge_feature(source_dir)

    raw = pd.read_csv(edge_source)
    required = {"u", "i", "ts", "idx"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{edge_source} is missing DGB columns {sorted(missing)}")
    eids = pd.to_numeric(raw["idx"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(eids, np.arange(1, len(raw) + 1, dtype=np.int64)):
        raise ValueError("DGB idx must be contiguous and 1-based")
    times, source_time_multiplier = _integerize_timestamps(raw["ts"])
    if dataset.lower() in TIMESTAMP_INDEX_DATASETS:
        times = _normalize_timestamp_indices(times)
        time_representation = "contiguous_unique_timestamp_index"
    else:
        time_representation = "scaled_integer"

    edges = pd.DataFrame(
        {
            "eid": eids,
            "src": pd.to_numeric(raw["u"], errors="raise").to_numpy(dtype=np.int64),
            "dst": pd.to_numeric(raw["i"], errors="raise").to_numpy(dtype=np.int64),
            "time": times,
        }
    )
    edges["default_split"] = assign_temporal_splits(
        edges,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        strategy=split_strategy,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = output_dir / "edges.csv"
    if edges_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {edges_path}")
    edges_tmp = edges_path.with_suffix(".csv.tmp")
    edges.to_csv(edges_tmp, index=False)
    edges_tmp.replace(edges_path)

    artifacts: dict[str, Any] = {}
    if feature_source is not None:
        features = np.load(feature_source, mmap_mode="r", allow_pickle=False)
        feature_meta: dict[str, Any] = {
            "source": feature_source.name,
            "shape": list(features.shape),
            "source_dtype": str(features.dtype),
        }
        if features.size and np.count_nonzero(features) == 0:
            feature_meta["representation"] = "implicit_zeros"
        else:
            target_dtype = EDGE_FEATURE_DTYPES.get(dataset.lower(), features.dtype)
            cast = np.asarray(features, dtype=target_dtype)
            if not np.array_equal(features, cast.astype(features.dtype, copy=False)):
                raise ValueError(
                    f"lossy edge feature conversion {features.dtype} -> {target_dtype}"
                )
            feature_path = output_dir / "edge_features.npy"
            if feature_path.exists() and not force:
                raise FileExistsError(
                    f"refusing to overwrite existing output: {feature_path}"
                )
            np.save(feature_path, cast)
            feature_meta.update({"representation": "npy", "dtype": str(cast.dtype)})
        artifacts["edge_features"] = feature_meta

    graph_path = output_dir / graph_filename(graph_direction)
    graph_summary = write_temporal_csr(
        edges_path,
        graph_path,
        add_reverse=graph_direction == "bidirectional",
        force=force,
    )
    negative_path = output_dir / "neg_pre_sampling_1_result.csv"
    generate_precomputed_negative_file(
        edges_path,
        output_path=negative_path,
        seed=negative_seed,
        force=force,
    )

    split_counts = edges["default_split"].value_counts().sort_index()
    split_time_ranges = [
        [
            int(edges.loc[edges["default_split"] == split, "time"].min()),
            int(edges.loc[edges["default_split"] == split, "time"].max()),
        ]
        for split in range(3)
    ]
    manifest = {
        "schema_version": 1,
        "dataset": output_dir.name,
        "upstream": {
            "project": "DGB",
            "dataset": dataset,
            "zenodo_record": DGB_ZENODO_RECORD,
            "url": DGB_ZENODO_URL,
            "license": "not_specified_by_upstream_record",
            "edge_file": edge_source.name,
            "edge_file_sha256": _sha256(edge_source),
        },
        "edge_schema": ["eid", "src", "dst", "time", "default_split"],
        "time": {
            "representation": time_representation,
            "source_multiplier": source_time_multiplier,
        },
        "split": {
            "strategy": split_strategy,
            "ratios_requested": [train_ratio, val_ratio, 1.0 - train_ratio - val_ratio],
            "counts": [int(split_counts.get(split, 0)) for split in range(3)],
            "time_ranges": split_time_ranges,
            "tied_timestamp_boundaries": [False, False],
        },
        "graph": {
            "event_direction": "upstream",
            "sampling_adjacency": graph_direction,
            "artifact": graph_path.name,
            "stored_adjacencies": graph_summary["stored_adjacencies"],
        },
        "negative_sampling": {
            "use": "evaluation_only",
            "strategy": "fixed_random_destination_corruption",
            "candidate_set": "observed_dst_nodes",
            "generator": "python_random_v1",
            "seed": negative_seed,
            "artifact": negative_path.name,
        },
        "artifacts": artifacts,
    }
    _write_json(output_dir / "manifest.json", manifest, force=force)
    return manifest
