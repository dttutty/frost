"""CSV inputs and explicit node-feature alignment for the prediction interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"{path}: CSV contains no rows.")
    return frame


def column(frame: pd.DataFrame, requested: str | None, aliases: tuple[str, ...]) -> str:
    if requested is not None:
        if requested not in frame:
            raise ValueError(f"Missing column {requested!r}.")
        return requested
    for name in aliases:
        if name in frame:
            return name
    raise ValueError(f"Expected one of these columns: {', '.join(aliases)}.")


def ids(values) -> list[str]:
    result = [str(value).strip() for value in values]
    if any(not value for value in result):
        raise ValueError("Node IDs must not be empty.")
    return result


def times(values, kind: str | None = None) -> tuple[np.ndarray, str]:
    series = pd.Series(list(values), dtype=str)
    numeric = pd.to_numeric(series, errors="coerce")
    if kind in ("numeric", "order") or (kind is None and numeric.notna().all()):
        result = numeric.to_numpy(dtype=np.float64)
        resolved = kind or "numeric"
    else:
        if numeric.notna().any():
            raise ValueError("Do not mix numeric timestamps and datetime strings.")
        dates = pd.to_datetime(series, utc=True, format="mixed", errors="coerce")
        if dates.isna().any():
            raise ValueError(
                "Invalid timestamp; use finite numbers or ISO-8601 datetimes."
            )
        result = np.array([value.timestamp() for value in dates], dtype=np.float64)
        resolved = "datetime"
    if not np.isfinite(result).all():
        raise ValueError(
            "Timestamps must be finite and must match the training time format."
        )
    return result, resolved


def events(path, src_col=None, dst_col=None, time_col=None):
    frame = read_csv(path)
    src = ids(frame[column(frame, src_col, ("src", "src_id", "u"))])
    dst = ids(frame[column(frame, dst_col, ("dst", "dst_id", "i"))])
    detected = time_col or next(
        (c for c in ("time", "timestamp", "ts") if c in frame), None
    )
    if detected is None:
        ts, kind = np.arange(1, len(frame) + 1, dtype=np.float64), "order"
    else:
        ts, kind = times(frame[column(frame, detected, ())])
    return src, dst, ts, kind


def load_features(
    path: str | Path | None, node_ids: list[str], ids_path=None
) -> np.ndarray:
    if path is None:
        if ids_path is not None:
            raise ValueError("--feature-ids requires --node-features.")
        return np.empty((len(node_ids), 0), dtype=np.float32)
    features = np.load(path, allow_pickle=False)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("Node features must be a nonempty 2-D numeric NPY array.")
    if features.dtype.kind not in "fiu" or not np.isfinite(features).all():
        raise ValueError("Node features must contain finite numeric values.")
    sidecar = Path(str(path) + ".ids.json") if ids_path is None else Path(ids_path)
    if sidecar.exists():
        row_ids = ids(json.loads(sidecar.read_text()))
        if len(row_ids) != len(features) or len(set(row_ids)) != len(row_ids):
            raise ValueError("Feature ID list must have one unique ID per NPY row.")
        lookup = {node: index for index, node in enumerate(row_ids)}
        missing = [node for node in node_ids if node not in lookup]
        if missing:
            raise ValueError(f"Missing node features for IDs: {missing[:5]}.")
        rows = [lookup[node] for node in node_ids]
    else:
        if ids_path is not None:
            raise ValueError(f"Feature ID file does not exist: {ids_path}.")
        if any(not node.isdecimal() or str(int(node)) != node for node in node_ids):
            raise ValueError(
                "Without a feature ID file, IDs must be canonical nonnegative integers."
            )
        rows = [int(node) for node in node_ids]
        if max(rows) >= len(features):
            raise ValueError(
                "NPY row n must contain features for node ID n; an ID is out of range."
            )
    aligned = features[rows].astype(np.float32)
    if not np.isfinite(aligned).all():
        raise ValueError("Node features overflow float32.")
    return aligned


def convert_features(csv_path, output, id_col="n_id", categorical_dim=64):
    """Standardize numeric columns and hash categorical values into bounded width."""
    from sklearn.feature_extraction import FeatureHasher

    if categorical_dim < 1:
        raise ValueError("Categorical dimension must be positive.")
    target = Path(output)
    if target.suffix != ".npy":
        raise ValueError("Feature output must end in .npy.")
    sidecar = Path(str(target) + ".ids.json")
    schema_path = Path(str(target) + ".schema.json")
    if any(p.exists() for p in (target, sidecar, schema_path)):
        raise ValueError(
            "Feature output or sidecar already exists; choose a new output path."
        )
    frame = read_csv(csv_path)
    node_ids = ids(frame[column(frame, id_col, ())])
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Feature CSV must have one row per unique node ID.")
    columns = frame.drop(columns=[id_col])
    if columns.shape[1] == 0:
        raise ValueError("Feature CSV must contain at least one feature column.")
    blocks, numeric_info, categorical = [], {}, []
    for name in columns:
        raw = columns[name].str.strip()
        numeric = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
        if numeric[raw != ""].notna().all() and (raw != "").any():
            values = numeric.to_numpy(dtype=np.float64)
            observed = values[~np.isnan(values)]
            if not np.isfinite(observed).all():
                raise ValueError(f"Nonfinite numeric values in {name}.")
            mean = float(observed.mean())
            scale = float(observed.std()) or 1.0
            blocks.append(((np.nan_to_num(values, nan=mean) - mean) / scale)[:, None])
            numeric_info[name] = {"mean": mean, "scale": scale}
        else:
            categorical.append(name)
    if categorical:
        tokens = [
            [f"{name}={value}" for name, value in zip(categorical, row)]
            for row in columns[categorical].itertuples(index=False, name=None)
        ]
        blocks.append(
            FeatureHasher(n_features=categorical_dim, input_type="string")
            .transform(tokens)
            .toarray()
        )
    array = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Converted features overflow float32.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    sidecar.write_text(json.dumps(node_ids, ensure_ascii=False) + "\n")
    schema_path.write_text(
        json.dumps(
            {
                "numeric": numeric_info,
                "categorical": categorical,
                "categorical_dim": categorical_dim,
            },
            indent=2,
        )
        + "\n"
    )
    return array.shape
