"""Machine-local dataset root configuration and dataset discovery."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ENV_PATH = REPO_ROOT / ".env.local"

# Explicit shell exports win over machine-local file settings.
load_dotenv(LOCAL_ENV_PATH, override=False)


def _configured_root(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser().resolve()
    raise RuntimeError(
        f"{env_name} is not configured. Set it in the environment or in "
        f"{LOCAL_ENV_PATH}."
    )


def frost_data_root() -> Path:
    """Return the root containing materialized FROST artifact datasets."""
    return _configured_root("FROST_DATA_ROOT")


def tgb_data_root() -> Path:
    """Return the root containing official TGB downloads under ``raw/``."""
    return _configured_root("TGB_DATA_ROOT")


def resolve_child_directory(root: Path, name: str) -> Path:
    """Resolve one direct child case-insensitively and reject ambiguity."""
    direct = root / name
    if direct.is_dir():
        return direct

    lower = root / name.lower()
    if lower.is_dir():
        return lower

    matches: list[Path] = []
    if root.is_dir():
        matches = [
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.lower() == name.lower()
        ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Dataset directory {name!r} is ambiguous under {root}: {matches}"
        )

    available = []
    if root.is_dir():
        available = sorted(child.name for child in root.iterdir() if child.is_dir())
    raise FileNotFoundError(
        f"Could not resolve dataset directory for {name} under {root}. "
        f"Available datasets: {available}"
    )


def resolve_artifact_dataset_dir(
    dataset: str, *, data_root: str | Path | None = None
) -> Path:
    """Resolve a materialized dataset under the configured FROST artifact root."""
    root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else frost_data_root()
    )
    return resolve_child_directory(root, dataset)


_TGB_RAW_DIRECTORY_NAMES = {
    "TGBL_WIKI": "tgbl_wiki",
    "TGBL_COIN": "tgbl_coin",
    "TGBL_COMMENT": "tgbl_comment",
    "TGBL_FLIGHT": "tgbl_flight",
    "TGBL_REVIEW": "tgbl_review",
}

_TGB_RAW_STATIC_FEATURES = {
    "TGBL_WIKI": (False, True),
    "TGBL_COIN": (False, True),
    "TGBL_COMMENT": (False, True),
    "TGBL_FLIGHT": (True, True),
    "TGBL_REVIEW": (False, True),
}


def is_raw_tgb_dataset(dataset: str) -> bool:
    return dataset.strip().upper() in _TGB_RAW_DIRECTORY_NAMES


def static_features_present_on_disk(
    dataset: str,
    *,
    ignore_node_features: bool = False,
    ignore_edge_features: bool = False,
) -> bool:
    """Probe static-feature presence without importing the training runtime.

    Launch uses this before Torch/CUDA worker initialization so automatic NUMA
    affinity is in place before allocators and thread pools first-touch memory.
    DatasetManager remains the authority for loading and validating the files.
    """
    if is_raw_tgb_dataset(dataset):
        has_node_features, has_edge_features = _TGB_RAW_STATIC_FEATURES[
            dataset.strip().upper()
        ]
        return (
            has_node_features and not ignore_node_features
        ) or (has_edge_features and not ignore_edge_features)

    dataset_dir = resolve_artifact_dataset_dir(dataset)
    if not ignore_node_features and any(
        (dataset_dir / f"node_features.{suffix}").exists()
        for suffix in ("npy", "pt")
    ):
        return True
    return not ignore_edge_features and any(
        (dataset_dir / f"edge_features.{suffix}").exists()
        for suffix in ("npy", "pt")
    )


def resolve_tgb_raw_dataset_dir(
    dataset: str, *, data_root: str | Path | None = None
) -> Path:
    """Resolve an official raw TGB dataset directory."""
    normalized = dataset.strip().upper()
    try:
        raw_name = _TGB_RAW_DIRECTORY_NAMES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"No raw TGB dataset source is registered for {dataset!r}."
        ) from exc
    root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else tgb_data_root()
    )
    return resolve_child_directory(root / "raw", raw_name)
