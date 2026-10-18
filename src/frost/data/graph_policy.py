"""Temporal-graph direction policy and artifact discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast


GraphDirection = Literal["auto", "directed", "bidirectional"]

DIRECTED_GRAPH_FILENAME = "full_graph_no_reverse_edges.npz"
BIDIRECTIONAL_GRAPH_FILENAME = "full_graph_with_reverse_edges.npz"
TGL_GRAPH_FILENAME = "ext_full.npz"
GRAPH_FILENAMES = (
    DIRECTED_GRAPH_FILENAME,
    BIDIRECTIONAL_GRAPH_FILENAME,
    TGL_GRAPH_FILENAME,
)

# TGL's small interaction graphs expose both directions to temporal sampling.
# Its two large homogeneous graphs retain the direction of the published event.
TGL_OFFICIAL_GRAPH_DIRECTIONS: dict[str, GraphDirection] = {
    "WIKI": "bidirectional",
    "WIKIPEDIA": "bidirectional",
    "TGBL_WIKI": "bidirectional",
    "TGBL_COIN": "bidirectional",
    "TGBL_COMMENT": "bidirectional",
    "TGBL_FLIGHT": "bidirectional",
    "TGBL_REVIEW": "bidirectional",
    "REDDIT": "bidirectional",
    "MOOC": "bidirectional",
    "LASTFM": "bidirectional",
    "GDELT": "directed",
    "MAG": "directed",
}


def normalize_graph_direction(direction: str) -> GraphDirection:
    """Validate and normalize a graph-direction setting."""
    normalized = direction.strip().lower()
    if normalized not in {"auto", "directed", "bidirectional"}:
        raise ValueError(
            "graph_direction must be one of auto, directed, or bidirectional; "
            f"got {direction!r}"
        )
    return cast(GraphDirection, normalized)


def resolve_graph_direction(dataset: str, requested: str = "auto") -> GraphDirection:
    """Resolve ``auto`` using TGL's official dataset topology policy."""
    normalized = normalize_graph_direction(requested)
    if normalized != "auto":
        return normalized
    return TGL_OFFICIAL_GRAPH_DIRECTIONS.get(dataset.upper(), "auto")


def graph_filename(direction: str) -> str:
    """Return the canonical artifact name for an explicit direction."""
    normalized = normalize_graph_direction(direction)
    if normalized == "auto":
        raise ValueError("A graph filename requires an explicit graph direction.")
    if normalized == "directed":
        return DIRECTED_GRAPH_FILENAME
    return BIDIRECTIONAL_GRAPH_FILENAME


def resolve_graph_artifact(dataset_dir: Path) -> Path:
    """Find the single supported temporal-graph artifact in a dataset directory."""
    existing = [
        dataset_dir / name for name in GRAPH_FILENAMES if (dataset_dir / name).is_file()
    ]
    if not existing:
        checked = ", ".join(str(dataset_dir / name) for name in GRAPH_FILENAMES)
        raise FileNotFoundError(
            f"Could not find a temporal graph artifact. Checked: {checked}"
        )
    if len(existing) > 1:
        paths = ", ".join(str(path) for path in existing)
        raise ValueError(
            "Dataset contains multiple temporal graph artifacts; keep exactly one "
            f"to make graph direction unambiguous: {paths}"
        )
    return existing[0]


def infer_graph_direction(*, n_edges: int, n_adjacencies: int) -> GraphDirection:
    """Infer whether a CSR stores original edges only or adds every reverse edge."""
    if n_adjacencies == n_edges:
        return "directed"
    if n_adjacencies == 2 * n_edges:
        return "bidirectional"
    raise ValueError(
        "Temporal graph adjacency count must equal n_edges for a directed graph or "
        "2 * n_edges for a graph with reverse edges; "
        f"got n_edges={n_edges}, n_adjacencies={n_adjacencies}."
    )


def check_graph_direction(
    *, dataset: str, requested: str, n_edges: int, n_adjacencies: int
) -> GraphDirection:
    """Infer the artifact direction and enforce the requested/TGL policy."""
    actual = infer_graph_direction(n_edges=n_edges, n_adjacencies=n_adjacencies)
    expected = resolve_graph_direction(dataset, requested)
    if expected != "auto" and actual != expected:
        raise ValueError(
            f"Dataset {dataset} requires a {expected} temporal graph under the "
            f"selected policy, but its artifact is {actual} "
            f"(n_edges={n_edges}, n_adjacencies={n_adjacencies})."
        )
    return actual
