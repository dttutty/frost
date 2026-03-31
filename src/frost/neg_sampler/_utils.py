"""Shared path-resolution and file-loading utilities for negative samplers."""
from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PRECOMPUTED_NEGATIVE_FILENAME = "neg_pre_sampling_1_result.csv"
LEGACY_PRECOMPUTED_NEGATIVE_FILENAME = "pre_neg_sampler_result.csv"
_REJECTION_TRIES = 64


def resolve_dataset_dir(
    dataset: str,
    data_root: str | Path = "DATA",
) -> Path:
    """Locate the directory for a given dataset under *data_root*.

    Resolution is attempted in three passes to tolerate case differences
    between the caller-supplied name and the actual directory name on disk:

    1. Exact match  –  ``<data_root>/<dataset>``
    2. Lower-case match  –  ``<data_root>/<dataset.lower()>``
    3. Case-insensitive scan of every direct child of *data_root*

    Args:
        dataset:   Dataset name as referenced in the configuration
                   (e.g. ``"LASTFM"`` or ``"Wikipedia"``).
        data_root: Root directory that contains one sub-folder per dataset.
                   Defaults to ``"DATA"`` relative to the working directory.

    Returns:
        Resolved :class:`~pathlib.Path` pointing to the dataset directory.

    Raises:
        FileNotFoundError: If no matching directory is found under *data_root*.
                           The error message lists all available dataset folders
                           to help diagnose typos.
    """
    root = Path(data_root)

    # Pass 1 – try the name verbatim.
    direct = root / dataset
    if direct.is_dir():
        return direct

    # Pass 2 – try the lower-cased name (common when datasets are stored with
    # all-lower directory names but referred to in upper-case in configs).
    lower = root / dataset.lower()
    if lower.is_dir():
        return lower

    # Pass 3 – full case-insensitive scan over all immediate children.
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == dataset.lower():
                return child

    # None of the passes succeeded; collect available names for a helpful error.
    available = []
    if root.is_dir():
        available = sorted(child.name for child in root.iterdir() if child.is_dir())
    raise FileNotFoundError(
        f"Could not resolve dataset directory for {dataset} under {root}. "
        f"Available datasets: {available}"
    )


def resolve_precomputed_negative_path(
    dataset: str,
    data_root: str | Path = "DATA",
) -> Path:
    """Return the path to the precomputed negative-sample CSV for *dataset*.

    The function first locates the dataset directory via
    :func:`resolve_dataset_dir`, then looks for the pre-generated negative
    sample file.  Two filenames are supported so that older cached files remain
    usable without renaming:

    1. Canonical name – ``neg_pre_sampling_1_result.csv``
    2. Legacy name    – ``pre_neg_sampler_result.csv``

    Args:
        dataset:   Dataset name (e.g. ``"LASTFM"``).
        data_root: Root directory that contains one sub-folder per dataset.

    Returns:
        Absolute :class:`~pathlib.Path` to the precomputed negative file.

    Raises:
        FileNotFoundError: If neither the canonical nor the legacy file exists
                           inside the resolved dataset directory.
    """
    dataset_dir = resolve_dataset_dir(dataset, data_root=data_root)

    # Prefer the canonical (current) filename.
    canonical = dataset_dir / PRECOMPUTED_NEGATIVE_FILENAME
    if canonical.is_file():
        return canonical

    # Fall back to the legacy filename for backward compatibility.
    legacy = dataset_dir / LEGACY_PRECOMPUTED_NEGATIVE_FILENAME
    if legacy.is_file():
        return legacy

    raise FileNotFoundError(
        f"Could not find a precomputed negative file for dataset {dataset}. "
        f"Expected {canonical} or {legacy}."
    )


def _load_precomputed_negatives(path: Path) -> torch.Tensor:
    """Load a precomputed negative-sample file into a 1-D integer tensor.

    The file is expected to contain exactly one non-empty integer per line,
    where line *i* holds the pre-drawn negative destination node ID for the
    *i*-th edge (1-indexed).  Empty lines are treated as a format error.

    Args:
        path: Path to the precomputed negative CSV/text file.

    Returns:
        A 1-D ``torch.long`` tensor of shape ``[num_edges]`` containing one
        negative destination node ID per edge.

    Raises:
        ValueError: If an empty line is encountered, or if the file contains
                    no data at all.
    """
    negatives: list[int] = []
    with path.open(newline="") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            # Blank lines are disallowed – each row must correspond to one edge.
            if not stripped:
                raise ValueError(f"Empty line found in {path} at line {line_no}.")
            negatives.append(int(stripped))
    if not negatives:
        raise ValueError(f"Precomputed negative file is empty: {path}")
    return torch.tensor(negatives, dtype=torch.long)


def _load_unique_dst_nodes(path: Path) -> torch.Tensor:
    """Read an edges CSV and return the sorted set of unique destination node IDs.

    The result is used as the candidate pool for inductive negative sampling:
    any node that has appeared at least once as a destination in the observed
    graph is eligible to be drawn as a negative sample.

    Args:
        path: Path to the edges CSV file.  Must contain at minimum a ``"dst"``
              column (additional columns such as ``src``, ``time``, etc. are
              ignored).

    Returns:
        A 1-D ``torch.long`` tensor of **sorted** unique destination node IDs.

    Raises:
        ValueError: If the CSV is missing the required ``"dst"`` column, or if
                    the file contains no rows.
    """
    unique_dst: set[int] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        # Guard against malformed CSV files before iterating over all rows.
        required_columns = {"dst"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_columns)}"
            )
        for row in reader:
            unique_dst.add(int(row["dst"]))

    if not unique_dst:
        raise ValueError(f"No destination nodes found in {path}")
    # Sort before converting to a tensor so that the output is deterministic
    # regardless of set iteration order.
    return torch.tensor(sorted(unique_dst), dtype=torch.long)


def _pick_negative(
    *,
    dst_candidates: tuple[int, ...],
    current_dst: int,
    observed_neighbors: set[int],
    rng: random.Random,
) -> int:
    """Sample a single **historical-negative** destination node for one edge.

    A candidate is considered *valid* if it is **not** the current positive
    destination and has **not** been seen as a neighbour of the source node in
    earlier edges of the same streaming sequence.  This mirrors the historical
    negative-sampling strategy used in DyGLib.

    Sampling proceeds in two phases to balance speed and correctness:

    1. **Rejection sampling** – draw up to :data:`_REJECTION_TRIES` random
       candidates and return the first valid one.  This is O(1) in the common
       case where the forbidden set is small relative to the candidate pool.
    2. **Linear fallback** – if rejection sampling fails (dense forbidden set),
       scan *dst_candidates* sequentially and return the first valid entry.

    Args:
        dst_candidates:      Tuple of all eligible destination node IDs
                             (typically the universe of observed destinations).
        current_dst:         The true positive destination for this edge;
                             must not be returned as a negative.
        observed_neighbors:  Set of destination nodes already seen from the
                             same source node in earlier timesteps; these are
                             also forbidden (historical negatives must be
                             genuinely unobserved).
        rng:                 Seeded :class:`random.Random` instance used for
                             all draws, ensuring reproducibility.

    Returns:
        A valid negative destination node ID.

    Raises:
        ValueError:  If the forbidden set is so large that no valid candidate
                     remains (i.e. every element of *dst_candidates* is
                     excluded).  Checked upfront to avoid a silent infinite
                     loop.
        RuntimeError: If both phases somehow exhaust all candidates without
                      finding a valid node (should be unreachable given the
                      upfront check, but retained as a safety net).
    """
    # Upfront feasibility check: count distinct forbidden nodes.
    # current_dst may already be in observed_neighbors, so add it only once.
    forbidden_count = len(observed_neighbors) + int(current_dst not in observed_neighbors)
    if forbidden_count >= len(dst_candidates):
        raise ValueError(
            "No valid negative destination remains after excluding the current "
            f"positive dst={current_dst} and previously observed neighbors."
        )

    # Phase 1 – fast rejection sampling.
    for _ in range(_REJECTION_TRIES):
        neg_dst = dst_candidates[rng.randrange(len(dst_candidates))]
        if neg_dst != current_dst and neg_dst not in observed_neighbors:
            return neg_dst

    # Phase 2 – linear fallback when the forbidden set is dense.
    for neg_dst in dst_candidates:
        if neg_dst != current_dst and neg_dst not in observed_neighbors:
            return neg_dst

    # Unreachable in practice (guarded by the feasibility check above).
    raise RuntimeError("Failed to draw a valid negative destination.")


def generate_precomputed_negative_file(
    edges_path: str | Path,
    *,
    output_path: str | Path | None = None,
    seed: int = 0,
) -> Path:
    """Pre-generate one historical-negative destination per edge and write to disk.

    This is an **offline preprocessing step**.  The resulting file is consumed
    at training time by the historical-negative sampler so that negative
    destinations never need to be re-sampled on the fly.

    Algorithm
    ---------
    Edges are processed in chronological (eid) order.  For each edge
    ``(src → dst)`` the function draws a negative destination from the full
    node-ID universe ``[0, max_node_id]`` subject to two constraints:

    * The sampled node must not be *current_dst* (the true positive).
    * The sampled node must not have been seen as a destination from *src* in
      any **previous** edge (historical exclusion via :func:`_pick_negative`).

    After sampling, *dst* is added to ``seen_neighbors_by_src[src]`` so that
    future edges from the same source correctly exclude it.

    The file is first written to a ``.tmp`` sibling and then atomically renamed
    to *output_path* to avoid leaving corrupt files on disk if the process is
    interrupted.

    Args:
        edges_path:  Path to the edges CSV file.  Must contain ``eid``, ``src``,
                     and ``dst`` columns; ``eid`` values must be contiguous
                     integers starting at 1 (i.e. row 1 → eid=1).
        output_path: Destination path for the generated file.  Defaults to
                     ``<edges_path.parent>/neg_pre_sampling_1_result.csv``.
        seed:        Integer seed for the RNG.  The actual seed string also
                     incorporates the dataset directory name so that different
                     datasets with the same ``seed`` still produce independent
                     sequences.

    Returns:
        The resolved :class:`~pathlib.Path` of the written output file.

    Raises:
        ValueError:  If the CSV is missing required columns, contains
                     non-contiguous ``eid`` values, or has no data rows.
        ValueError:  (propagated from :func:`_pick_negative`) If the candidate
                     pool is exhausted for any edge.
    """
    edges_path = Path(edges_path)
    # Default output location sits alongside the edges file.
    if output_path is None:
        output_path = edges_path.with_name(PRECOMPUTED_NEGATIVE_FILENAME)
    output_path = Path(output_path)

    # --- Pass 1: validate the file and determine the node-ID universe ---
    max_node_id = -1
    num_rows = 0
    with edges_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"eid", "src", "dst"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{edges_path} is missing required columns: {sorted(missing_columns)}"
            )

        for row_idx, row in enumerate(reader, start=1):
            eid = int(row["eid"])
            # Enforce contiguous eid ordering so that the output file's line
            # number always matches the edge's eid.
            if eid != row_idx:
                raise ValueError(
                    f"{edges_path} has non-contiguous eid values: expected {row_idx}, got {eid}"
                )
            src = int(row["src"])
            dst = int(row["dst"])
            max_node_id = max(max_node_id, src, dst)
            num_rows = row_idx

    if num_rows == 0:
        raise ValueError(f"{edges_path} does not contain any rows.")

    # Build the full candidate pool: every integer node ID in [0, max_node_id].
    candidate_nodes = tuple(range(max_node_id + 1))
    # Seed incorporates the dataset name so different datasets are independent.
    rng = random.Random(f"{seed}:{edges_path.parent.name}")
    # Per-source running history of all observed positive destinations.
    seen_neighbors_by_src: dict[int, set[int]] = {}

    # --- Pass 2: sample negatives and write to a temporary file atomically ---
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with edges_path.open(newline="") as src_f, tmp_path.open("w", newline="") as out_f:
        reader = csv.DictReader(src_f)
        for row_idx, row in enumerate(reader, start=1):
            eid = int(row["eid"])
            if eid != row_idx:
                raise ValueError(
                    f"{edges_path} has non-contiguous eid values: expected {row_idx}, got {eid}"
                )

            src = int(row["src"])
            dst = int(row["dst"])
            # Lazily initialise the per-source neighbour history.
            observed_neighbors = seen_neighbors_by_src.setdefault(src, set())
            neg_dst = _pick_negative(
                dst_candidates=candidate_nodes,
                current_dst=dst,
                observed_neighbors=observed_neighbors,
                rng=rng,
            )
            out_f.write(f"{neg_dst}\n")
            # Record dst **after** sampling so the current edge's positive node
            # is excluded only from *future* edges, not this one.
            observed_neighbors.add(dst)

    # Atomic rename: minimises the window where a partial file is observable.
    tmp_path.replace(output_path)
    logger.info("Generated %s from %s", output_path, edges_path)
    return output_path
