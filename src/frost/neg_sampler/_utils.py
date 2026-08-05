"""Shared path-resolution and file-loading utilities for negative samplers."""

from __future__ import annotations

import csv
import itertools
import logging
import random
from pathlib import Path
from typing import Literal

import torch

from frost.data.paths import resolve_artifact_dataset_dir

logger = logging.getLogger(__name__)

PRECOMPUTED_NEGATIVE_FILENAME = "neg_pre_sampling_1_result.csv"
LEGACY_PRECOMPUTED_NEGATIVE_FILENAME = "pre_neg_sampler_result.csv"
_REJECTION_TRIES = 64


def resolve_precomputed_negative_path(
    dataset: str,
    data_root: str | Path | None = None,
) -> Path:
    """Return the path to the precomputed negative-sample CSV for *dataset*.

    The function first locates the dataset directory via the shared artifact
    resolver, then looks for the pre-generated negative sample file. Two
    filenames are supported so that older cached files remain usable without
    renaming:

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
    dataset_dir = resolve_artifact_dataset_dir(dataset, data_root=data_root)

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
    """Load an eid-keyed negative-sample CSV into a 1-D integer tensor.

    The canonical format is ``eid,neg_dst`` with contiguous 1-based edge IDs.
    Headerless one-integer-per-line files remain readable for migration, but
    callers should validate their candidate domain against ``edges.csv``.

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
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"Precomputed negative file is empty: {path}")
        if first_line.strip() == "eid,neg_dst":
            reader = csv.DictReader(itertools.chain([first_line], f))
            for row_idx, row in enumerate(reader, start=1):
                eid = int(row["eid"])
                if eid != row_idx:
                    raise ValueError(
                        f"{path} has non-contiguous eid values: expected "
                        f"{row_idx}, got {eid}"
                    )
                negatives.append(int(row["neg_dst"]))
        else:
            logger.warning(
                "Loading legacy headerless negative file %s; regenerate it to "
                "the canonical eid,neg_dst format.",
                path,
            )
            for line_no, line in enumerate(itertools.chain([first_line], f), start=1):
                stripped = line.strip()
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
    rng: random.Random,
) -> int:
    """Sample one destination-corruption negative for an observed edge.

    A candidate is considered *valid* if it is a real destination node, is
    **not** the current positive destination.  This is a fixed random-negative
    protocol, not DyGLib's full-pair historical-negative protocol.

    Sampling proceeds in two phases to balance speed and correctness:

    1. **Rejection sampling** – draw up to :data:`_REJECTION_TRIES` random
       candidates and return the first valid one.  This is O(1) in the common
       case where the forbidden set is small relative to the candidate pool.
    2. **Linear fallback** – if rejection sampling fails (dense forbidden set),
       scan *dst_candidates* sequentially and return the first valid entry.

    Args:
        dst_candidates:      Tuple of all eligible destination node IDs
                             (the observed destination-node partition).
        current_dst:         The true positive destination for this edge;
                             must not be returned as a negative.
        rng:                 Seeded :class:`random.Random` instance used for
                             all draws, ensuring reproducibility.

    Returns:
        A valid negative destination node ID.

    Raises:
        ValueError: If no destination other than ``current_dst`` exists.
    """
    if len(dst_candidates) <= 1 and current_dst in dst_candidates:
        raise ValueError(
            "No valid negative destination remains after excluding the current "
            f"positive dst={current_dst}."
        )

    # Phase 1 – fast rejection sampling.
    for _ in range(_REJECTION_TRIES):
        neg_dst = dst_candidates[rng.randrange(len(dst_candidates))]
        if neg_dst != current_dst:
            return neg_dst

    # Phase 2 – linear fallback when the forbidden set is dense.
    for neg_dst in dst_candidates:
        if neg_dst != current_dst:
            return neg_dst

    raise RuntimeError("Failed to draw a valid negative destination.")


def generate_precomputed_negative_file(
    edges_path: str | Path,
    *,
    output_path: str | Path | None = None,
    seed: int = 0,
    force: bool = False,
    candidate_mode: Literal["all", "dst_partition"] = "dst_partition",
) -> Path:
    """Pre-generate one fixed random-negative destination per edge.

    This is an **offline preprocessing step**.  The resulting file is consumed
    by the evaluation sampler so that negative destinations are stable across
    models and runs.

    Algorithm
    ---------
    Edges are processed in chronological (eid) order.  For each edge
    ``(src → dst)`` the function draws a negative destination from the set
    of node IDs that actually occur in the table's ``dst`` column, subject to
    one constraint: it must not equal the current positive destination.
    Historical positives remain eligible, matching a fixed random
    destination-corruption evaluation rather than an unseen-edge or
    historical-edge benchmark.

    The file is first written to a ``.tmp`` sibling and then atomically renamed
    to *output_path* to avoid leaving corrupt files on disk if the process is
    interrupted.

    Args:
        edges_path:  Path to the edges CSV file.  Must contain ``src`` and
                     ``dst`` columns.  If ``eid`` is present, its values must
                     be contiguous integers starting at 1 (i.e. row 1 →
                     eid=1).  Otherwise eid is synthesized from that canonical
                     row order, matching the runtime edge-table normalizer.
        output_path: Destination path for the generated file.  Defaults to
                     ``<edges_path.parent>/neg_pre_sampling_1_result.csv``.
        seed:        Integer seed for the RNG. The artifact depends only on this
                     seed and the canonical edge table, not its directory name.

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
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    if candidate_mode not in {"all", "dst_partition"}:
        raise ValueError(
            "candidate_mode must be 'all' or 'dst_partition', got "
            f"{candidate_mode!r}"
        )

    # --- Pass 1: validate the file and determine the candidate population ---
    destination_nodes: set[int] = set()
    max_node_id = -1
    num_rows = 0
    with edges_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"src", "dst"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{edges_path} is missing required columns: {sorted(missing_columns)}"
            )

        has_eid = "eid" in (reader.fieldnames or [])
        for row_idx, row in enumerate(reader, start=1):
            # Enforce contiguous eid ordering when it is materialized.  Legacy
            # TGL tables omit this column; the runtime synthesizes the same
            # 1-based identifier from row order.
            if has_eid and int(row["eid"]) != row_idx:
                raise ValueError(
                    f"{edges_path} has non-contiguous eid values: expected "
                    f"{row_idx}, got {row['eid']}"
                )
            src = int(row["src"])
            dst = int(row["dst"])
            if src < 0 or dst < 0:
                raise ValueError("src and dst node IDs must be non-negative")
            destination_nodes.add(dst)
            max_node_id = max(max_node_id, src, dst)
            num_rows = row_idx

    if num_rows == 0:
        raise ValueError(f"{edges_path} does not contain any rows.")

    candidate_nodes = (
        tuple(range(max_node_id + 1))
        if candidate_mode == "all"
        else tuple(sorted(destination_nodes))
    )
    rng = random.Random(seed)
    # --- Pass 2: sample negatives and write to a temporary file atomically ---
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with edges_path.open(newline="") as src_f, tmp_path.open("w", newline="") as out_f:
        reader = csv.DictReader(src_f)
        has_eid = "eid" in (reader.fieldnames or [])
        writer = csv.writer(out_f)
        writer.writerow(["eid", "neg_dst"])
        for row_idx, row in enumerate(reader, start=1):
            if has_eid and int(row["eid"]) != row_idx:
                raise ValueError(
                    f"{edges_path} has non-contiguous eid values: expected "
                    f"{row_idx}, got {row['eid']}"
                )

            dst = int(row["dst"])
            neg_dst = _pick_negative(
                dst_candidates=candidate_nodes,
                current_dst=dst,
                rng=rng,
            )
            writer.writerow([row_idx, neg_dst])

    # Atomic rename: minimises the window where a partial file is observable.
    tmp_path.replace(output_path)
    logger.info("Generated %s from %s", output_path, edges_path)
    return output_path
