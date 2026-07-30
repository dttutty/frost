"""Shared destination drawing for contextual evaluation samplers."""

from __future__ import annotations

import numpy as np


def draw_negative_destinations(
    *,
    src_ids: np.ndarray,
    observed_edges: set[tuple[int, int]],
    batch_src: np.ndarray,
    batch_dst: np.ndarray,
    all_destinations: np.ndarray,
    n_neg: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Prefer per-source observed destinations and exclude current positives."""
    current_by_src: dict[int, set[int]] = {}
    for src, dst in zip(batch_src.tolist(), batch_dst.tolist()):
        current_by_src.setdefault(src, set()).add(dst)

    preferred_by_src: dict[int, list[int]] = {}
    for src, dst in sorted(observed_edges):
        if dst not in current_by_src.get(src, set()):
            preferred_by_src.setdefault(src, []).append(dst)

    result: list[int] = []
    destinations = all_destinations.tolist()
    for src in src_ids.tolist():
        current = current_by_src.get(src, set())
        candidates = preferred_by_src.get(src) or [
            dst for dst in destinations if dst not in current
        ]
        if not candidates:
            raise RuntimeError(
                f"No valid destination negative remains for source {src}."
            )
        result.extend(
            candidates[rng.randint(0, len(candidates))] for _ in range(n_neg)
        )
    return np.asarray(result, dtype=np.int64)
