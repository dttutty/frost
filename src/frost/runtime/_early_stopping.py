"""Shared validation-metric improvement rule."""

from __future__ import annotations

import math


def is_significant_improvement(
    score: float,
    best_score: float,
    relative_tolerance: float,
) -> bool:
    """Match TGB's relative-tolerance early-stopping comparison."""
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be >= 0.")
    if math.isinf(best_score) and best_score < 0:
        return True
    if best_score == 0:
        return score > best_score
    return (score - best_score) / abs(best_score) > relative_tolerance
