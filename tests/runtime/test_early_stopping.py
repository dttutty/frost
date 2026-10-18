from __future__ import annotations

from frost.runtime._early_stopping import is_significant_improvement


def test_first_finite_score_is_an_improvement() -> None:
    assert is_significant_improvement(0.2, float("-inf"), 1e-6)


def test_tolerance_is_relative_to_best_score() -> None:
    assert not is_significant_improvement(0.5000004, 0.5, 1e-6)
    assert is_significant_improvement(0.5000006, 0.5, 1e-6)


def test_zero_best_requires_a_strict_increase() -> None:
    assert not is_significant_improvement(0.0, 0.0, 1e-6)
    assert is_significant_improvement(0.1, 0.0, 1e-6)
