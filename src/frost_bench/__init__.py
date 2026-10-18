"""Experiment orchestration for FROST and external baselines.

This package deliberately sits outside :mod:`frost`: benchmark integration may
depend on external command-line interfaces, while the training runtime must not.
"""

from frost_bench.spec import BatchSizeScope, ExperimentSpec, PreparedRun

__all__ = ["BatchSizeScope", "ExperimentSpec", "PreparedRun"]
