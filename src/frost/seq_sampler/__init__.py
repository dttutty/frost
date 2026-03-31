from collections.abc import Callable
from typing import Any, Literal, Optional, Union

import numpy as np

from . import _core

# Define type aliases for external type hinting
TemporalGraphType = Union[
    _core.TemporalGraph_32_32,
    _core.TemporalGraph_64_32,
    _core.TemporalGraph_64_64,
]
SeqSamplerType = Union[
    _core.SeqSampler_32_32,
    _core.SeqSampler_64_32,
    _core.SeqSampler_64_64,
]


def _instantiate_seq_sampler(
    factory: Callable[..., Any],
    graph: TemporalGraphType,
    strategy: Literal["recent", "uniform", "time_interval_aware"],
    seed: Optional[int],
    time_scaling_factor: float,
) -> SeqSamplerType:
    """Construct a native SeqSampler with a clearer stale-extension failure.

    ``time_scaling_factor`` was added after the initial nanobind constructor
    signature shipped. If the installed ``_core`` extension is older than the
    Python wrapper, nanobind raises a generic ``TypeError`` about incompatible
    arguments. Retrying the legacy 3-argument path keeps zero-scaling callers
    working, while non-zero scaling requests fail with an actionable rebuild
    message instead of the raw nanobind traceback.
    """

    try:
        return factory(graph, strategy, seed, time_scaling_factor)
    except TypeError as exc:
        if "incompatible function arguments" not in str(exc):
            raise
        if time_scaling_factor == 0.0:
            return factory(graph, strategy, seed)
        raise RuntimeError(
            "The installed frost.seq_sampler native extension is stale and "
            "does not accept `time_scaling_factor`. Rebuild it with "
            "`uv pip install -e .` and rerun."
        ) from exc


def TemporalGraph(
    indptr: np.ndarray, indices: np.ndarray, eid: np.ndarray, ts: np.ndarray
) -> TemporalGraphType:
    """Create a TemporalGraph instance dynamically."""
    if indices.dtype == np.int32 and ts.dtype == np.float32:
        return _core.TemporalGraph_32_32(indptr, indices, eid, ts)
    elif indices.dtype == np.int64 and ts.dtype == np.float32:
        return _core.TemporalGraph_64_32(indptr, indices, eid, ts)
    elif indices.dtype == np.int64 and ts.dtype == np.int64:
        return _core.TemporalGraph_64_64(indptr, indices, eid, ts)
    raise ValueError(
        "Unsupported TemporalGraph dtypes. Expected "
        "(int32,float32), (int64,float32), or (int64,int64)."
    )


def SeqSampler(
    graph: TemporalGraphType,
    strategy: Literal["recent", "uniform", "time_interval_aware"] = "recent",
    seed: Optional[int] = None,
    time_scaling_factor: float = 0.0,
) -> SeqSamplerType:
    """Create a SeqSampler instance matching the graph type."""
    if isinstance(graph, _core.TemporalGraph_32_32):
        return _instantiate_seq_sampler(
            _core.SeqSampler_32_32, graph, strategy, seed, time_scaling_factor
        )
    elif isinstance(graph, _core.TemporalGraph_64_32):
        return _instantiate_seq_sampler(
            _core.SeqSampler_64_32, graph, strategy, seed, time_scaling_factor
        )
    elif isinstance(graph, _core.TemporalGraph_64_64):
        return _instantiate_seq_sampler(
            _core.SeqSampler_64_64, graph, strategy, seed, time_scaling_factor
        )
    raise TypeError("Invalid graph type.")


# Expose pseudo-classes and types
__all__ = [
    "TemporalGraph",
    "SeqSampler",
    "TemporalGraphType",
    "SeqSamplerType",
]
