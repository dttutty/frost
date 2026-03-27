from typing import Literal, Optional, Union

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
    strategy: Literal["recent", "uniform"] = "recent",
    seed: Optional[int] = None,
) -> SeqSamplerType:
    """Create a SeqSampler instance matching the graph type."""
    if isinstance(graph, _core.TemporalGraph_32_32):
        return _core.SeqSampler_32_32(graph, strategy, seed)
    elif isinstance(graph, _core.TemporalGraph_64_32):
        return _core.SeqSampler_64_32(graph, strategy, seed)
    elif isinstance(graph, _core.TemporalGraph_64_64):
        return _core.SeqSampler_64_64(graph, strategy, seed)
    raise TypeError("Invalid graph type.")


# Expose pseudo-classes and types
__all__ = [
    "TemporalGraph",
    "SeqSampler",
    "TemporalGraphType",
    "SeqSamplerType",
]
