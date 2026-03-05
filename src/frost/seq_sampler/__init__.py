from typing import Literal, Optional, Union

import numpy as np

from . import _core

# Define type aliases for external type hinting
TemporalGraphType = Union[_core.TemporalGraph_32_32, _core.TemporalGraph_64_32]
SeqSamplerType = Union[_core.SeqSampler_32_32, _core.SeqSampler_64_32]


def TemporalGraph(
    indptr: np.ndarray, indices: np.ndarray, eid: np.ndarray, ts: np.ndarray
) -> TemporalGraphType:
    """Create a TemporalGraph instance dynamically."""
    if indices.dtype == np.int32:
        return _core.TemporalGraph_32_32(indptr, indices, eid, ts)
    elif indices.dtype == np.int64:
        return _core.TemporalGraph_64_32(indptr, indices, eid, ts)
    else:
        raise ValueError("Indices array must be int32 or int64.")


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
    else:
        raise TypeError("Invalid graph type.")


# Expose pseudo-classes and types
__all__ = [
    "TemporalGraph",
    "SeqSampler",
    "TemporalGraphType",
    "SeqSamplerType",
]
