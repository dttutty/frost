from __future__ import annotations

from typing import Literal, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

NeighborStrategy = Literal["recent", "uniform", "time_interval_aware"]

# Define 32-bit arrays
Int32Array1D: TypeAlias = npt.NDArray[np.int32]
Int32Array2D: TypeAlias = npt.NDArray[np.int32]
Int32Array3D: TypeAlias = npt.NDArray[np.int32]

# Define 64-bit arrays
Int64Array1D: TypeAlias = npt.NDArray[np.int64]
Int64Array2D: TypeAlias = npt.NDArray[np.int64]
Int64Array3D: TypeAlias = npt.NDArray[np.int64]

# Time arrays
Float32Array1D: TypeAlias = npt.NDArray[np.float32]
Float32Array2D: TypeAlias = npt.NDArray[np.float32]
Float32Array3D: TypeAlias = npt.NDArray[np.float32]
Int64TimeArray1D: TypeAlias = npt.NDArray[np.int64]
Int64TimeArray2D: TypeAlias = npt.NDArray[np.int64]
Int64TimeArray3D: TypeAlias = npt.NDArray[np.int64]

def plan_owned_routes(
    updates: Sequence[Int64Array1D],
    needs: Sequence[Int64Array1D],
    prefer_higher_rank: bool = True,
) -> tuple[
    list[Int64Array1D],
    Int64Array1D,
    list[list[Int64Array1D]],
]: ...

def materialize_node_scopes_32(
    nid: Int32Array1D,
    n_root: int,
    n_positive_root: int,
    positive_neighbor_start: int,
    positive_neighbor_end: int,
    requested_scopes: int,
) -> tuple[list[Int32Array1D], Int64Array1D]: ...

def materialize_node_scopes_64(
    nid: Int64Array1D,
    n_root: int,
    n_positive_root: int,
    positive_neighbor_start: int,
    positive_neighbor_end: int,
    requested_scopes: int,
) -> tuple[list[Int64Array1D], Int64Array1D]: ...

class TemporalGraph_32_32:
    """Temporal graph stored in CSR-like format.

    Parameters
    ----------
    indptr:
        1D int64 array of shape ``[num_nodes + 1]``.
    indices:
        1D int64 array of destination/source neighbor node ids.
    eid:
        1D int64 array of edge ids aligned with ``indices``.
    ts:
        1D float32 array of edge timestamps aligned with ``indices``.
    """

    indptr: Int32Array1D
    indices: Int32Array1D
    eid: Int32Array1D
    ts: Float32Array1D
    def __init__(
        self,
        indptr: Int32Array1D,
        indices: Int32Array1D,
        eid: Int32Array1D,
        ts: Float32Array1D,
    ) -> None: ...

class SeqSampler_32_32:
    """Continuous-time sequence sampler.

    Parameters
    ----------
    graph:
        Temporal graph.
    sample_neighbor_strategy:
        Sampling strategy. Supported values: ``"recent"``, ``"uniform"``,
        or ``"time_interval_aware"``.
    seed:
        Optional random seed.
    time_scaling_factor:
        Only used by ``"time_interval_aware"``.
    """
    def __init__(
        self,
        graph: TemporalGraph_32_32,
        sample_neighbor_strategy: NeighborStrategy = "recent",
        seed: int | None = None,
        time_scaling_factor: float = 0.0,
    ) -> None: ...
    def get_historical_neighbors(
        self,
        nid: Sequence[int],
        ts: Sequence[float],
        n_ngh: int = 20,
    ) -> tuple[Int32Array2D, Int32Array2D, Float32Array2D]: ...
    def get_historical_neighbors_into(
        self,
        nid: Int32Array1D,
        ts: Float32Array1D,
        out_n: Int32Array2D,
        out_e: Int32Array2D,
        out_t: Float32Array2D,
    ) -> None: ...
    # ... other methods ...

# --- 64-bit Classes ---
class TemporalGraph_64_32:
    """Temporal graph stored in CSR-like format.

    Parameters
    ----------
    indptr:
        1D int64 array of shape ``[num_nodes + 1]``.
    indices:
        1D int64 array of destination/source neighbor node ids.
    eid:
        1D int64 array of edge ids aligned with ``indices``.
    ts:
        1D float32 array of edge timestamps aligned with ``indices``.
    """

    indptr: Int64Array1D
    indices: Int64Array1D
    eid: Int64Array1D
    ts: Float32Array1D
    def __init__(
        self,
        indptr: Int64Array1D,
        indices: Int64Array1D,
        eid: Int64Array1D,
        ts: Float32Array1D,
    ) -> None: ...

class SeqSampler_64_32:
    """Continuous-time sequence sampler.

    Parameters
    ----------
    graph:
        Temporal graph.
    sample_neighbor_strategy:
        Sampling strategy. Supported values: ``"recent"``, ``"uniform"``,
        or ``"time_interval_aware"``.
    seed:
        Optional random seed.
    time_scaling_factor:
        Only used by ``"time_interval_aware"``.
    """
    def __init__(
        self,
        graph: TemporalGraph_64_32,
        sample_neighbor_strategy: NeighborStrategy = "recent",
        seed: int | None = None,
        time_scaling_factor: float = 0.0,
    ) -> None: ...
    def get_historical_neighbors(
        self,
        nid: Sequence[int],
        ts: Sequence[float],
        n_ngh: int = 20,
    ) -> tuple[Int64Array2D, Int64Array2D, Float32Array2D]: ...
    def get_historical_neighbors_into(
        self,
        nid: Int64Array1D,
        ts: Float32Array1D,
        out_n: Int64Array2D,
        out_e: Int64Array2D,
        out_t: Float32Array2D,
    ) -> None: ...

class TemporalGraph_64_64:
    """Temporal graph stored in CSR-like format with int64 timestamps."""

    indptr: Int64Array1D
    indices: Int64Array1D
    eid: Int64Array1D
    ts: Int64TimeArray1D
    def __init__(
        self,
        indptr: Int64Array1D,
        indices: Int64Array1D,
        eid: Int64Array1D,
        ts: Int64TimeArray1D,
    ) -> None: ...

class SeqSampler_64_64:
    """Continuous-time sequence sampler operating on int64 timestamps."""

    def __init__(
        self,
        graph: TemporalGraph_64_64,
        sample_neighbor_strategy: NeighborStrategy = "recent",
        seed: int | None = None,
        time_scaling_factor: float = 0.0,
    ) -> None: ...
    def get_historical_neighbors(
        self,
        nid: Sequence[int],
        ts: Sequence[int],
        n_ngh: int = 20,
    ) -> tuple[Int64Array2D, Int64Array2D, Int64TimeArray2D]: ...
    def get_historical_neighbors_into(
        self,
        nid: Int64Array1D,
        ts: Int64TimeArray1D,
        out_n: Int64Array2D,
        out_e: Int64Array2D,
        out_t: Int64TimeArray2D,
    ) -> None: ...
