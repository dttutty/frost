"""DyGLib-style historical negative sampler (eval only, train uses random)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class HistoricalNegLinkSampler:
    """DyGLib-style *historical* negative sampler ported to the FROST interface.

    Negative destinations are sampled preferentially from (src, dst) pairs that
    appeared historically *before* the current batch but are *not* in the
    current batch.  If the historical pool is smaller than the requested size,
    the remainder is filled with collision-checked random negatives drawn from
    ``unique_dst_node_ids``.

    Requires ``init_from_df()`` to be called once (done automatically by the
    launcher) and ``set_batch_context()`` before each ``sample()`` call (done
    automatically by the host engine).

    Args:
        n_nodes:   Total number of nodes in the graph.
        n_neg:     Number of negative destinations per source.
        dataset:   Dataset name (used to resolve data_root).
        data_root: Root directory for datasets.
        seed:      Random seed.  **Required** for reproducible draws.
    """

    def __init__(
        self,
        n_nodes: int,
        n_neg: int,
        dataset: str,
        data_root: str | Path = "DATA",
        seed: int = 0,
    ) -> None:
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be > 0, got {n_nodes}")
        if n_neg <= 0:
            raise ValueError(f"n_neg must be > 0, got {n_neg}")
        if seed is None:
            raise ValueError("HistoricalNegLinkSampler requires an explicit seed.")

        self.n_nodes = n_nodes
        self.n_neg = n_neg
        self.dataset = dataset
        self.data_root = Path(data_root)
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Populated via init_from_df() called by the launcher.
        self._src_np: np.ndarray | None = None
        self._dst_np: np.ndarray | None = None
        self._ts_np: np.ndarray | None = None
        self._unique_dst: np.ndarray | None = None
        self._earliest_time: float | None = None
        self._possible_edges: set | None = None

        # Set per-batch by set_batch_context().
        self._batch_src: np.ndarray | None = None
        self._batch_dst: np.ndarray | None = None
        self._batch_start_time: float = 0.0
        self._batch_end_time: float = 0.0

    def reset(self) -> None:
        """Reset RNG and per-batch context for deterministic evaluation."""
        self._rng = np.random.RandomState(self.seed)
        self._batch_src = None
        self._batch_dst = None
        self._batch_start_time = 0.0
        self._batch_end_time = 0.0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def init_from_df(self, df: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        """Must be called once after the edge DataFrame is available."""
        self._src_np = df["src"].to_numpy(dtype=np.int64)
        self._dst_np = df["dst"].to_numpy(dtype=np.int64)
        self._ts_np = df["time"].to_numpy(dtype=np.float64)
        self._unique_dst = np.unique(self._dst_np)
        self._earliest_time = float(self._ts_np.min())
        unique_src = np.unique(self._src_np)
        self._possible_edges = {
            (s, d)
            for s in unique_src.tolist()
            for d in self._unique_dst.tolist()
        }

    def set_batch_context(
        self,
        batch_src: np.ndarray,
        batch_dst: np.ndarray,
        batch_start_time: float,
        batch_end_time: float,
    ) -> None:
        self._batch_src = batch_src
        self._batch_dst = batch_dst
        self._batch_start_time = batch_start_time
        self._batch_end_time = batch_end_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _edges_in_interval(self, t_start: float, t_end: float) -> set[tuple[int, int]]:
        assert self._ts_np is not None and self._src_np is not None and self._dst_np is not None
        mask = (self._ts_np >= t_start) & (self._ts_np <= t_end)
        return set(zip(self._src_np[mask].tolist(), self._dst_np[mask].tolist()))

    def _random_with_collision_check(
        self, size: int, batch_edges: set[tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._possible_edges is not None
        candidates = list(self._possible_edges - batch_edges)
        if not candidates:
            raise RuntimeError("No valid negative edges remain after collision check.")
        idx = self._rng.choice(len(candidates), size=size, replace=len(candidates) < size)
        srcs = np.array([candidates[i][0] for i in idx], dtype=np.int64)
        dsts = np.array([candidates[i][1] for i in idx], dtype=np.int64)
        return srcs, dsts

    # ------------------------------------------------------------------
    # FROST interface
    # ------------------------------------------------------------------

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        """Return ``n_neg`` negative *dst* ids per source, shape ``[n_src * n_neg]``."""
        n_src = src_ids.numel()
        if n_src == 0:
            return torch.empty(0, dtype=torch.long)

        size = n_src * self.n_neg

        if self._src_np is None:
            raise RuntimeError(
                "HistoricalNegLinkSampler.init_from_df() has not been called. "
                "Ensure the launcher calls it after loading the edge DataFrame."
            )

        # Fall back to pure random when no batch context has been set yet.
        if self._batch_src is None or self._batch_dst is None:
            assert self._unique_dst is not None
            idx = self._rng.randint(0, len(self._unique_dst), size)
            return torch.from_numpy(self._unique_dst[idx])

        batch_edges = set(zip(self._batch_src.tolist(), self._batch_dst.tolist()))
        historical = self._edges_in_interval(self._earliest_time, self._batch_start_time)
        current = self._edges_in_interval(self._batch_start_time, self._batch_end_time)
        unique_hist = historical - current

        hist_list = list(unique_hist)
        if size > len(hist_list):
            n_random = size - len(hist_list)
            _, rand_dsts = self._random_with_collision_check(n_random, batch_edges)
            hist_dst = np.array([e[1] for e in hist_list], dtype=np.int64)
            neg_dsts = np.concatenate([rand_dsts, hist_dst]).astype(np.int64)
        else:
            idx = self._rng.choice(len(hist_list), size=size, replace=False)
            neg_dsts = np.array([hist_list[i][1] for i in idx], dtype=np.int64)

        return torch.from_numpy(neg_dsts)
