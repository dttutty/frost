"""Per-source historical destination negatives (eval only)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import pandas as pd


class HistoricalNegLinkSampler:
    """Sample previously observed destinations for each current source.

    FROST's root layout keeps the positive source fixed, so this is a
    destination-corruption historical protocol: for each current source it
    prefers destinations connected to that same source strictly before the
    current batch, excluding current positives. If none exist, it falls back to
    the observed destination partition. It is intentionally not labelled as an
    exact port of DyGLib's full-pair protocol, which may replace both endpoints.

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

    def init_from_df(self, df: pd.DataFrame) -> None:
        """Must be called once after the edge DataFrame is available."""
        self._src_np = df["src"].to_numpy(dtype=np.int64)
        self._dst_np = df["dst"].to_numpy(dtype=np.int64)
        self._ts_np = df["time"].to_numpy(dtype=np.float64)
        self._unique_dst = np.unique(self._dst_np)

    def set_batch_context(
        self,
        batch_src: np.ndarray,
        batch_dst: np.ndarray,
        batch_start_time: float,
        batch_end_time: float,
        batch_ts: np.ndarray | None = None,
    ) -> None:
        del batch_ts
        self._batch_src = batch_src
        self._batch_dst = batch_dst
        self._batch_start_time = batch_start_time
        self._batch_end_time = batch_end_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_for_sources(
        self, src_ids: np.ndarray, historical_edges: set[tuple[int, int]]
    ) -> np.ndarray:
        assert self._unique_dst is not None
        assert self._batch_src is not None and self._batch_dst is not None
        current_by_src: dict[int, set[int]] = {}
        for src, dst in zip(self._batch_src.tolist(), self._batch_dst.tolist()):
            current_by_src.setdefault(src, set()).add(dst)
        historical_by_src: dict[int, list[int]] = {}
        for src, dst in sorted(historical_edges):
            if dst not in current_by_src.get(src, set()):
                historical_by_src.setdefault(src, []).append(dst)

        result: list[int] = []
        for src in src_ids.tolist():
            current = current_by_src.get(src, set())
            preferred = historical_by_src.get(src, [])
            fallback = [dst for dst in self._unique_dst.tolist() if dst not in current]
            candidates = preferred or fallback
            if not candidates:
                raise RuntimeError(
                    f"No valid destination negative remains for source {src}."
                )
            for _ in range(self.n_neg):
                result.append(candidates[self._rng.randint(0, len(candidates))])
        return np.asarray(result, dtype=np.int64)

    # ------------------------------------------------------------------
    # FROST interface
    # ------------------------------------------------------------------

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        """Return ``n_neg`` negative *dst* ids per source, shape ``[n_src * n_neg]``."""
        n_src = src_ids.numel()
        if n_src == 0:
            return torch.empty(0, dtype=torch.long)

        if self._src_np is None:
            raise RuntimeError(
                "HistoricalNegLinkSampler.init_from_df() has not been called. "
                "Ensure the launcher calls it after loading the edge DataFrame."
            )

        # Fall back to pure random when no batch context has been set yet.
        if self._batch_src is None or self._batch_dst is None:
            assert self._unique_dst is not None
            idx = self._rng.randint(0, len(self._unique_dst), n_src * self.n_neg)
            return torch.from_numpy(self._unique_dst[idx])

        assert self._ts_np is not None and self._dst_np is not None
        historical_mask = self._ts_np < self._batch_start_time
        historical = set(
            zip(
                self._src_np[historical_mask].tolist(),
                self._dst_np[historical_mask].tolist(),
            )
        )
        neg_dsts = self._draw_for_sources(src_ids.cpu().numpy(), historical)
        return torch.from_numpy(neg_dsts)
