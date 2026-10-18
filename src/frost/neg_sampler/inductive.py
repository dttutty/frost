"""Per-source inductive destination negatives (eval only)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from frost.neg_sampler._contextual import draw_negative_destinations

if TYPE_CHECKING:
    import pandas as pd


class InductiveNegLinkSampler:
    """Sample post-training destinations for each current source.

    Negative destinations are sampled from edges that appeared *after*
    ``last_observed_time`` but strictly before the current batch, while
    excluding current positives. FROST keeps the current source fixed, so this
    is a destination-corruption analogue rather than an exact port of DyGLib's
    full-pair inductive protocol.

    Requires ``init_from_df()`` to be called once (done automatically by the
    launcher) and ``set_batch_context()`` before each ``sample()`` call (done
    automatically by the host engine).

    Args:
        n_nodes:            Total number of nodes.
        n_neg:              Number of negative destinations per source.
        dataset:            Dataset name.
        data_root:          Root directory for datasets.
        last_observed_time: The timestamp of the last training edge (train/val
                            split boundary).  Inductive negatives are drawn
                            from edges **after** this time.  Resolved
                            automatically from the DataFrame if ``None``.
        seed:               Random seed.  **Required**.
    """

    def __init__(
        self,
        n_nodes: int,
        n_neg: int,
        dataset: str,
        data_root: str | Path | None = None,
        last_observed_time: float | None = None,
        seed: int = 0,
    ) -> None:
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be > 0, got {n_nodes}")
        if n_neg <= 0:
            raise ValueError(f"n_neg must be > 0, got {n_neg}")
        if seed is None:
            raise ValueError("InductiveNegLinkSampler requires an explicit seed.")

        self.n_nodes = n_nodes
        self.n_neg = n_neg
        self.dataset = dataset
        self.data_root = None if data_root is None else Path(data_root)
        self.seed = seed
        self.last_observed_time = last_observed_time
        self._rng = np.random.RandomState(seed)

        self._src_np: np.ndarray | None = None
        self._dst_np: np.ndarray | None = None
        self._ts_np: np.ndarray | None = None
        self._unique_dst: np.ndarray | None = None

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
        if self.last_observed_time is None:
            if "default_split" not in df.columns:
                raise ValueError(
                    "Resolving last_observed_time requires default_split in edges.csv"
                )
            train_times = df.loc[df["default_split"].eq(0), "time"]
            if train_times.empty:
                raise ValueError(
                    "Cannot resolve last_observed_time from an empty train split"
                )
            self.last_observed_time = float(train_times.max())

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
    # FROST interface
    # ------------------------------------------------------------------

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        """Return ``n_neg`` negative *dst* ids per source, shape ``[n_src * n_neg]``."""
        n_src = src_ids.numel()
        if n_src == 0:
            return torch.empty(0, dtype=torch.long)

        if self._src_np is None:
            raise RuntimeError(
                "InductiveNegLinkSampler.init_from_df() has not been called. "
                "Ensure the launcher calls it after loading the edge DataFrame."
            )

        if self._batch_src is None or self._batch_dst is None:
            assert self._unique_dst is not None
            idx = self._rng.randint(0, len(self._unique_dst), n_src * self.n_neg)
            return torch.from_numpy(self._unique_dst[idx])

        assert self.last_observed_time is not None
        assert self._ts_np is not None and self._dst_np is not None
        inductive_mask = (self._ts_np > self.last_observed_time) & (
            self._ts_np < self._batch_start_time
        )
        inductive_edges = set(
            zip(
                self._src_np[inductive_mask].tolist(),
                self._dst_np[inductive_mask].tolist(),
            )
        )
        assert self._unique_dst is not None
        assert self._batch_src is not None and self._batch_dst is not None
        neg_dsts = draw_negative_destinations(
            src_ids=src_ids.cpu().numpy(),
            observed_edges=inductive_edges,
            batch_src=self._batch_src,
            batch_dst=self._batch_dst,
            all_destinations=self._unique_dst,
            n_neg=self.n_neg,
            rng=self._rng,
        )
        return torch.from_numpy(neg_dsts)

    def get_last_negative_mask(self) -> None:
        """Inductive sampling never pads its output."""
        return None
