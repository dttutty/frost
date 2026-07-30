"""Random negative link sampler — on-the-fly uniform or dst-partition sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from frost.data.paths import resolve_artifact_dataset_dir
from frost.neg_sampler._utils import _load_unique_dst_nodes


class RandomNegLinkSampler:
    def __init__(
        self,
        n_nodes: int,
        n_neg: int,
        dataset: str,
        data_root: str | Path | None = None,
        candidate_mode: Literal["all", "dst_partition"] = "all",
        seed: int | None = None,
        collision_check: bool = False,
        replacement: bool = True,
        candidate_nodes: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be > 0, got {n_nodes}")
        if n_neg <= 0:
            raise ValueError(f"n_neg must be > 0, got {n_neg}")

        self.n_nodes = n_nodes
        self.n_neg = n_neg
        self.dataset = dataset
        self.data_root = None if data_root is None else Path(data_root)
        self.candidate_mode = candidate_mode
        self.seed = seed
        self.collision_check = collision_check
        self.replacement = replacement

        # State to store true destinations of the current batch for collision check
        self._batch_dst: torch.Tensor | None = None

        if candidate_mode == "all":
            if candidate_nodes is not None:
                raise ValueError(
                    "candidate_nodes is only valid with candidate_mode='dst_partition'."
                )
            self.candidates = torch.arange(n_nodes, dtype=torch.long)
        elif candidate_mode == "dst_partition":
            if candidate_nodes is None:
                dataset_dir = resolve_artifact_dataset_dir(
                    dataset, data_root=self.data_root
                )
                self.candidates = _load_unique_dst_nodes(dataset_dir / "edges.csv")
            else:
                self.candidates = torch.as_tensor(
                    candidate_nodes, dtype=torch.long
                ).reshape(-1)
                if self.candidates.numel() > 1 and not bool(
                    torch.all(self.candidates[1:] > self.candidates[:-1])
                ):
                    raise ValueError(
                        "candidate_nodes must be sorted and contain unique node IDs."
                    )
        else:
            raise ValueError(
                f"Unsupported candidate_mode={candidate_mode!r}. "
                "Expected 'all' or 'dst_partition'."
            )

        if self.candidates.numel() == 0:
            raise ValueError(
                "RandomNegLinkSampler requires at least one candidate node."
            )
        if int(self.candidates[0]) < 0 or int(self.candidates[-1]) >= n_nodes:
            raise ValueError(
                f"candidate node IDs must be within [0, {n_nodes}); got range "
                f"[{int(self.candidates[0])}, {int(self.candidates[-1])}]"
            )

        self._generator: torch.Generator | None = None
        self.reset()

    @staticmethod
    def candidates_from_dst_nodes(
        dst_nodes: np.ndarray, *, n_nodes: int
    ) -> torch.Tensor:
        """Build a sorted destination candidate pool without rereading edges.csv.

        The dataset loader already materializes the destination column.  A boolean
        presence map turns it into the same sorted unique pool as
        :func:`_load_unique_dst_nodes`, while avoiding another Python CSV pass.
        """
        dst = np.asarray(dst_nodes)
        if dst.ndim != 1:
            raise ValueError(f"dst_nodes must be 1-D, got shape {dst.shape}")
        if dst.size == 0:
            raise ValueError("dst_nodes must not be empty")
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be > 0, got {n_nodes}")

        min_dst = int(dst.min())
        max_dst = int(dst.max())
        if min_dst < 0 or max_dst >= n_nodes:
            raise ValueError(
                f"destination node IDs must be within [0, {n_nodes}); got range "
                f"[{min_dst}, {max_dst}]"
            )

        present = np.zeros(n_nodes, dtype=np.bool_)
        present[dst] = True
        candidates = np.flatnonzero(present).astype(np.int64, copy=False)
        return torch.from_numpy(candidates)

    def reset(self) -> None:
        """Reset sampler RNG so eval splits reuse the same negatives every epoch."""
        self._batch_dst = None
        if self.seed is None:
            self._generator = None
            return

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def init_from_df(self, df: object) -> None:
        """Random candidates are fully initialized by the constructor."""
        del df

    def get_last_negative_mask(self) -> None:
        """Random sampling never pads its output."""
        return None

    def set_batch_context(
        self,
        *,
        batch_src: np.ndarray,
        batch_dst: np.ndarray,
        batch_start_time: float,
        batch_end_time: float,
        batch_ts: np.ndarray | None = None,
    ) -> None:
        del batch_src, batch_start_time, batch_end_time, batch_ts
        if self.collision_check:
            self._batch_dst = torch.from_numpy(batch_dst)

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        if src_ids.numel() != edge_ids.numel():
            raise ValueError(
                f"src_ids and edge_ids must have the same length, got {src_ids.numel()} "
                f"and {edge_ids.numel()}"
            )

        n_src = src_ids.numel()
        if n_src == 0:
            return torch.empty(0, dtype=torch.long)

        if not self.replacement:
            max_available = self.candidates.numel() - int(self.collision_check)
            if self.n_neg > max_available:
                raise ValueError(
                    f"Cannot sample n_neg={self.n_neg} without replacement from "
                    f"{self.candidates.numel()} candidates (collision_check={self.collision_check})."
                )

        n_samples = n_src * self.n_neg

        # Fast path: pure random with replacement (original behaviour)
        if not self.collision_check and self.replacement:
            sampled_indices = torch.randint(
                low=0,
                high=self.candidates.numel(),
                size=(n_samples,),
                generator=self._generator,
            )
            return self.candidates[sampled_indices]

        # Slower path: enforces no-replacement and/or collision checking
        if self.collision_check and (
            self._batch_dst is None or self._batch_dst.numel() != n_src
        ):
            raise RuntimeError(
                "collision_check=True requires set_batch_context() to be called before "
                "sample() with the exact same batch size."
            )

        sampled_indices = torch.empty(n_samples, dtype=torch.long)
        batch_dst = self._batch_dst
        if self.collision_check:
            assert batch_dst is not None

        for i in range(n_src):
            pos_dst = batch_dst[i].item() if batch_dst is not None else -1

            if self.replacement:
                # With replacement, but avoid pos_dst
                row_indices: list[int] = []
                while len(row_indices) < self.n_neg:
                    idx = torch.randint(
                        0,
                        self.candidates.numel(),
                        (self.n_neg,),
                        generator=self._generator,
                    )
                    if self.collision_check:
                        cands = self.candidates[idx]
                        idx = idx[cands != pos_dst]
                    row_indices.extend(idx.tolist())
                # Take exactly n_neg
                sampled_indices[i * self.n_neg : (i + 1) * self.n_neg] = torch.tensor(
                    row_indices[: self.n_neg]
                )

            else:
                # Without replacement, optionally avoid pos_dst
                row_indices_set: set[int] = set()
                row_indices = []
                while len(row_indices) < self.n_neg:
                    idx = torch.randint(
                        0,
                        self.candidates.numel(),
                        (self.n_neg * 2,),
                        generator=self._generator,
                    )
                    if self.collision_check:
                        cands = self.candidates[idx]
                        idx = idx[cands != pos_dst]
                    for x in idx.tolist():
                        if x not in row_indices_set:
                            row_indices_set.add(x)
                            row_indices.append(x)
                            if len(row_indices) == self.n_neg:
                                break
                sampled_indices[i * self.n_neg : (i + 1) * self.n_neg] = torch.tensor(
                    row_indices
                )

        return self.candidates[sampled_indices]
