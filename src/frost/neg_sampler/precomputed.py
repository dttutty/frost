"""Precomputed negative sampler — reads negatives from a pre-generated CSV file."""
from __future__ import annotations

from pathlib import Path

import torch

from frost.neg_sampler._utils import (
    PRECOMPUTED_NEGATIVE_FILENAME,
    _load_precomputed_negatives,
    resolve_dataset_dir,
    resolve_precomputed_negative_path,
)


class PrecomputedNegSampler:
    def __init__(
        self,
        n_nodes: int,
        n_neg: int,
        dataset: str,
        data_root: str | Path = "DATA",
        neg_file: str = PRECOMPUTED_NEGATIVE_FILENAME,
    ) -> None:
        self.n_nodes = n_nodes  # from hydra config
        self.n_neg = n_neg
        self.dataset = dataset
        self.data_root = Path(data_root)

        if self.n_neg != 1:
            raise ValueError(
                "PrecomputedNegSampler currently requires n_neg=1 because it reads "
                f"a precomputed file with one negative per edge."
            )

        dataset_dir = resolve_dataset_dir(self.dataset, data_root=self.data_root)
        self.precomputed_path = dataset_dir / neg_file
        if not self.precomputed_path.is_file():
            self.precomputed_path = resolve_precomputed_negative_path(
                dataset=self.dataset,
                data_root=self.data_root,
            )
        self.precomputed_negatives = _load_precomputed_negatives(self.precomputed_path)

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        if src_ids.numel() != edge_ids.numel():
            raise ValueError(
                f"src_ids and edge_ids must have the same length, got {src_ids.numel()} "
                f"and {edge_ids.numel()}"
            )

        line_indices = edge_ids.to(dtype=torch.long) - 1
        if line_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        min_idx = int(line_indices.min().item())
        max_idx = int(line_indices.max().item())
        if min_idx < 0 or max_idx >= self.precomputed_negatives.numel():
            raise IndexError(
                f"edge_ids must map into {self.precomputed_path} lines 1.."
                f"{self.precomputed_negatives.numel()}, got min={min_idx + 1}, max={max_idx + 1}"
            )

        return self.precomputed_negatives[line_indices]
