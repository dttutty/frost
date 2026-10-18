"""Precomputed negative sampler — reads negatives from a pre-generated CSV file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from frost.data.paths import resolve_artifact_dataset_dir
from frost.neg_sampler._utils import (
    PRECOMPUTED_NEGATIVE_FILENAME,
    _load_precomputed_negatives,
    resolve_precomputed_negative_path,
)


class PrecomputedNegSampler:
    def __init__(
        self,
        n_nodes: int,
        n_neg: int,
        dataset: str,
        data_root: str | Path | None = None,
        neg_file: str = PRECOMPUTED_NEGATIVE_FILENAME,
        seed: int | None = None,
        candidate_mode: Literal["all", "dst_partition"] = "dst_partition",
    ) -> None:
        self.n_nodes = n_nodes  # from hydra config
        self.n_neg = n_neg
        self.dataset = dataset
        self.data_root = None if data_root is None else Path(data_root)
        self.seed = seed
        self.candidate_mode = candidate_mode
        self._validated = False

        if candidate_mode not in {"all", "dst_partition"}:
            raise ValueError(
                "candidate_mode must be 'all' or 'dst_partition', got "
                f"{candidate_mode!r}"
            )

        if self.n_neg != 1:
            raise ValueError(
                "PrecomputedNegSampler currently requires n_neg=1 because it reads "
                "a precomputed file with one negative per edge."
            )

        dataset_dir = resolve_artifact_dataset_dir(
            self.dataset, data_root=self.data_root
        )
        self.precomputed_path = dataset_dir / neg_file
        if not self.precomputed_path.is_file():
            self.precomputed_path = resolve_precomputed_negative_path(
                dataset=self.dataset,
                data_root=self.data_root,
            )
        self.precomputed_negatives = _load_precomputed_negatives(self.precomputed_path)

    def init_from_df(self, df: Any) -> None:
        """Validate row alignment, candidate domain, and positive collisions."""
        required_columns = {"eid", "dst"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(
                f"Cannot validate {self.precomputed_path}: edges are missing "
                f"columns {sorted(missing)}."
            )
        if len(df) != self.precomputed_negatives.numel():
            raise ValueError(
                f"Row mismatch for {self.precomputed_path}: expected {len(df)} "
                f"negatives, got {self.precomputed_negatives.numel()}."
            )

        edge_ids = torch.as_tensor(df["eid"].to_numpy(copy=True), dtype=torch.long)
        expected_ids = torch.arange(1, len(df) + 1, dtype=torch.long)
        if not torch.equal(edge_ids, expected_ids):
            raise ValueError("edges.csv must use contiguous eid values 1..N.")

        positive_dst = torch.as_tensor(df["dst"].to_numpy(copy=True), dtype=torch.long)
        if self.candidate_mode == "all":
            valid_domain = (self.precomputed_negatives >= 0) & (
                self.precomputed_negatives < self.n_nodes
            )
            domain_description = f"the full node range [0, {self.n_nodes})"
        else:
            candidates = torch.unique(positive_dst, sorted=True)
            positions = torch.searchsorted(candidates, self.precomputed_negatives)
            in_range = positions < candidates.numel()
            valid_domain = torch.zeros_like(in_range)
            valid_domain[in_range] = (
                candidates[positions[in_range]] == self.precomputed_negatives[in_range]
            )
            domain_description = "the observed dst partition"
        if not bool(valid_domain.all()):
            invalid = self.precomputed_negatives[~valid_domain][:5].tolist()
            raise ValueError(
                f"{self.precomputed_path} contains negative destinations outside "
                f"{domain_description}; examples: {invalid}"
            )
        collisions = self.precomputed_negatives == positive_dst
        if bool(collisions.any()):
            collision_eids = edge_ids[collisions][:5].tolist()
            raise ValueError(
                f"{self.precomputed_path} contains positive-edge collisions at "
                f"eids {collision_eids}."
            )
        self._validated = True

    def set_batch_context(self, **context: Any) -> None:
        """Accept the common per-batch hook; edge ids select negatives directly."""

    def reset(self) -> None:
        """Precomputed samples have no mutable iteration state."""

    def get_last_negative_mask(self) -> None:
        """Precomputed samples never pad their output."""
        return None

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        if not self._validated:
            raise RuntimeError(
                "PrecomputedNegSampler.init_from_df() must validate the negative "
                "file before sampling."
            )
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
