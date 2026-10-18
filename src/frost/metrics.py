"""Metric primitives shared by Frost runtimes and external model adapters."""

from __future__ import annotations

import torch


def tgb_reciprocal_ranks(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    negative_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return TGB-compatible reciprocal ranks for fixed-width candidates.

    Ties use the mean of optimistic and pessimistic ranks. A validity mask can
    exclude padding used to represent variable-width candidate sets.
    """
    if positive_scores.dim() != 1:
        raise ValueError(
            f"positive_scores must be 1D, got shape {tuple(positive_scores.shape)}"
        )
    if negative_scores.dim() != 2:
        raise ValueError(
            f"negative_scores must be 2D, got shape {tuple(negative_scores.shape)}"
        )
    if positive_scores.shape[0] != negative_scores.shape[0]:
        raise ValueError(
            "positive_scores and negative_scores must contain the same number "
            f"of queries, got {positive_scores.shape[0]} and "
            f"{negative_scores.shape[0]}"
        )
    if negative_valid_mask is not None:
        if negative_valid_mask.shape != negative_scores.shape:
            raise ValueError(
                "negative_valid_mask must match negative_scores, got "
                f"{tuple(negative_valid_mask.shape)} and "
                f"{tuple(negative_scores.shape)}"
            )
        negative_valid_mask = negative_valid_mask.to(
            device=negative_scores.device, dtype=torch.bool
        )

    positive_scores = positive_scores.unsqueeze(1)
    if negative_valid_mask is None:
        negative_valid_mask = torch.ones_like(negative_scores, dtype=torch.bool)
    optimistic_rank = ((negative_scores > positive_scores) & negative_valid_mask).sum(
        dim=1
    )
    pessimistic_rank = ((negative_scores >= positive_scores) & negative_valid_mask).sum(
        dim=1
    )
    rank = 0.5 * (optimistic_rank + pessimistic_rank).float() + 1.0
    return rank.reciprocal()
