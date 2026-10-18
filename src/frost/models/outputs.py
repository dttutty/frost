"""Structured values exchanged by encoders, representation adapters, and heads."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from frost.models.contracts import RepresentationKind


class EncoderResult(NamedTuple):
    """Standard output of every temporal encoder."""

    embeddings: torch.Tensor


class EdgeRepresentations(NamedTuple):
    """Aligned representations for positive and negative candidate edges."""

    positive_source: torch.Tensor
    positive_destination: torch.Tensor
    negative_source: torch.Tensor
    negative_destination: torch.Tensor


class RepresentationAdapter(nn.Module):
    """Normalize encoder-specific root layouts into explicit edge pairs."""

    def __init__(self, kind: RepresentationKind) -> None:
        super().__init__()
        self.kind = kind

    def forward(
        self,
        result: EncoderResult,
        *,
        n_src: int,
        n_neg: int,
    ) -> EdgeRepresentations:
        if n_src <= 0:
            raise ValueError(f"n_src must be positive, got {n_src}.")
        if n_neg <= 0:
            raise ValueError(f"n_neg must be positive, got {n_neg}.")

        h = result.embeddings
        negative_count = n_src * n_neg
        if self.kind is RepresentationKind.NODE_ALIGNED:
            expected = n_src * (2 + n_neg)
            if h.shape[0] != expected:
                raise ValueError(
                    f"Node-aligned encoder returned {h.shape[0]} rows; "
                    f"expected {expected} for n_src={n_src}, n_neg={n_neg}."
                )
            positive_source = h[:n_src]
            positive_destination = h[n_src : 2 * n_src]
            negative_source = positive_source.repeat_interleave(n_neg, dim=0)
            negative_destination = h[2 * n_src :]
        elif self.kind is RepresentationKind.PAIR_ALIGNED:
            expected = 2 * n_src * (1 + n_neg)
            if h.shape[0] != expected:
                raise ValueError(
                    f"Pair-aligned encoder returned {h.shape[0]} rows; "
                    f"expected {expected} for n_src={n_src}, n_neg={n_neg}."
                )
            positive_source = h[:n_src]
            positive_destination = h[n_src : 2 * n_src]
            negative_source = h[2 * n_src : 2 * n_src + negative_count]
            negative_destination = h[2 * n_src + negative_count :]
        else:
            raise ValueError(f"Unsupported representation kind: {self.kind!r}")

        return EdgeRepresentations(
            positive_source=positive_source,
            positive_destination=positive_destination,
            negative_source=negative_source,
            negative_destination=negative_destination,
        )
