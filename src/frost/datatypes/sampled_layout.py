"""Host-side views over the flat node layout emitted by neighborhood samplers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class SampledLayoutView:
    """Zero-copy structural view over a flat sampled node layout.

    The host-side samplers write node IDs into a single flat tensor following the
    RingBuf layout: roots first, then one tensor slice per hop. This dataclass
    exposes that layout as named views so analyzer logic does not need to repeat
    offset arithmetic.

    Attributes:
        nid: Flat node-ID layout returned by the sampler.
        n_src: Number of positive source edges in the batch.
        n_neg: Number of negative destinations per source edge.
        fanout: Sampling fanout per hop, or ``None`` for 0-hop layouts.
        sample_4_neg: Whether first-hop neighborhoods are also expanded for
            negative roots.
        padding_nid: Sentinel used for missing / padded node slots.
    """

    nid: Tensor
    n_src: int
    n_neg: int
    fanout: list[int] | None
    sample_4_neg: bool = True
    padding_nid: int = -1

    def __post_init__(self) -> None:
        if self.nid.dim() != 1:
            raise ValueError(
                f"SampledLayoutView expects a 1D tensor, got shape {tuple(self.nid.shape)}"
            )
        if self.n_src < 0:
            raise ValueError(f"n_src must be non-negative, got {self.n_src}")
        if self.n_neg < 0:
            raise ValueError(f"n_neg must be non-negative, got {self.n_neg}")

    @cached_property
    def n_root(self) -> int:
        return self.n_src * (2 + self.n_neg)

    @cached_property
    def roots(self) -> Tensor:
        return self.nid[: self.n_root]

    @cached_property
    def src(self) -> Tensor:
        return self.roots[: self.n_src]

    @cached_property
    def pos_dst(self) -> Tensor:
        return self.roots[self.n_src : 2 * self.n_src]

    @cached_property
    def pos_neg(self) -> Tensor:
        return self.roots[2 * self.n_src : self.n_root]

    @cached_property
    def pos_roots(self) -> Tensor:
        return self.roots[: 2 * self.n_src]

    def filter_valid(self, nid: Tensor) -> Tensor:
        """Drop padding entries while preserving the underlying order."""
        return nid[nid != self.padding_nid]

    @cached_property
    def valid_nid(self) -> Tensor:
        return self.filter_valid(self.nid)

    def valid_hop(self, hop_idx: int) -> Tensor:
        """Return one non-APAN hop with all padding entries removed."""
        if hop_idx < 0:
            raise ValueError(f"hop_idx must be non-negative, got {hop_idx}")

        fanout = self.fanout or []
        if not fanout:
            raise IndexError("valid_hop is unavailable for 0-hop layouts.")
        assert self.sample_4_neg, (
            "valid_hop only supports layouts where sample_4_neg=True. "
            "APAN-specific layouts should use apan_pos_hop_0 instead."
        )
        if hop_idx >= len(fanout):
            raise IndexError(
                f"hop_idx={hop_idx} is out of range for {len(fanout)} hops"
            )

        start = self.n_root
        current_size = self.n_root
        for i, per_hop_fanout in enumerate(fanout):
            current_size *= per_hop_fanout
            end = start + current_size
            if i == hop_idx:
                return self.filter_valid(self.nid[start:end])
            start = end

        raise IndexError(f"hop_idx={hop_idx} is out of range for {len(fanout)} hops")

    @cached_property
    def apan_pos_ngh(self) -> Tensor:
        """Return APAN's 1-hop neighbors for ``src`` + ``pos_dst`` roots.

        This is the raw 1-hop slice, so padding entries (``-1``) are preserved.
        Call ``filter_valid(...)`` if the consumer needs only valid node IDs.
        """
        if self.sample_4_neg:
            raise ValueError("apan_pos_hop_0 is only valid when sample_4_neg=False.")
        if self.fanout is None or len(self.fanout) != 1:
            raise ValueError("apan_pos_hop_0 requires a single-hop APAN layout.")
        start = self.n_root
        end = start + self.n_src * 2 * self.fanout[0]
        return self.nid[start:end]
