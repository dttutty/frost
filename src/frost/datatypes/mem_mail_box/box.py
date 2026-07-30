"""Mailbox (box) entry datatype for per-node circular mail slot storage."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(kw_only=True)
class BoxEntry:
    """Mailbox slot buffer for a batch of nodes.

    Each node holds ``boxsize`` mail slots arranged in a circular FIFO buffer.
    Slot-0 is the oldest entry; slot ``boxsize - 1`` is the most recent.

    Attributes:
        nid: Node IDs, shape ``[N]``.  May contain ``-1`` padding and
            duplicate IDs when loaded directly from the flat graph.
        data: Mail content tensor, shape ``[N, boxsize, D]``.
        ts: Timestamps of each mail slot, shape ``[N, boxsize]``.
        is_sorted: Whether ``nid`` is sorted in ascending order.
        is_unique: Whether all entries in ``nid`` are distinct.
    """

    nid: Tensor
    data: Tensor
    ts: torch.Tensor
    is_sorted: bool = False
    is_unique: bool = False

    def __repr__(self) -> str:
        return (
            f"BoxEntry(nid={self.nid.shape}, data={self.data.shape}, "
            f"ts={self.ts.shape}, is_sorted={self.is_sorted}, is_unique={self.is_unique})"
        )

    def sort_by_nid(self) -> BoxEntry:
        """Return a new ``BoxEntry`` with rows sorted in ascending order of ``nid``.

        Idempotent: returns ``self`` if ``is_sorted`` is already ``True``.
        """
        if self.is_sorted:
            return self
        sorted_indices = torch.argsort(self.nid)

        # Return a new instance; self is not modified (immutable pattern)
        return self.__class__(
            nid=self.nid[sorted_indices],
            data=self.data[sorted_indices],
            ts=self.ts[sorted_indices],
            is_sorted=True,
            is_unique=self.is_unique,
        )

    def to(self, device: torch.device) -> BoxEntry:
        """Return a new ``BoxEntry`` with all tensors moved to ``device``."""
        return BoxEntry(
            nid=self.nid.to(device),
            data=self.data.to(device),
            ts=self.ts.to(device),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )
