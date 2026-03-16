"""Mailbox (box) entry datatype for per-node circular mail slot storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .mail import MailEntry


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

    def clone(self) -> BoxEntry:
        """Return a deep copy of this entry with all tensors cloned."""
        return BoxEntry(
            nid=self.nid.clone(),
            data=self.data.clone(),
            ts=self.ts.clone(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def cuda(self) -> BoxEntry:
        """Return a new ``BoxEntry`` with all tensors moved to the default CUDA device."""
        return BoxEntry(
            nid=self.nid.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            is_unique=self.is_unique,
            is_sorted=self.is_sorted,
        )

    def merge_box(
        self, other: BoxEntry, is_overlap: bool, sort_idx: torch.Tensor | None = None
    ) -> BoxEntry:
        """Concatenate two ``BoxEntry`` instances, sorting the result by ``nid``.

        Args:
            other: The second entry to merge with ``self``.
            is_overlap: Whether the two entries share any node IDs.  When
                ``True``, ``is_unique`` is set to ``False`` on the result.
            sort_idx: Optional pre-computed sort indices for the concatenated
                ``nid``; computed internally if ``None``.

        Returns:
            A new sorted ``BoxEntry`` containing entries from both inputs.
        """
        # 1. Handle empty cases
        if self.nid.numel() == 0:
            return other
        if other.nid.numel() == 0:
            return self

        # 2. Fallback: concatenate and sort
        cat_nid = torch.cat((self.nid, other.nid), dim=0)
        cat_data = torch.cat((self.data, other.data), dim=0)
        cat_ts = torch.cat((self.ts, other.ts), dim=0)

        # Compute sort order
        if sort_idx is None:
            sort_idx = torch.argsort(cat_nid)

        return BoxEntry(
            nid=cat_nid[sort_idx],
            data=cat_data[sort_idx],
            ts=cat_ts[sort_idx],
            is_sorted=True,
            is_unique=self.is_unique and other.is_unique and not is_overlap,
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

    def merge_1_mail(self, mail: MailEntry) -> BoxEntry:
        """Insert one new mail per node into the circular FIFO mailbox buffer.

        Delegates to ``_data_ops.merge_1_mail``.

        Args:
            mail: Deduplicated new mail to insert.

        Returns:
            Updated ``BoxEntry`` with the new mail written into the last slot.
        """
        from ._data_ops import merge_1_mail

        return merge_1_mail(self, mail)
