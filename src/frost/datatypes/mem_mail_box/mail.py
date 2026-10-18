"""Mail entry datatype representing aggregated messages destined for nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class MailEntry:
    """A batch of mail messages aggregated over source nodes for a single step.

    Unlike ``BoxEntry``, ``MailEntry`` holds a single message per row (no
    slot dimension).  It is typically produced by a mailbox router and then
    converted to a single-slot ``BoxEntry`` for mailbox writeback.

    Attributes:
        nid: Destination node IDs, shape ``[N]``.
        data: Message content tensor, shape ``[N, D]``.
        ts: Timestamps of the messages, shape ``[N]``.
        is_sorted: Whether ``nid`` is sorted in ascending order.
        is_unique: Whether all entries in ``nid`` are distinct.
    """

    nid: Tensor
    data: Tensor
    ts: Tensor
    is_sorted: bool = False
    is_unique: bool = False

    def __repr__(self) -> str:
        return (
            f"MailEntry(nid={self.nid.shape}, data={self.data.shape}, "
            f"ts={self.ts.shape}, is_sorted={self.is_sorted}, is_unique={self.is_unique})"
        )

    def sort_by_nid(self) -> MailEntry:
        if self.is_sorted:
            return self
        sorted_indices = torch.argsort(self.nid)
        self.nid = self.nid[sorted_indices]
        self.data = self.data[sorted_indices]
        self.ts = self.ts[sorted_indices]
        self.is_sorted = True
        return self
