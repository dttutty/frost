"""Paired node-ID / node-feature container with sorting and merging utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

@dataclass(kw_only=True)
class NodeFeatNid:
    """Pairs node IDs with their corresponding static node-feature rows.

    Attributes:
        nid: 1-D integer tensor of node IDs.
        nfeat: 2-D float tensor of shape ``[len(nid), ndim]``.
        is_sorted: Whether ``nid`` is already sorted in ascending order.
    """

    nid: torch.Tensor
    nfeat: torch.Tensor
    is_sorted: bool = False

    def sort_by_nid(self) -> NodeFeatNid:
        """Sort ``nid`` and ``nfeat`` in ascending order of node ID in-place.

        Returns:
            ``self`` after sorting (idempotent if already sorted).
        """
        if self.is_sorted:
            return self

        if self.nid is not None:
            sorted_indices = torch.argsort(self.nid)
            self.nid = self.nid[sorted_indices]

            if self.nfeat is not None:
                self.nfeat = self.nfeat[sorted_indices]

        self.is_sorted = True
        return self

    def merge(self, other: NodeFeatNid) -> NodeFeatNid:
        """Append a non-overlapping feature lookup table in place."""
        # StaticPlanner constructs ``fetch`` as current needs minus the
        # previous batch's available IDs, so production CUDA lookups are
        # disjoint by construction.  Keep the defensive validation for CPU
        # callers and tests without forcing a device-to-host synchronization
        # in every worker batch.
        if (
            self.nid.device.type == "cpu"
            and torch.isin(self.nid, other.nid).any()
        ):
            raise ValueError("NodeFeatNid instances must not overlap.")
        self.nid = torch.cat((self.nid, other.nid), dim=0)
        self.nfeat = torch.cat((self.nfeat, other.nfeat), dim=0)
        self.is_sorted = False
        return self

    def filter_by_nid(self, nid: torch.Tensor) -> NodeFeatNid:
        """Return a new ``NodeFeatNid`` containing only rows whose IDs are in ``nid``.

        Args:
            nid: 1-D tensor of node IDs to retain.

        Returns:
            A new ``NodeFeatNid`` with rows filtered to the given node IDs.
        """
        mask = torch.isin(self.nid, nid)
        nid = self.nid[mask]
        nfeat = self.nfeat[mask]
        return self.__class__(nid=nid, nfeat=nfeat)


@dataclass(frozen=True, kw_only=True)
class NodeFeatureLookup:
    """One or two sorted feature tables that jointly cover a batch.

    ``fetched`` contains rows transferred for the current batch, while
    ``cached`` contains rows retained from the preceding batch.  Keeping the
    tables separate avoids concatenating and reordering their wide feature
    matrices merely to perform an ID lookup.
    """

    fetched: NodeFeatNid | None
    cached: NodeFeatNid | None

    @property
    def sources(self) -> tuple[NodeFeatNid, ...]:
        """Return the populated lookup tables in fresh-then-cached order."""
        return tuple(
            source
            for source in (self.fetched, self.cached)
            if source is not None
        )
