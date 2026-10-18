"""Graph datatypes used to carry sampled subgraph data through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from frost.models.contracts import RootExpansion


@dataclass(kw_only=True)
class FlatGraph:
    """Contiguous flat representation of a sampled subgraph stored in a RingBuf slot.

    All node/edge tensors are concatenated in order: root nodes first, then
    per-hop neighbor tiers.  Call ``unflatten`` to recover the hierarchical
    ``HierGraph`` view before model forward passes.

    Attributes:
        nid: Concatenated node IDs for every tier.
        eid: Concatenated edge IDs for every tier.
        ts: Concatenated timestamps for every tier.
        nfeat: Static node features loaded by H2D, or ``None``.
        efeat: Static edge features loaded by H2D, or ``None``.
        mem: Node memory vectors loaded by H2D, or ``None``.
        mem_ts: Timestamps of the last memory update, or ``None``.
        box: Mailbox slot tensors loaded by H2D, or ``None``.
        box_ts: Timestamps of mailbox slots, or ``None``.
        detached_mem: Post-update memory written by ``TransformerMemoryUpdater``
            for downstream consumption; must be written to this field, not to
            a module-level attribute.
        detached_ts: Timestamps paired with ``detached_mem``.
        detached_nid: Node IDs paired with ``detached_mem``.
    """

    nid: Tensor  # Total length of all nids in HierGraph
    eid: Tensor  # Total length of all eids in HierGraph
    ts: Tensor  # Total length of all timestamps in HierGraph
    nfeat: Tensor | None = None
    efeat: Tensor | None = None
    mem: Tensor | None = None
    mem_ts: Tensor | None = None
    box: Tensor | None = None
    box_ts: Tensor | None = None
    detached_mem: Tensor | None = None
    detached_ts: Tensor | None = None
    detached_nid: Tensor | None = None
    _box_pos_delta: Tensor | None = None
    _box_delta_positions: Tensor | None = None

    def to(self, device: torch.device) -> FlatGraph:
        """Return a new ``FlatGraph`` with all tensors moved to ``device``."""
        # discard detached_mem
        return FlatGraph(
            nid=self.nid.to(device),
            eid=self.eid.to(device),
            ts=self.ts.to(device),
            nfeat=self.nfeat.to(device) if self.nfeat is not None else None,
            efeat=self.efeat.to(device) if self.efeat is not None else None,
            mem=self.mem.to(device) if self.mem is not None else None,
            mem_ts=self.mem_ts.to(device) if self.mem_ts is not None else None,
            box=self.box.to(device) if self.box is not None else None,
            box_ts=self.box_ts.to(device) if self.box_ts is not None else None,
            detached_mem=self.detached_mem.to(device)
            if self.detached_mem is not None
            else None,
            detached_ts=self.detached_ts.to(device)
            if self.detached_ts is not None
            else None,
            detached_nid=self.detached_nid.to(device)
            if self.detached_nid is not None
            else None,
            _box_pos_delta=self._box_pos_delta.to(device)
            if self._box_pos_delta is not None
            else None,
            _box_delta_positions=self._box_delta_positions.to(device)
            if self._box_delta_positions is not None
            else None,
        )

    def unflatten(
        self,
        n_src: int,
        n_neg: int,
        fanout: list[int] | None = None,
        root_expansion: RootExpansion = RootExpansion.ALL_ROOTS,
    ) -> HierGraph:
        """
        Splits a flattened FlatGraph back into a hierarchical HierGraph
        based on the root node count and sampling configuration.
        """
        if fanout is None:
            fanout = []

        # Root node length is always n_src * (2 + n_neg) in RingBuf
        n_root = n_src * (2 + n_neg)
        sizes = [n_root]

        if fanout and root_expansion is RootExpansion.POSITIVE_ROOTS:
            if len(fanout) > 1:
                raise ValueError("Positive-root expansion only supports one hop.")
            sizes.append(n_src * 2 * fanout[0])
        else:
            # Standard multi-hop or full sampling case
            current_size = n_root
            for k in fanout:
                current_size *= k
                sizes.append(current_size)

        # Helper function to split tensors safely
        def _split(t: Tensor | None) -> tuple:
            if t is None:
                return tuple([None] * len(sizes))
            return torch.split(t, sizes, dim=0)

        # Split all tensors
        nids = _split(self.nid)
        eids = _split(self.eid)
        tss = _split(self.ts)
        nfeats = _split(self.nfeat)
        efeats = _split(self.efeat)
        mems = _split(self.mem)
        mem_tss = _split(self.mem_ts)
        boxs = _split(self.box)
        box_tss = _split(self.box_ts)

        # Reconstruct neighbors
        neighbors = []
        for i in range(len(fanout)):
            idx = i + 1  # idx 0 is reserved for roots
            nbr = NeighborOnHop(
                nid=nids[idx],
                eid=eids[idx],
                ts=tss[idx],
                nfeat=nfeats[idx],
                efeat=efeats[idx],
                mem=mems[idx],
                mem_ts=mem_tss[idx],
                box=boxs[idx],
                box_ts=box_tss[idx],
            )
            neighbors.append(nbr)

        # 5. Build and return the final HierGraph
        # remove root_eid to prevent info leakage,
        return HierGraph(
            root_nid=nids[0],
            root_ts=tss[0],
            root_nfeat=nfeats[0],
            root_mem=mems[0],
            root_mem_ts=mem_tss[0],
            root_box=boxs[0],
            root_box_ts=box_tss[0],
            n_neg=n_neg,
            neighbors=neighbors if neighbors else None,
            fanout=fanout,
        )


@dataclass(kw_only=True)
class NeighborOnHop:
    """Sampled neighbors for a single hop in the multi-hop neighborhood.

    The total number of rows equals the product of root count and all fanout
    values up to this hop.  ``nid == -1`` indicates a padding entry and must
    not be aggregated.

    Attributes:
        nid: Neighbor node IDs; shape ``[prev_tier_size * fanout[k]]``.
        eid: Edge IDs connecting parent tier to these neighbors.
        ts: Timestamps of the corresponding edges.
        nfeat: Static node features for these neighbors, or ``None``.
        efeat: Static edge features for the connecting edges, or ``None``.
        mem: Memory vectors for these neighbors, or ``None``.
        mem_ts: Timestamps of the last memory update, or ``None``.
        box: Mailbox slot tensors for these neighbors, or ``None``.
        box_ts: Timestamps of the mailbox slots, or ``None``.
    """

    # For 1-hop neighbors, length = root_nid length * fanout[0]
    # For 2-hop neighbors, length = root_nid length * fanout[0] * fanout[1]
    # and so on.
    nid: Tensor
    eid: Tensor
    ts: Tensor
    nfeat: Tensor | None = None
    efeat: Tensor | None = None
    mem: Tensor | None = None
    mem_ts: Tensor | None = None
    box: Tensor | None = None
    box_ts: Tensor | None = None


@dataclass(kw_only=True)
class HierGraph:
    """Hierarchical sampled subgraph consumed by encoder modules.

    Produced by ``FlatGraph.unflatten``.  Root nodes represent the
    ``(src, pos_dst, neg_dst×n_neg)`` triplets; ``neighbors`` holds per-hop
    ``NeighborOnHop`` objects indexed from 0 (1-hop) outward.

    Attributes:
        root_nid: Root node IDs, shape ``[n_src * (2 + n_neg)]``.
            ``nid == -1`` marks padding rows that should not be aggregated.
        root_ts: Timestamps of the root interaction events.
        root_nfeat: Static node features for root nodes, or ``None``.
        root_mem: Memory vectors for root nodes, or ``None``.
        root_mem_ts: Timestamps of root nodes' last memory update, or ``None``.
        root_box: Mailbox slot tensors for root nodes, or ``None``.
        root_box_ts: Timestamps of root mailbox slots, or ``None``.
        neighbors: Per-hop neighbor data; ``None`` for 0-hop (JODIE) models.
        fanout: Fanout list used when sampling, e.g. ``[10]`` for 1-hop TGN.
    """

    # If nid is -1, then eid is also -1, indicating padding that should not be aggregated.
    # Length is (1+n_neg)x number of src nodes = (src, pos, neg)
    root_nid: Tensor
    root_ts: Tensor
    root_nfeat: Tensor | None = None
    root_mem: Tensor | None = None
    root_mem_ts: Tensor | None = None
    root_box: Tensor | None = None
    root_box_ts: Tensor | None = None
    n_neg: int = 1

    # Multi-hop neighbors. n_neighbor is a list like [10, 25].
    # Currently TGN only supports 1 hop, while TGAT supports multiple hops.
    neighbors: list[NeighborOnHop] | None = None
    fanout: list[int] | None = None
