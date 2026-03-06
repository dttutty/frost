from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class FlatGraph:
    """Sequence sampling metadata."""

    nid: torch.Tensor  # Total length of all nids in SeqGraph
    eid: torch.Tensor  # Total length of all eids in SeqGraph
    ts: torch.Tensor  # Total length of all timestamps in SeqGraph
    nfeat: torch.Tensor | None = None
    efeat: torch.Tensor | None = None
    mem: torch.Tensor | None = None
    mem_ts: torch.Tensor | None = None
    box: torch.Tensor | None = None
    box_ts: torch.Tensor | None = None
    detached_mem: torch.Tensor | None = None
    detached_ts: torch.Tensor | None = None
    detached_nid: torch.Tensor | None = None

    def to(self, device: torch.device) -> FlatGraph:
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
        )

    def unflatten(
        self,
        n_src: int,
        n_neg: int,
        fanout: list[int] | None = None,
        sample_4_neg: bool = True,
    ) -> Graph:
        """
        Splits a flattened ZipSeqGraph back into a hierarchical SeqGraph
        based on the root node count and sampling configuration.
        """
        if fanout is None:
            fanout = []

        # Root node length is always n_src * (2 + n_neg) in RingBuf
        n_root = n_src * (2 + n_neg)
        sizes = [n_root]

        if len(fanout) == 1 and not sample_4_neg:
            # Special case for NghSampler without negative sampling
            sizes.append(n_src * 2 * fanout[0])
        else:
            if len(fanout) > 1 and not sample_4_neg:
                raise ValueError(
                    "sample_4_neg is False(APAN), fanout layer cannot be multi-hop."
                )
            # Standard multi-hop or full sampling case
            current_size = n_root
            for k in fanout:
                current_size *= k
                sizes.append(current_size)

        # Helper function to split tensors safely
        def _split(t: torch.Tensor | None) -> tuple:
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

        # 5. Build and return the final SeqGraph
        return Graph(
            root_nid=nids[0],
            root_eid=eids[0],
            root_ts=tss[0],
            root_nfeat=nfeats[0],
            root_mem=mems[0],
            root_mem_ts=mem_tss[0],
            root_box=boxs[0],
            root_box_ts=box_tss[0],
            neighbors=neighbors if neighbors else None,
            fanout=fanout,
        )


@dataclass(kw_only=True)
class NeighborOnHop:
    # For 1-hop neighbors, length is root_nid count * fanout[0]
    # For 2-hop neighbors, length is root_nid count * fanout[0] * fanout[1]
    nid: torch.Tensor
    eid: torch.Tensor
    ts: torch.Tensor
    nfeat: torch.Tensor | None = None
    efeat: torch.Tensor | None = None
    mem: torch.Tensor | None = None
    mem_ts: torch.Tensor | None = None
    box: torch.Tensor | None = None
    box_ts: torch.Tensor | None = None


@dataclass(kw_only=True)
class Graph:
    """Sequence sampling metadata."""

    # If nid is -1, then eid is also -1, indicating padding that should not be aggregated.

    root_nid: torch.Tensor  # Length is 3x number of src nodes (src, pos, neg)
    root_eid: torch.Tensor  # Length is 3x number of src nodes
    root_ts: torch.Tensor  # Length is 3x number of src nodes

    root_nfeat: torch.Tensor | None = None  # Length is 3x number of src nodes

    root_mem: torch.Tensor | None = None  # Length is root_nid count * fanout[0]
    root_mem_ts: torch.Tensor | None = None  # Length is root_nid count * fanout[0]
    root_box: torch.Tensor | None = None  # Length is root_nid count * fanout[0] * (box_size)
    root_box_ts: torch.Tensor | None = None  # Length is root_nid count * fanout[0]

    # Multi-hop neighbors. n_neighbor is a list like [10, 25].
    # Currently TGN only supports 1 hop, while TGAT supports multiple hops.
    neighbors: list[NeighborOnHop] | None = None

    # Sampling configuration like [10] or [10, 25]
    fanout: list[int] | None = None

    root_efeat: torch.Tensor | None = None  # Length is 1x number of src nodes
