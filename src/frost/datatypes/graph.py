from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class FlatGraph:
    """Sequence sampling metadata."""

    nid: torch.Tensor  # 数量是SeqGraph所有nid的长度
    eid: torch.Tensor  # 数量是SeqGraph所有nid的长度
    ts: torch.Tensor  # 数量是SeqGraph所有nid的长度
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

    def inflate(
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
            sampling_config=fanout,
        )


@dataclass(kw_only=True)
class NeighborOnHop:
    # 如果是第一层邻居, 长度是root_nid数量的n_neighbor_0倍
    # 如果是第二层邻居,长度是root_nid数量的n_neighbor_0倍的n_neighbor_1倍
    nid: torch.Tensor  #
    eid: torch.Tensor
    ts: torch.Tensor  #
    nfeat: torch.Tensor | None = None  #
    efeat: torch.Tensor | None = None  #
    mem: torch.Tensor | None = None  #
    mem_ts: torch.Tensor | None = None
    box: torch.Tensor | None = None
    box_ts: torch.Tensor | None = None


@dataclass(kw_only=True)
class Graph:
    """Sequence sampling metadata."""

    # 如果nid为-1,那么对应的eid肯定也为-1,表示这个位置是padding的,不需要参与聚合

    root_nid: torch.Tensor  # 长度是src数量3倍
    root_eid: torch.Tensor  # 长度是src数量3倍
    root_ts: torch.Tensor  # 长度是src数量3倍

    root_nfeat: torch.Tensor | None = None  # 长度是src数量3倍

    root_mem: torch.Tensor | None = None  # 长度是root_nid数量的n_neighbor倍_0倍
    root_mem_ts: torch.Tensor | None = None  # 长度是root_nid数量的n_neighbor倍_0倍
    root_box: torch.Tensor | None = None  # 长度是root_nid数量的n_neighbor倍_0倍_0倍
    root_box_ts: torch.Tensor | None = None  # 长度是root_nid数量的n_neighbor倍_0倍

    # 下面是1层邻居, n_neighbor是一个[10 25]这样的list,目前TGN不支持多层邻居,但是TGAT支持多层邻居
    neighbors: list[NeighborOnHop] | None = None

    # sampling_config类似于这样的[10] 或者[10,25]
    sampling_config: list

    root_efeat: torch.Tensor | None = None  # 长度是src数量1倍
