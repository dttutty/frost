"""TGN (Temporal Graph Network) encoder module."""
# worker/trainer.py

import logging

import torch

from frost.datatypes.graph import SeqGraph
from frost.models.modules.temporal_graph_attention_layer import (
    TemporalGraphAttentionLayer,
)

logger = logging.getLogger(__name__)


class TGN(torch.nn.Module):
    """Temporal Graph Network encoder (1-hop).

    Combines pre-fetched node memory with optional static node features,
    then applies one layer of temporal graph attention over the 1-hop
    neighborhood to produce root-node embeddings.

    Args:
        ndim: Dimensionality of raw node features.  Set to 0 if no node
            features are available.
        edim: Dimensionality of edge features.
        d_mem: Dimensionality of node memory (must equal ``final_emb_dim``).
        d_time: Dimensionality of the time encoding.
        d_hid: Dimensionality of the attention output (hidden layer).
        heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        ndim: int,  # from hydra config
        edim: int,  # from hydra config
        d_mem: int,  # from hydra config
        d_time: int,  # from hydra config
        d_hid: int,  # from hydra config
        heads: int,  # from hydra config
        dropout: float,  # from hydra config
    ) -> None:
        super().__init__()
        self.dim_node = ndim
        self.dim_edge = edim
        self.dim_memory = d_mem

        self.n_feat_mapper: torch.nn.Linear | None = None
        if ndim > 0 and ndim != d_mem:
            self.n_feat_mapper = torch.nn.Linear(ndim, d_mem)

        assert d_time >= 0, "must have time encoding if memory is used"

        self.encoder = TemporalGraphAttentionLayer(
            dim_node=d_mem,
            dim_edge=edim,
            dim_time=d_time,
            dim_out=d_hid,
            num_head=heads,
            dropout=dropout,
        )

    def forward(
        self,
        graph: SeqGraph,
    ) -> torch.Tensor:
        """Compute root-node embeddings via 1-hop temporal graph attention.

        Args:
            graph: Inflated graph with ``root_mem``, ``root_nfeat``,
                ``root_ts``, ``root_nid``, and ``neighbors[0]`` populated.

        Returns:
            Embedding tensor of shape ``[n_root, d_hid]``.
        """
        assert graph.neighbors is not None, "TGN requires neighbors for memory update"
        assert graph.fanout is not None, "TGN requires fanout metadata"
        assert graph.root_mem is not None, "TGN requires root memory"
        # 1. Prepare Target Features (Roots)
        root_mem = graph.root_mem
        if self.dim_node > 0 and graph.root_nfeat is not None:
            root_nfeat_emb = (
                self.n_feat_mapper(graph.root_nfeat)
                if self.n_feat_mapper
                else graph.root_nfeat
            )
            target_feat = root_mem + root_nfeat_emb
        else:
            target_feat = root_mem

        # 2. Prepare Neighbor Features (Hop 0)
        # Note: We assume 1-hop TGN here.
        nbr_data = graph.neighbors[0]
        assert nbr_data.mem is not None, "TGN requires neighbor memory"
        nbr_mem = nbr_data.mem
        if self.dim_node > 0 and nbr_data.nfeat is not None:
            nbr_nfeat_emb = (
                self.n_feat_mapper(nbr_data.nfeat)
                if self.n_feat_mapper
                else nbr_data.nfeat
            )
            neighbor_combined_feat = nbr_mem + nbr_nfeat_emb
        else:
            neighbor_combined_feat = nbr_mem

        # 3. Reshape inputs to dense tensors [N, S, D] for Attention Layer
        N = graph.root_nid.numel()
        S = graph.fanout[0]

        neighbor_feat = neighbor_combined_feat.view(N, S, -1)
        edge_feat = (
            nbr_data.efeat.view(N, S, -1) if nbr_data.efeat is not None else None
        )

        # 4. Compute delta time (dt)
        neighbor_ts = nbr_data.ts.view(N, S)
        dt = graph.root_ts.unsqueeze(1) - neighbor_ts

        # 5. Create boolean mask (True for valid edges, False for padding)
        neighbor_nid = nbr_data.nid.view(N, S)
        mask = neighbor_nid != -1

        # 6. Pass into the attention layer
        out = self.encoder(
            target_feat=target_feat,
            neighbor_feat=neighbor_feat,
            edge_feat=edge_feat,
            dt=dt,
            mask=mask,
        )

        return out
