"""DyRep encoder module.

This implementation keeps DyRep's RNN-updated memory state, then applies a
lightweight 1-hop structural attention block so the representation and the next
mail message can depend on the temporal neighborhood.
"""

import logging

import torch
import torch.nn as nn

from frost.datatypes.graph import HierGraph
from frost.models.modules.temporal_graph_attention_layer import (
    TemporalGraphAttentionLayer,
)

logger = logging.getLogger(__name__)


DyRepOutput = tuple[torch.Tensor, torch.Tensor | None]


class DyRep(nn.Module):
    """
    DyRep encoder adapted for the FROST framework.

    FROST's AssembledModel already runs the memory updater before the encoder,
    so ``graph.root_mem`` contains the latest RNN memory.  The DyGLib-style path
    uses a 1-hop temporal graph-attention embedding for both the current root
    embedding and the peer embedding written into the next mailbox message.

    Base memory representation:
        h = root_mem  (+  Linear(root_nfeat) if node features exist)

    The structural path feeds this representation directly into temporal
    attention, matching TGL's DyRep path.  The attention block already owns
    the output dropout and LayerNorm.  Zero-hop mode retains the historical
    IdentityNorm-style LayerNorm and dropout.

    Structural representation:
        h_struct = TemporalGraphAttention(h, neighbor_mem + neighbor_nfeat, dt)
        h = Linear(h, d_out)  [only if d_mem != d_out]

    The returned tuple is ``(h, mail_peer_mem)``.  ``mail_peer_mem`` stays in
    ``d_mem`` space so the mailbox width remains ``2 * d_mem + edim``.
    """

    def __init__(
        self,
        ndim: int,  # node feature dim  (${ndim})
        d_mem: int,  # memory dim        (${encoder.out_dim}, must match mem_updater.d_mem)
        d_out: int,  # output dim        (${encoder.out_dim})
        edim: int = 0,  # edge feature dim  (${edim})
        d_time: int = 0,
        heads: int = 2,
        dropout: float = 0.1,
        use_structural_attention: bool = True,
    ) -> None:
        """
        Args:
            ndim: Dimensionality of raw node features.  Set to 0 if no node
                features are available.
            edim: Dimensionality of raw edge features.
            d_mem: Dimensionality of the node memory (must match the memory
                updater's output dim).
            d_out: Dimensionality of the encoder output.
            d_time: Dimensionality of temporal encodings used by structural
                attention.
            heads: Number of attention heads used by structural attention.
            dropout: Dropout probability applied after LayerNorm.
            use_structural_attention: If ``False``, keep the historical 0-hop
                memory-only behavior and return no structural mail embedding.
        """
        super().__init__()
        self.dim_node = ndim
        self.dim_edge = edim
        self.dim_memory = d_mem
        self.use_structural_attention = use_structural_attention

        # Optional projection if node features exist and have a different dim
        self.node_feat_map: nn.Linear | None = None
        if ndim > 0 and ndim != d_mem:
            self.node_feat_map = nn.Linear(ndim, d_mem)

        self.layer_norm = (
            None if use_structural_attention else nn.LayerNorm(d_mem)
        )
        self.dropout = nn.Dropout(dropout)

        # Output projection (identity when d_mem == d_out)
        self.out_proj: nn.Linear | None = None
        if d_mem != d_out:
            self.out_proj = nn.Linear(d_mem, d_out)

        self.structural_encoder: TemporalGraphAttentionLayer | None = None
        if use_structural_attention:
            self.structural_encoder = TemporalGraphAttentionLayer(
                dim_node=d_mem,
                dim_edge=edim,
                dim_time=d_time,
                dim_out=d_mem,
                num_head=heads,
                dropout=dropout,
            )

    def _memory_feature(
        self, mem: torch.Tensor, nfeat: torch.Tensor | None
    ) -> torch.Tensor:
        h = mem
        if self.dim_node > 0 and nfeat is not None:
            nfeat_proj = self.node_feat_map(nfeat) if self.node_feat_map else nfeat
            h = h + nfeat_proj
        if self.layer_norm is not None:
            h = self.dropout(self.layer_norm(h))
        return h

    def _structural_embedding(
        self,
        graph: HierGraph,
        root_feat: torch.Tensor,
    ) -> torch.Tensor:
        assert self.structural_encoder is not None
        assert graph.neighbors is not None, (
            "DyRep structural attention requires 1-hop neighbors; set "
            "encoder.use_structural_attention=false for 0-hop DyRep."
        )
        assert graph.fanout is not None and len(graph.fanout) > 0, (
            "DyRep structural attention requires non-empty fanout."
        )

        nbr_data = graph.neighbors[0]
        assert nbr_data.mem is not None, (
            "DyRep structural attention requires neighbor memory"
        )
        if self.dim_edge > 0:
            assert nbr_data.efeat is not None, (
                "DyRep structural attention requires neighbor edge features when edim > 0"
            )

        neighbor_feat = self._memory_feature(nbr_data.mem, nbr_data.nfeat)
        n_root = graph.root_nid.numel()
        fanout = graph.fanout[0]

        dense_neighbor_feat = neighbor_feat.view(n_root, fanout, -1)
        edge_feat = (
            nbr_data.efeat.view(n_root, fanout, -1)
            if nbr_data.efeat is not None
            else None
        )
        neighbor_ts = nbr_data.ts.view(n_root, fanout)
        neighbor_nid = nbr_data.nid.view(n_root, fanout)
        dt = graph.root_ts.unsqueeze(1) - neighbor_ts
        mask = neighbor_nid != -1

        return self.structural_encoder(
            target_feat=root_feat,
            neighbor_feat=dense_neighbor_feat,
            edge_feat=edge_feat,
            dt=dt,
            mask=mask,
        )

    def forward(self, graph: HierGraph) -> DyRepOutput:
        """
        graph.root_mem  : [n_root, d_mem]   updated memory from RNN updater
        graph.root_nfeat: [n_root, ndim]    raw node features (optional)

        Returns:
            h: [n_root, d_out]
            mail_peer_mem: [n_root, d_mem] structural peer embedding for
                next-batch mailbox generation, or ``None`` in 0-hop mode.
        """
        assert graph.root_mem is not None, "DyRep requires root_mem (use a mem_updater)"

        h_mem = self._memory_feature(graph.root_mem, graph.root_nfeat)
        mail_peer_mem: torch.Tensor | None = None
        if self.structural_encoder is not None:
            h_mem = self._structural_embedding(graph, h_mem)
            mail_peer_mem = h_mem

        h = h_mem
        if self.out_proj is not None:
            h = self.out_proj(h)

        return h, mail_peer_mem
