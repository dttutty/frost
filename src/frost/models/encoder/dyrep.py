"""DyRep encoder module.

DyRep treats the updated node memory directly as the node embedding.  Because
FROST's ``AssembledModel`` already runs the memory updater before the encoder,
``graph.root_mem`` already contains the updated memory when this encoder is
called.  The encoder simply normalizes the memory and applies an optional
output projection.
"""

import logging
from typing import cast

import torch
import torch.nn as nn

from frost.datatypes.graph import SeqGraph

logger = logging.getLogger(__name__)


class DyRep(nn.Module):
    """
    DyRep encoder adapted for the FROST framework.

    In the reference implementation, DyRep's final node embedding is simply
    the raw updated memory:
        src_emb = updated_node_memories[src_ids]
        dst_emb = updated_node_memories[dst_ids]

    FROST's AssembledModel already runs the memory updater (RNN cell) before
    the encoder, so `graph.root_mem` already contains the updated memory when
    this encoder is called.

    Architecture:
        h = root_mem  (+  Linear(root_nfeat) if node features exist)
        h = LayerNorm(h)
        h = Dropout(h)
        h = Linear(h, d_out)  [only if d_mem != d_out]

    Key differences from JODIE encoder:
        - No time projection  ( JODIE:  h = h * (1 + W * delta_t) )
        - Simpler: essentially memory → output projection

    Key differences from TGN encoder:
        - No graph attention on neighbors
        - No neighbor sampling needed  (use ngh_sampler=0_hop)

    Note on mail approximation
    --------------------------
    The original DyRep paper builds messages as
        [self_mem, dst_graph_attn_emb, delta_t_enc, edge_feat]
    where  dst_graph_attn_emb  is computed by a graph attention module.
    FROST's mail pipeline (interaction_ops.generate_interaction_mails) uses
        [self_mem, other_mem, edge_feat]
    which is a standard TGN/JODIE-style message.  Changing this would require
    modifying the assembled model timing (encoder must run before box_task is
    queued).  The current implementation is therefore an approximation: the
    message uses raw memory rather than graph-attention memory.  The memory
    cell (RNN vs GRU) and output semantics (memory as embedding) are exact.
    """

    def __init__(
        self,
        ndim: int,  # node feature dim  (${ndim})
        d_mem: int,  # memory dim        (${final_emb_dim}, must match mem_updater.d_mem)
        d_out: int,  # output dim        (${final_emb_dim})
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            ndim: Dimensionality of raw node features.  Set to 0 if no node
                features are available.
            d_mem: Dimensionality of the node memory (must match the memory
                updater's output dim).
            d_out: Dimensionality of the encoder output.
            dropout: Dropout probability applied after LayerNorm.
        """
        super().__init__()
        self.dim_node = ndim

        # Optional projection if node features exist and have a different dim
        self.node_feat_map: nn.Linear | None = None
        if ndim > 0 and ndim != d_mem:
            self.node_feat_map = nn.Linear(ndim, d_mem)

        self.layer_norm = nn.LayerNorm(d_mem)
        self.dropout = nn.Dropout(dropout)

        # Output projection (identity when d_mem == d_out)
        self.out_proj: nn.Linear | None = None
        if d_mem != d_out:
            self.out_proj = nn.Linear(d_mem, d_out)

    def forward(self, graph: SeqGraph) -> torch.Tensor:
        """
        graph.root_mem  : [n_root, d_mem]   updated memory from RNN updater
        graph.root_nfeat: [n_root, ndim]    raw node features (optional)

        Returns h : [n_root, d_out]
        """
        assert graph.root_mem is not None, "DyRep requires root_mem (use a mem_updater)"

        h: torch.Tensor = graph.root_mem

        if self.dim_node > 0 and graph.root_nfeat is not None:
            nfeat_proj = (
                self.node_feat_map(graph.root_nfeat)
                if self.node_feat_map is not None
                else graph.root_nfeat
            )
            h = h + nfeat_proj

        h = self.layer_norm(h)
        h = self.dropout(h)

        if self.out_proj is not None:
            h = self.out_proj(h)

        return cast(torch.Tensor, h)
