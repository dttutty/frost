"""JODIE encoder module.

Contains the ``JODIE`` encoder, the ``JODIETimeEmbedding`` time-projection
layer used by JODIE, and ``IdentityNormLayer`` — a simple LayerNorm passthrough
used when no graph attention is needed.
"""

import logging
import math
from typing import cast

import torch
import torch.nn as nn

from frost.datatypes.graph import SeqGraph

logger = logging.getLogger(__name__)


# --- Helper Modules (Copied & Simplified from layers.py) ---


class IdentityNormLayer(nn.Module):
    """
    Applies LayerNorm to the target node features.
    Used as a passthrough layer when the neighborhood is empty.
    """

    # how to use:

    # # 1. Prepare Target Features (Roots)
    #     target_feat = seq_graph.root_mem
    #     if self.dim_node > 0 and seq_graph.root_nfeat is not None:
    #         target_feat = target_feat + self.n_feat_mapper(seq_graph.root_nfeat)

    #     # 2. Route based on neighborhood existence
    #     if not seq_graph.neighbors or seq_graph.sampling_config[0] == 0:
    #         # Fast path: No neighbors to attend to (e.g., APAN or 0-neighbor sampling)
    #         out = self.identity_norm(target_feat)
    #     else:
    #         # Standard path: Attend to neighbors
    #         nbr = seq_graph.neighbors[0]
    #         # ... (extract neighbor_feat, edge_feat, dt, mask as we discussed before) ...
    #         out = self.encoder(
    #             target_feat=target_feat,
    #             neighbor_feat=neighbor_feat,
    #             edge_feat=edge_feat,
    #             dt=dt,
    #             mask=mask,
    #         )

    #     return out

    def __init__(self, d_hid: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_hid)

    def forward(self, target_feat: torch.Tensor) -> torch.Tensor:
        """
        target_feat: Tensor of shape [N, d_hid] representing the root/target nodes.
        """
        return cast(torch.Tensor, self.norm(target_feat))


class JODIETimeEmbedding(nn.Module):
    """
    Specific to JODIE: Projects the static embedding based on time elapsed
    since the last update to predict the future state.
    """

    def __init__(self, dim_out: int) -> None:
        "dim_out = gnn dim_out = dim memory"
        super().__init__()
        self.time_emb = nn.Linear(1, dim_out)
        stdv = 1.0 / math.sqrt(self.time_emb.in_features)
        with torch.no_grad():
            self.time_emb.weight.normal_(0.0, stdv)
            self.time_emb.bias.normal_(0.0, stdv)

    def forward(
        self, h: torch.Tensor, mem_ts: torch.Tensor, ts: torch.Tensor
    ) -> torch.Tensor:
        """Project node embeddings forward in time using JODIE's time scaling.

        Computes ``h * (1 + W * ((ts - mem_ts) / (ts + 1)))``, where ``W``
        is a learnable linear projection of the normalised time delta.

        Args:
            h: Current node embedding of shape ``[N, d_mem]``.
            mem_ts: Timestamp of the last memory update, shape ``[N]``.
            ts: Current event timestamp, shape ``[N]``.

        Returns:
            Time-projected embedding of shape ``[N, d_mem]``.
        """
        # Time difference normalization as per JODIE paper/code
        time_diff = (ts - mem_ts) / (ts + 1)
        # Scaling the embedding: h * (1 + w * DyX_t)
        # h dim: [n_nodes, dim_mem]

        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst


class JODIE(nn.Module):
    def __init__(
        self,
        ndim: int,
        d_mem: int = 100,
        d_time: int = 100,
    ) -> None:
        """
        Args:
            dim_node: Dimension of raw node features (nfeat).
            dim_edge: Dimension of edge features (efeat).
            dim_memory: Dimension of memory / hidden state.
            dim_time: Dimension of time encoding.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dim_node = ndim

        self.node_feat_map = (
            nn.Linear(ndim, d_mem) if ndim > 0 and ndim != d_mem else None
        )

        # --- 2. GNN / Embedding Layer ---
        # JODIE.yml: arch='identity', time_transform='JODIE'
        self.layer_norm = nn.LayerNorm(d_mem)
        self.jodie_time_proj = JODIETimeEmbedding(dim_out=d_mem)

    def forward(self, graph: SeqGraph) -> torch.Tensor:
        """
        graph: Graph with root_mem, root_mem_ts, root_ts, root_nfeat populated.
        root_* tensors cover all root nodes: [src, pos_dst, neg_dst].
        """
        assert graph.root_mem is not None, "root_mem is missing in Graph"
        assert graph.root_mem_ts is not None, "root_mem_ts is missing in Graph"

        h: torch.Tensor = graph.root_mem

        if self.dim_node > 0 and graph.root_nfeat is not None:
            node_emb = (
                self.node_feat_map(graph.root_nfeat)
                if self.node_feat_map
                else graph.root_nfeat
            )
            h = h + node_emb

        h = self.layer_norm(h)
        h = self.jodie_time_proj(h, graph.root_mem_ts, graph.root_ts)

        return cast(torch.Tensor, h)
