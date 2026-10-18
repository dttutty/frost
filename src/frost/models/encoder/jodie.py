"""JODIE encoder and time-projection layer."""

import math
from typing import cast

import torch
import torch.nn as nn

from frost.datatypes.graph import HierGraph
from frost.models.outputs import EncoderResult

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
        # Subtract integral timestamps before conversion so large absolute
        # timestamps do not lose their low-order bits in float32.  Compute the
        # ratio in float64, then hand the model a regular float32 feature.
        elapsed = ts - mem_ts
        time_diff = (elapsed.to(torch.float64) / (ts.to(torch.float64) + 1.0)).to(
            torch.float32
        )
        # Scaling the embedding: h * (1 + w * DyX_t)
        # h dim: [n_nodes, dim_mem]

        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst


class JODIE(nn.Module):
    def __init__(
        self,
        ndim: int,
        d_mem: int = 100,
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

    def forward(self, graph: HierGraph) -> EncoderResult:
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

        return EncoderResult(cast(torch.Tensor, h))
