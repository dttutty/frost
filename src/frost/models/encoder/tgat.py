"""TGAT (Temporal Graph Attention Network) encoder module.

Supports single- and multi-layer temporal attention by stacking
``TemporalGraphAttentionLayer`` blocks and aggregating neighborhood
information in a bottom-up fashion.
"""

import logging
from typing import cast

import torch
import torch.nn as nn

from frost.datatypes.graph import SeqGraph
from frost.models.modules.temporal_graph_attention_layer import (
    TemporalGraphAttentionLayer,
)

logger = logging.getLogger(__name__)


class TGAT(nn.Module):
    """
    TGAT Model (Multi-layer).
    """

    def __init__(
        self,
        ndim: int,  # from hydra config
        edim: int,  # from hydra config
        d_time: int,  # from hydra config
        d_hid: int,  # from hydra config
        fanout: list[int],  # from hydra config
        heads: int = 2,  # from hydra config
        dropout: float = 0.1,  # from hydra config
    ) -> None:
        super().__init__()
        self.num_layers = len(fanout)
        self.dim_hid = d_hid
        self.layers = nn.ModuleList()

        # --- Layer 1: Input (dim_node) -> Hidden (dim_hid) ---
        self.layers.append(
            TemporalGraphAttentionLayer(
                dim_node=ndim,
                dim_edge=edim,
                dim_time=d_time,
                dim_out=d_hid,
                num_head=heads,
                dropout=dropout,
            )
        )

        # --- Layer 2..N: Hidden (dim_hid) -> Hidden (dim_hid) ---
        for _ in range(self.num_layers - 1):
            self.layers.append(
                TemporalGraphAttentionLayer(
                    dim_node=d_hid,  # Input is previous layer's output
                    dim_edge=edim,
                    dim_time=d_time,
                    dim_out=d_hid,
                    num_head=heads,
                    dropout=dropout,
                )
            )

    def forward(self, graph: SeqGraph) -> torch.Tensor:
        """
        Forward pass using pure PyTorch SeqGraph structure.
        """
        assert graph.neighbors is not None, "Graph must have neighbors for TGAT."
        assert graph.fanout is not None, "TGAT requires fanout metadata."
        assert graph.root_nfeat is not None, "TGAT requires node features."
        assert len(graph.neighbors) >= self.num_layers, (
            f"Expected at least {self.num_layers} hops of neighbors, "
            f"but got {len(graph.neighbors)}."
        )

        # Extract root features
        h_root = graph.root_nfeat
        if graph.root_mem is not None:
            # Combine node features and memory if both exist
            h_root = h_root + graph.root_mem if h_root is not None else graph.root_mem

        # Fast path for single-layer TGAT
        if self.num_layers == 1:
            N = graph.root_nid.numel()
            S = graph.fanout[0]
            nbr = graph.neighbors[0]

            # Prepare neighbor features
            assert nbr.nfeat is not None, "TGAT requires neighbor node features."
            h_nbr = nbr.nfeat.view(N, S, -1)
            if nbr.mem is not None:
                h_nbr = h_nbr + nbr.mem.view(N, S, -1)

            # Prepare edge features and time deltas
            assert nbr.efeat is not None, "TGAT requires edge features."
            e_feat = nbr.efeat.view(N, S, -1)
            dt = graph.root_ts.unsqueeze(1) - nbr.ts.view(N, S)
            mask = nbr.nid.view(N, S) != -1

            # Compute attention
            out = self.layers[0](
                target_feat=h_root,
                neighbor_feat=h_nbr,
                edge_feat=e_feat,
                dt=dt,
                mask=mask,
            )
            return cast(torch.Tensor, out)

        # Multi-layer path (bottom-up aggregation)
        # We need to maintain a list of hidden states for each hop level
        hidden_states: list[torch.Tensor] = []

        # 1. Initialize features for all required hops
        # Hop 0 is root, Hop 1 is neighbors[0], Hop 2 is neighbors[1], etc.
        hidden_states.append(h_root)

        for i in range(self.num_layers):
            nbr = graph.neighbors[i]
            assert nbr.nfeat is not None, "TGAT requires neighbor node features."
            h_n = nbr.nfeat
            if nbr.mem is not None:
                h_n = h_n + nbr.mem if h_n is not None else nbr.mem
            hidden_states.append(h_n)

        # 2. Iterate through layers (bottom-up)
        for layer_idx in range(self.num_layers):
            # Target hop index decreases as we move up the network
            # For 2 layers:
            # Layer 0 updates Hop 1 (using Hop 2 neighbors), then updates Hop 0 (using Hop 1)
            hop_idx = self.num_layers - 1 - layer_idx

            # The current target nodes we are embedding
            target_feat = hidden_states[hop_idx]
            num_targets = target_feat.shape[0]

            # The neighbors of the current target nodes
            S = graph.fanout[hop_idx]
            nbr_level = graph.neighbors[hop_idx]
            assert nbr_level.efeat is not None, "TGAT requires edge features."

            # Reshape neighbor inputs to [num_targets, S, dim]
            neighbor_feat = hidden_states[hop_idx + 1].view(num_targets, S, -1)
            e_feat = nbr_level.efeat.view(num_targets, S, -1)

            # Get target timestamps to compute dt
            if hop_idx == 0:
                target_ts = graph.root_ts
            else:
                target_ts = graph.neighbors[hop_idx - 1].ts

            dt = target_ts.unsqueeze(1) - nbr_level.ts.view(num_targets, S)
            mask = nbr_level.nid.view(num_targets, S) != -1

            # Apply temporal attention layer
            updated_feat = self.layers[layer_idx](
                target_feat=target_feat,
                neighbor_feat=neighbor_feat,
                edge_feat=e_feat,
                dt=dt,
                mask=mask,
            )

            # Update the target representations for the next layer
            hidden_states[hop_idx] = updated_feat

        # The final representation of the root nodes is at index 0
        return cast(torch.Tensor, hidden_states[0])
