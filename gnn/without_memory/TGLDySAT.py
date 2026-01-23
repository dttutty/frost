import logging

import torch
import torch.nn as nn

from gnn.components import EdgePredictor, TemporalGraphAttentionLayer
from gnn.sample_mixin import SampleMixin


logger = logging.getLogger(__name__)


class DySAT_TGL(nn.Module, SampleMixin):
    """
    GCRN: Graph Convolutional Recurrent Network.
    Structure: [GAT -> GAT] (per snapshot) -> RNN (aggregate history)
    Faithful to DySAT.yml config.
    """

    def __init__(
        self,
        dim_node,
        dim_edge,
        dim_time=0,
        dim_hid=100,
        num_layers=2,
        heads=2,
        dropout=0.1,
        history_len=3,
    ):
        super(DySAT_TGL, self).__init__()
        self.num_layers = num_layers
        self.history_len = history_len
        self.dim_hid = dim_hid
        assert self.history_len >= 1  # at least 1 snapshot

        # --- 1. Structural Layers (GAT) ---
        # Shared weights across all timestamps (Standard GCRN practice)
        self.structural_layers = nn.ModuleList()

        # Layer 1: Input (dim_node) -> Hidden
        self.structural_layers.append(
            TemporalGraphAttentionLayer(
                dim_node, dim_edge, dim_time, dim_hid, num_head=heads, dropout=dropout
            )
        )

        # Layer 2..N: Hidden -> Hidden
        for _ in range(num_layers - 1):
            self.structural_layers.append(
                TemporalGraphAttentionLayer(
                    dim_hid,
                    dim_edge,
                    dim_time,
                    dim_hid,
                    num_head=heads,
                    dropout=dropout,
                )
            )

        # --- 2. Temporal Combiner (RNN) ---
        # YAML: combine: 'rnn'
        # Input: Sequence of GNN embeddings [Batch, History, Dim]
        # Output: Final hidden state
        self.rnn = nn.RNN(input_size=dim_hid, hidden_size=dim_hid, batch_first=True)

        # --- 3. Predictor ---
        self.edge_predictor = EdgePredictor(dim_hid)

    def forward(self, mfgs, neg_samples=1):
        """
        mfgs: List of List of Blocks.
        Structure: mfgs[layer_idx][history_idx]
        """

        # To store the final GNN output for each snapshot
        snapshot_embeddings = []

        # Loop over history windows (Time Steps)
        for h in range(self.history_len):
            # --- GNN Pass for Snapshot h ---
            # Initial Feature
            # Assuming mfgs[0][h] contains the input features for this snapshot
            x = mfgs[0][h].srcdata["nfeat"]

            for l in range(self.num_layers):
                block = mfgs[l][h]

                # Prepare Block Inputs
                # srcdata['h'] comes from previous layer (or raw feat)
                block.srcdata["h"] = x
                # dstdata['h'] is sliced from src
                block.dstdata["h"] = x[: block.num_dst_nodes()]  # this may be redundant

                # Run GAT
                x = self.structural_layers[l](block)

            # After passing all GNN layers, x is the embedding for snapshot h
            # x shape: [Batch_Size, Dim_Hid]
            snapshot_embeddings.append(x)

        # --- Temporal Aggregation (RNN) ---
        # Stack snapshots: [Batch, History, Dim]
        seq_input = torch.stack(snapshot_embeddings, dim=1)

        # Run RNN
        # out: [Batch, History, Dim], hn: [1, Batch, Dim]
        out, _ = self.rnn(seq_input)

        # Take the output of the last time step
        final_embedding = out[:, -1, :]

        # --- Prediction ---

        return final_embedding
