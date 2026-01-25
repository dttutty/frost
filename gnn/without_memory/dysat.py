import logging
from typing import List

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from dgl.heterograph import DGLBlock

from gnn.sample_mixin import SampleMixin

from ..components import EdgePredictor


logger = logging.getLogger(__name__)


class StructuralGATLayer(nn.Module):
    """
    Structural Attention Layer (Standard GAT).
    Operates on a single static snapshot graph.
    """

    def __init__(self, dim_in, dim_out, num_heads=2, dropout=0.1, att_dropout=0.1):
        super(StructuralGATLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads

        assert dim_out % num_heads == 0
        self.head_dim = dim_out // num_heads

        self.w = nn.Linear(dim_in, dim_out, bias=False)
        self.attn_vec = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.attn_vec)

    def forward(self, block):
        """
        b: DGL Block (Snapshot)
        """
        if block.num_edges() == 0:
            return torch.zeros(
                (block.num_dst_nodes(), self.dim_out), device=block.device
            )

        # 1. Linear Projection
        # src: [N_src, D_in] -> [N_src, H, D_head]
        h_src = self.w(block.srcdata["h"]).view(-1, self.num_heads, self.head_dim)

        # dst: [N_dst, D_in] -> [N_dst, H, D_head]
        h_dst = h_src[: block.num_dst_nodes()]  # Target nodes are the first N ones

        # 2. Attention Calculation
        # GAT logic: (h_src || h_dst) * att_vec
        # Here implemented as GATv2 style or standard GAT style using DGL ops

        # Prepare data for ops
        block.srcdata["z"] = h_src
        block.dstdata["z"] = h_dst

        # Compute e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # Using DGL's u_add_v trick to simulate concatenation for efficiency
        block.apply_edges(fn.u_add_v("z", "z", "e"))

        # e: [Edges, Heads, Head_Dim]
        e = self.leaky_relu(block.edata.pop("e"))

        # Dot product with attention vector -> [Edges, Heads, 1]
        # (E, H, D) * (1, H, D) -> sum dim 2 -> (E, H)
        score = (e * self.attn_vec).sum(dim=-1).unsqueeze(-1)

        # Softmax
        attn = dgl.ops.edge_softmax(block, score)
        attn = self.att_dropout(attn)

        # 3. Aggregation
        # V = h_src
        block.edata["a"] = attn
        block.update_all(fn.u_mul_e("z", "a", "m"), fn.sum("m", "ft"))

        rst = block.dstdata["ft"].reshape(-1, self.dim_out)
        rst = self.dropout(rst)
        return rst


class TemporalAttentionLayer(nn.Module):
    """
    Temporal Self-Attention Layer (Transformer).
    Aggregates representations across different time snapshots.
    """

    def __init__(self, dim_in, dim_out, num_heads=2, dropout=0.1, max_len=10):
        super(TemporalAttentionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads

        # Positional Encoding
        self.pe = nn.Parameter(torch.zeros(max_len, dim_in))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_in, 2).float() * (-np.log(10000.0) / dim_in)
        )
        self.pe.data[:, 0::2] = torch.sin(position * div_term)
        self.pe.data[:, 1::2] = torch.cos(position * div_term)

        # Transformer Encoder Layer
        # batch_first=True makes input [Batch, Seq, Feat]
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_in,
            nhead=num_heads,
            dim_feedforward=dim_in * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=1
        )

        self.w_out = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        """
        x: [Batch, History_Len, Dim]
        """
        B, T, D = x.shape

        # Add Positional Encoding
        # pe: [max_len, D] -> [T, D]
        pe = self.pe[:T, :].unsqueeze(0)  # [1, T, D]
        x = x + pe.to(x.device)

        # Self-Attention
        # Output: [Batch, T, D]
        # We assume causal masking isn't strictly necessary if 'x' represents a fixed past window,
        # but DySAT often uses a mask. For simplicity in batch processing, standard self-attention is used.
        out = self.transformer_encoder(x)

        # DySAT Strategy: typically we take the embedding at the *last* time step
        # as the representation for the link prediction at t+1.
        last_out = out[:, -1, :]  # [Batch, D]

        return self.w_out(last_out)


class DySAT(nn.Module, SampleMixin):
    """
    DySAT: Structural GAT + Temporal Self-Attention.
    """

    def __init__(
        self,
        dim_node,
        dim_edge,  # Not used in standard DySAT (structural only), kept for compatibility
        dim_hid=100,
        num_layers=2,
        heads=2,
        dropout=0.1,
        history_len=3,
    ):
        super(DySAT, self).__init__()
        self.num_layers = num_layers
        self.history_len = history_len
        self.dim_hid = dim_hid

        # --- Structural Attention Layers (GAT) ---
        # List of List: layers[layer_idx][snapshot_idx]
        # But actually, GAT weights are shared across time in DySAT?
        # Usually DySAT shares GNN weights across snapshots to capture common structural patterns.
        # So we just need `num_layers` GAT modules.

        self.structural_layers = nn.ModuleList()

        # Layer 1: Input -> Hidden
        self.structural_layers.append(
            StructuralGATLayer(dim_node, dim_hid, num_heads=heads, dropout=dropout)
        )

        # Layer 2..N: Hidden -> Hidden
        for _ in range(num_layers - 1):
            self.structural_layers.append(
                StructuralGATLayer(dim_hid, dim_hid, num_heads=heads, dropout=dropout)
            )

        # --- Temporal Attention Layer ---
        # Aggregates the history sequence into a single embedding
        self.temporal_layer = TemporalAttentionLayer(
            dim_in=dim_hid,
            dim_out=dim_hid,
            num_heads=heads,
            dropout=dropout,
            max_len=history_len + 5,  # Buffer
        )

        # --- Edge Predictor ---
        self.edge_predictor = EdgePredictor(dim_hid, neg_samples=1)

    def forward(
        self,
        mfgs: List[List[DGLBlock]],
    ):
        """
        mfgs: List of List of Blocks.
        Structure: mfgs[layer_idx][snapshot_idx]

        We need to run GNN on each snapshot independently, then aggregate over time.
        """

        # 1. Structural Attention (Spatial)
        # We iterate over history windows

        snapshot_embeddings = []

        # Assuming mfgs is organized as [layers][history]
        # But wait, your modules.py loops: for l in layers: for h in history.
        # Let's align with that data structure.

        # We need to collect the embeddings for all snapshots at the FINAL layer.

        # Let's verify input data structure assumption:
        # mfgs[0] is a list of blocks for the first GNN layer (across 'history' timestamps)
        # mfgs[0][0] = Block for t-2
        # mfgs[0][1] = Block for t-1
        # mfgs[0][2] = Block for t

        # Initialize input feature for all snapshots (Layer 0 input)
        # Typically sourced from mfgs[0][h].srcdata['nfeat']

        current_layer_embeddings = []  # [History] list

        # Load inputs
        for h in range(self.history_len):
            feat = mfgs[0][h].srcdata["nfeat"]
            current_layer_embeddings.append(feat)

        # Run GNN Layers
        for l in range(self.num_layers):
            next_layer_embeddings = []
            for h in range(self.history_len):
                block = mfgs[l][h]

                # Set input
                h_input = current_layer_embeddings[h]

                # Check mapping
                # Block src has N nodes, dst has M nodes. h_input is N.
                block.srcdata["h"] = h_input

                # Run GAT
                h_out = self.structural_layers[l](block)
                next_layer_embeddings.append(h_out)

            current_layer_embeddings = next_layer_embeddings

        # Now `current_layer_embeddings` contains the final GNN outputs for each snapshot.
        # Each element is [Batch, Dim_Hid]

        # 2. Temporal Attention
        # Stack: [Batch, History, Dim]
        temporal_input = torch.stack(current_layer_embeddings, dim=1)

        # Run Temporal Transformer
        final_embedding = self.temporal_layer(temporal_input)

        # 3. Prediction
        res_pos, res_neg = self.edge_predictor(final_embedding)

        return res_pos, res_neg
