import logging

import torch

# worker/trainer.py
import torch.nn as nn
import torch.nn.functional as F

from frost.datatypes.graph import SeqGraph

from .time_encoder import TimeEncoder

logger = logging.getLogger(__name__)


def process_seq_graph(
    seq_graph: SeqGraph, attn_layer: "TransformerAttentionLayer"
) -> torch.Tensor:
    """
    Example of unpacking SeqGraph for the first hop.
    """
    assert seq_graph.fanout is not None
    assert seq_graph.neighbors is not None
    assert seq_graph.root_mem is not None
    N = seq_graph.root_nid.numel()
    S = seq_graph.fanout[0]
    nbr = seq_graph.neighbors[0]

    # 1. Target node features (Roots)
    # Target memory + static features
    target_feat = seq_graph.root_mem
    if seq_graph.root_nfeat is not None:
        target_feat = target_feat + seq_graph.root_nfeat

    # 2. Neighbor node features (Hop 0)
    # Reshape from flat [N * S, D] to dense [N, S, D]
    assert nbr.mem is not None
    nbr_mem_dense = nbr.mem.view(N, S, -1)
    if nbr.nfeat is not None:
        nbr_feat_dense = nbr.nfeat.view(N, S, -1)
        neighbor_feat = nbr_mem_dense + nbr_feat_dense
    else:
        neighbor_feat = nbr_mem_dense

    # 3. Edge features
    assert nbr.efeat is not None
    edge_feat = nbr.efeat.view(N, S, -1)

    # 4. Time delta
    # root_ts: [N, 1], nbr.ts: [N, S]
    dt = seq_graph.root_ts.unsqueeze(1) - nbr.ts.view(N, S)

    # 5. Padding mask
    # True for valid nodes, False for padding (-1)
    mask = nbr.nid.view(N, S) != -1

    # 6. Forward pass
    out = attn_layer(
        target_feat=target_feat,
        neighbor_feat=neighbor_feat,
        edge_feat=edge_feat,
        dt=dt,
        mask=mask,
    )

    return out


class TransformerAttentionLayer(nn.Module):
    """
    Pure PyTorch TransformerAttentionLayer for dense tensor operations.
    Removes DGL dependencies and uses masking for padded neighbors.
    """

    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int,
        att_h: int,
        dropout: float,
        dim_out: int,
        combined: bool = False,
    ) -> None:
        super().__init__()
        self.num_head = att_h
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.combined = combined

        self.dropout = nn.Dropout(dropout)
        self.att_act = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(dim_out)

        if dim_time > 0:
            self.time_enc = TimeEncoder(dim_time)

        # Initialize linear layers based on 'combined' mode
        if combined:
            if dim_node > 0:
                self.w_q_n = nn.Linear(dim_node, dim_out)
                self.w_k_n = nn.Linear(dim_node, dim_out)
                self.w_v_n = nn.Linear(dim_node, dim_out)
            if dim_edge > 0:
                self.w_k_e = nn.Linear(dim_edge, dim_out)
                self.w_v_e = nn.Linear(dim_edge, dim_out)
            if dim_time > 0:
                self.w_q_t = nn.Linear(dim_time, dim_out)
                self.w_k_t = nn.Linear(dim_time, dim_out)
                self.w_v_t = nn.Linear(dim_time, dim_out)
        else:
            if dim_node + dim_time > 0:
                self.w_q = nn.Linear(dim_node + dim_time, dim_out)
            self.w_k = nn.Linear(dim_node + dim_edge + dim_time, dim_out)
            self.w_v = nn.Linear(dim_node + dim_edge + dim_time, dim_out)

        self.w_out = nn.Linear(dim_node + dim_out, dim_out)

    def forward(
        self,
        target_feat: torch.Tensor,  # [N, dim_node]
        neighbor_feat: torch.Tensor,  # [N, S, dim_node]
        edge_feat: torch.Tensor,  # [N, S, dim_edge]
        dt: torch.Tensor,  # [N, S]
        mask: torch.Tensor,  # [N, S] (True for valid, False for padding)
    ) -> torch.Tensor:
        N, S = dt.shape
        dev = target_feat.device

        # Fast path for empty neighborhoods
        if S == 0:
            return torch.zeros((N, self.dim_out), device=dev)

        # 1. Time Encoding
        if self.dim_time > 0:
            time_feat = self.time_enc(dt)  # [N, S, dim_time]
            zero_time_feat = self.time_enc(
                torch.zeros(N, dtype=torch.float32, device=dev)
            )
        else:
            time_feat = None
            zero_time_feat = None

        # 2. Compute Q, K, V
        if self.combined:
            Q = torch.zeros((N, self.dim_out), device=dev)
            K = torch.zeros((N, S, self.dim_out), device=dev)
            V = torch.zeros((N, S, self.dim_out), device=dev)

            if self.dim_node > 0:
                Q += self.w_q_n(target_feat)
                K += self.w_k_n(neighbor_feat)
                V += self.w_v_n(neighbor_feat)
            if self.dim_edge > 0:
                K += self.w_k_e(edge_feat)
                V += self.w_v_e(edge_feat)
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
        else:
            # Q projection
            q_inputs = []
            if self.dim_node > 0:
                q_inputs.append(target_feat)
            if self.dim_time > 0:
                q_inputs.append(zero_time_feat)
            Q = (
                self.w_q(torch.cat(q_inputs, dim=1))
                if q_inputs
                else torch.ones((N, self.dim_out), device=dev)
            )

            # K and V projection
            kv_inputs = []
            if self.dim_node > 0:
                kv_inputs.append(neighbor_feat)
            if self.dim_edge > 0:
                kv_inputs.append(edge_feat)
            if self.dim_time > 0:
                kv_inputs.append(time_feat)
            kv_input = torch.cat(kv_inputs, dim=2)
            K = self.w_k(kv_input)
            V = self.w_v(kv_input)

        # 3. Multi-head Attention
        # Reshape for heads: Q -> [N, 1, H, D], K/V -> [N, S, H, D]
        head_dim = self.dim_out // self.num_head
        Q = Q.view(N, 1, self.num_head, head_dim)
        K = K.view(N, S, self.num_head, head_dim)
        V = V.view(N, S, self.num_head, head_dim)

        # Score: [N, S, H]
        att_score = self.att_act(torch.sum(Q * K, dim=-1))

        # Apply mask: fill padded neighbor scores with -1e9 before softmax
        att_score = att_score.masked_fill(~mask.unsqueeze(-1), -1e9)

        att_weights = F.softmax(att_score, dim=1)

        # Zero out weights for padded elements to prevent NaN accumulation
        att_weights = att_weights.masked_fill(~mask.unsqueeze(-1), 0.0)
        att_weights = self.dropout(att_weights)

        # 4. Aggregate V
        # att_weights: [N, S, H, 1] * V: [N, S, H, D] -> sum over S -> [N, H, D]
        agg_v = torch.sum(att_weights.unsqueeze(-1) * V, dim=1)
        agg_v = agg_v.view(N, self.dim_out)

        # 5. Output Projection
        if self.dim_node != 0:
            rst = torch.cat([agg_v, target_feat], dim=1)
        else:
            rst = agg_v

        rst = self.w_out(rst)
        rst = F.relu(self.dropout(rst))
        return self.layer_norm(rst)
