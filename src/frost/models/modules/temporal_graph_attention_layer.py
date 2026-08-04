"""Temporal graph attention layer used by TGN and TGAT.

Provides ``TemporalGraphAttentionLayer``, a dense-tensor multi-head attention
module that attends over a node's sampled neighborhood while incorporating
temporal information via a time encoding.
"""

import logging

import torch

# worker/trainer.py
import torch.nn as nn
import torch.nn.functional as F

from frost.models.modules.time_encoder import TimeEncoder

logger = logging.getLogger(__name__)


class TemporalGraphAttentionLayer(nn.Module):
    """
    Pure PyTorch TransformerAttentionLayer for TGN.
    Uses dense tensor operations instead of DGL sparse message passing.
    """

    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int = 0,
        dim_out: int = 100,
        num_head: int = 2,
        dropout: float = 0.2,
        dim_neighbor: int | None = None,
    ) -> None:
        """
        Args:
            dim_node: Dimensionality of target node features.
            dim_edge: Dimensionality of edge features.
            dim_time: Dimensionality of the time encoding.  Set to 0 to
                disable time encoding.
            dim_out: Dimensionality of the attention output (must be divisible
                by ``num_head``).
            num_head: Number of attention heads.
            dropout: Dropout probability.
            dim_neighbor: Dimensionality of neighbor node features. If None,
                defaults to ``dim_node`` for backward compatibility.
        """
        super().__init__()
        self.num_head = num_head
        self.dim_out = dim_out
        self.dim_node = dim_node
        self.dim_neighbor = dim_neighbor if dim_neighbor is not None else dim_node

        assert dim_out % num_head == 0
        self.head_dim = dim_out // num_head

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.time_enc = TimeEncoder(dim_time) if dim_time > 0 else None

        # Q: Target Node Features + Time (0)
        self.w_q = nn.Linear(dim_node + dim_time, dim_out)
        # K, V: Neighbor Features + Edge Features + Time (dt)
        self.w_k = nn.Linear(self.dim_neighbor + dim_edge + dim_time, dim_out)
        self.w_v = nn.Linear(self.dim_neighbor + dim_edge + dim_time, dim_out)

        self.w_out = nn.Linear(dim_node + dim_out, dim_out)

    def forward(
        self,
        target_feat: torch.Tensor,  # Shape: [N, dim_node]
        neighbor_feat: torch.Tensor,  # Shape: [N, S, dim_node]
        edge_feat: torch.Tensor,  # Shape: [N, S, dim_edge]
        dt: torch.Tensor,  # Shape: [N, S]
        mask: torch.Tensor,  # Shape: [N, S] (True for valid, False for padding)
    ) -> torch.Tensor:
        """Compute temporally-aware attended node representations.

        Builds queries from target node features (+ time=0 encoding) and
        keys/values from neighbor node features, edge features, and
        time-delta encodings.  Pads invalid neighbors (``mask == False``)
        to ``-1e9`` before softmax so they contribute zero to the aggregate.

        Args:
            target_feat: Target node features of shape ``[N, dim_node]``.
            neighbor_feat: Neighbor node features of shape
                ``[N, S, dim_node]``.
            edge_feat: Edge feature tensor of shape ``[N, S, dim_edge]``.
            dt: Time deltas (target_ts - neighbor_ts) of shape ``[N, S]``.
            mask: Boolean validity mask of shape ``[N, S]`` where ``True``
                indicates a real neighbor and ``False`` indicates padding.

        Returns:
            Updated target node representations of shape ``[N, dim_out]``.
        """
        N, S = dt.shape

        # 1. Time Encoding
        if self.time_enc is not None:
            time_feat = self.time_enc(dt)  # [N, S, dim_time]
            zero_time_feat = self.time_enc(
                torch.zeros(N, device=dt.device, dtype=dt.dtype)
            )  # [N, dim_time]
        else:
            time_feat = None
            zero_time_feat = None

        # 2. Compute Q (Target nodes)
        q_inputs = [target_feat]
        if zero_time_feat is not None:
            q_inputs.append(zero_time_feat)

        Q = self.w_q(torch.cat(q_inputs, dim=-1))  # [N, dim_out]
        Q = Q.view(N, 1, self.num_head, self.head_dim)  # [N, 1, H, D_head]

        # 3. Compute K, V (Neighbor nodes + Edges)
        kv_inputs = [neighbor_feat]
        if edge_feat is not None:
            kv_inputs.append(edge_feat)
        if time_feat is not None:
            kv_inputs.append(time_feat)

        kv_input = torch.cat(
            kv_inputs, dim=-1
        )  # [N, S, dim_node + dim_edge + dim_time]

        K = self.w_k(kv_input).view(
            N, S, self.num_head, self.head_dim
        )  # [N, S, H, D_head]
        V = self.w_v(kv_input).view(
            N, S, self.num_head, self.head_dim
        )  # [N, S, H, D_head]

        # 4. Multi-head Attention
        # Q: [N, 1, H, D_head] * K: [N, S, H, D_head] -> sum over D_head -> [N, S, H]
        att_score = torch.sum(Q * K, dim=-1)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)

        # Apply padding mask: fill padded neighbor scores with -1e9 before softmax
        att_score = att_score.masked_fill(~mask.unsqueeze(-1), -1e9)

        att_weight = F.softmax(att_score, dim=1)  # Softmax over S dimension
        att_weight = self.dropout(att_weight)

        # 5. Aggregate V
        # att_weight: [N, S, H, 1] * V: [N, S, H, D_head] -> sum over S -> [N, H, D_head]
        V_weighted = att_weight.unsqueeze(-1) * V
        agg_h = torch.sum(V_weighted, dim=1)
        # Zero out nodes that have no neighbors at all
        all_pad = ~mask.any(dim=1)  # [N]
        # Keep this operation on the device.  Branching on ``all_pad.any()``
        # synchronizes the CUDA stream with Python once per forward pass.
        agg_h = agg_h.masked_fill(all_pad[:, None, None], 0.0)

        agg_h = agg_h.view(N, self.dim_out)  # [N, dim_out]

        # 6. Residual + Output Project.  Keep the aggregate first to match the
        # TGL/DGL implementation whose ``dstdata['h']`` precedes the target
        # node features in the output projection input.
        out = self.w_out(torch.cat([agg_h, target_feat], dim=1))
        out = F.relu(self.dropout(out))
        out = self.layer_norm(out)

        return out

    def forward_packed(
        self,
        target_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        edge_feat: torch.Tensor | None,
        dt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention only for semantic neighbor rows.

        The sampler emits a fixed ``[N, S]`` fanout layout.  Sparse datasets can
        leave many of those rows padded; this path packs valid rows before the
        time encoding and K/V projections, then performs a segmented softmax
        and scatter-add back to the root layout.
        """
        n_root, _ = dt.shape
        root_index, slot_index = mask.nonzero(as_tuple=True)

        if self.time_enc is not None:
            time_feat = self.time_enc(dt[root_index, slot_index])
            zero_time_feat = self.time_enc(
                torch.zeros(n_root, device=dt.device, dtype=dt.dtype)
            )
        else:
            time_feat = None
            zero_time_feat = None

        q_inputs = [target_feat]
        if zero_time_feat is not None:
            q_inputs.append(zero_time_feat)
        query = self.w_q(torch.cat(q_inputs, dim=-1)).view(
            n_root, self.num_head, self.head_dim
        )

        kv_inputs = [neighbor_feat[root_index, slot_index]]
        if edge_feat is not None:
            kv_inputs.append(edge_feat[root_index, slot_index])
        if time_feat is not None:
            kv_inputs.append(time_feat)
        kv_input = torch.cat(kv_inputs, dim=-1)
        key = self.w_k(kv_input).view(-1, self.num_head, self.head_dim)
        value = self.w_v(kv_input).view(-1, self.num_head, self.head_dim)

        score = F.leaky_relu(
            torch.sum(query[root_index] * key, dim=-1), negative_slope=0.2
        )
        segment_index = root_index[:, None].expand(-1, self.num_head)
        segment_max = torch.full(
            (n_root, self.num_head),
            -torch.inf,
            dtype=score.dtype,
            device=score.device,
        ).scatter_reduce(
            0,
            segment_index,
            score,
            reduce="amax",
            include_self=True,
        )
        exp_score = torch.exp(score - segment_max[root_index])
        denominator = torch.zeros_like(segment_max).scatter_add(
            0, segment_index, exp_score
        )
        attention = self.dropout(exp_score / denominator[root_index])
        weighted_value = value * attention.unsqueeze(-1)
        aggregate = torch.zeros(
            (n_root, self.num_head, self.head_dim),
            dtype=value.dtype,
            device=value.device,
        ).index_add(0, root_index, weighted_value)
        # Preserve zero-valued autograd edges for an entirely empty temporal
        # batch so DDP does not classify K/V parameters as unused.
        aggregate = aggregate + 0.0 * (
            self.w_k.weight.reshape(-1)[0] + self.w_v.weight.reshape(-1)[0]
        )
        aggregate = aggregate.view(n_root, self.dim_out)

        out = self.w_out(torch.cat([aggregate, target_feat], dim=1))
        out = F.relu(self.dropout(out))
        return self.layer_norm(out)
