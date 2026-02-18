import logging
from typing import cast

import dgl
import dgl.function as fn
import numpy as np
import torch

# worker/trainer.py
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock


logger = logging.getLogger(__name__)


class TimeEncode(nn.Module):
    """
    Time Encoding as described in TGN/JODIE.
    Formula: cos(w * t + b) where w is determined by 1 / 10^linspace(0, 9, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        # TGN original implementation uses numpy linspace
        seq = np.linspace(0, 9, dim, dtype=np.float32)
        # Initialize weights to match TGN exactly
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10**seq)).reshape(dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [batch_size] -> [batch_size, 1]
        output = torch.cos(self.w(t.reshape((-1, 1)).float()))
        return output


class EdgePredictor(nn.Module):
    """
    Predicts probability of an edge existing.
    Logic: MLP(h_src + h_dst) -> score
    """

    def __init__(self, dim_in: int, neg_samples: int = 1) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.neg_samples = neg_samples
        self.src_fc = nn.Linear(dim_in, dim_in)
        self.dst_fc = nn.Linear(dim_in, dim_in)
        self.out_fc = nn.Linear(dim_in, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h contains: [True Edges Src, True Edges Dst, Neg Edges Dst]
        num_edge = h.shape[0] // (self.neg_samples + 2)

        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge : 2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge :])

        h_pos_edge = F.relu(h_src + h_pos_dst)
        # Tile src for negative samples: [src1, src2] -> [src1, src1, src2, src2] if neg=2
        h_neg_edge = F.relu(h_src.tile(self.neg_samples, 1) + h_neg_dst)

        ret_1 = self.out_fc(h_pos_edge)
        ret_2 = self.out_fc(h_neg_edge)

        return torch.cat([ret_1, ret_2])


class TransfomerAttentionLayer(torch.nn.Module):
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

        self.dropout = torch.nn.Dropout(dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)

        if combined:
            if dim_node > 0:
                self.w_q_n = torch.nn.Linear(dim_node, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node, dim_out)
            if dim_edge > 0:
                self.w_k_e = torch.nn.Linear(dim_edge, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node + dim_edge + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node + dim_edge + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)

    def forward(self, b: DGLBlock) -> torch.Tensor:
        dev = b.srcdata["h"].device if "h" in b.srcdata else b.device

        assert self.dim_time + self.dim_node + self.dim_edge > 0
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=dev)
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata["dt"])
            zero_time_feat = self.time_enc(
                torch.zeros(
                    b.num_dst_nodes(),
                    dtype=torch.float32,
                    device=dev,
                )
            )
        dst_h = b.srcdata["h"][: b.num_dst_nodes()]  # DST位于srcdata['h']的前半部分
        src_h = b.srcdata["h"][b.num_dst_nodes() :]  # SRC位于后半部分
        dst_indices = b.edges()[1]
        src_indices = b.edges()[0] - b.num_dst_nodes()
        # e.g.: 0, 1, 2, 3, 4
        # dst: 0, 1
        # src: 2, 3, 4
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=dev)
            K = torch.zeros((b.num_edges(), self.dim_out), device=dev)
            V = torch.zeros((b.num_edges(), self.dim_out), device=dev)
            if self.dim_node > 0:
                Q += self.w_q_n(dst_h)[dst_indices]
                K += self.w_k_n(src_h)[src_indices]
                V += self.w_v_n(src_h)[src_indices]
            if self.dim_edge > 0:
                K += self.w_k_e(b.edata["efeat"])
                V += self.w_v_e(b.edata["efeat"])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[dst_indices]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
        else:
            q_input = []
            kv_input = []
            if self.dim_node > 0:
                # if the src_h size_0 not equal to edge number, it will cause error
                q_input.append(dst_h)
                kv_input.append(src_h)
            if self.dim_edge > 0:
                kv_input.append(b.edata["feat"])
            if self.dim_time > 0:
                q_input.append(zero_time_feat)
                kv_input.append(time_feat)
            Q = (
                self.w_q(torch.cat(q_input, dim=1))[dst_indices]
                if len(q_input) > 0
                else torch.ones((b.num_edges(), self.dim_out), device=dev)
            )
            K = self.w_k(torch.cat(kv_input, dim=1))
            V = self.w_v(torch.cat(kv_input, dim=1))
        Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
        K = torch.reshape(K, (K.shape[0], self.num_head, -1))
        V = torch.reshape(V, (V.shape[0], self.num_head, -1))
        att = dgl.ops.edge_softmax(
            b, F.leaky_relu(torch.sum(Q * K, dim=2), negative_slope=0.2)
        )
        att = self.dropout(att)
        V = torch.reshape(V * att[:, :, None], (V.shape[0], -1))
        if self.combined:
            b.edata["v"] = V
            b.update_all(dgl.function.copy_e("v", "m"), dgl.function.sum("m", "h"))
        else:
            dst_dummy = torch.zeros((b.num_dst_nodes(), V.shape[1]), device=dev)
            b.srcdata["v"] = torch.cat([dst_dummy, V], dim=0)
            b.update_all(dgl.function.copy_u("v", "m"), dgl.function.sum("m", "h"))

        rst: torch.Tensor
        if self.dim_node != 0:
            dst_h_from_graph = cast(torch.Tensor, b.dstdata["h"])
            rst = cast(torch.Tensor, torch.cat([dst_h_from_graph, dst_h], dim=1))
        else:
            rst = cast(torch.Tensor, b.dstdata["h"])
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        return cast(torch.Tensor, self.layer_norm(rst))


class NodeClassificationHeader(torch.nn.Module):
    def __init__(self, dim_in: int, d_hid: int, num_class: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim_in, d_hid)
        self.fc2 = torch.nn.Linear(d_hid, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return cast(torch.Tensor, x)


def _cat_optional(
    x: torch.Tensor,
    *extras: torch.Tensor | None,
    dim: int = 1,
) -> torch.Tensor:
    parts = [x]
    for t in extras:
        if t is not None:
            parts.append(t)
    return torch.cat(parts, dim=dim) if len(parts) > 1 else x


class TemporalGraphAttentionLayer(nn.Module):
    """
    Simplified TransfomerAttentionLayer specifically for TGN.
    Removes 'combined' logic and other general GNN variants.
    """

    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int = 0,
        dim_out: int = 100,
        num_head: int = 2,
        dropout: float = 0.2,
        ffn_expansion: int = 4,  # 新增: FFN 扩展倍数，默认 4 倍
    ) -> None:
        super().__init__()
        self.num_head = num_head
        self.dim_out = dim_out

        assert dim_out % num_head == 0
        self.head_dim = dim_out // num_head

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.time_enc: TimeEncode | None = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)

        # Q: Source Node Features + Time (0)
        self.w_q = nn.Linear(dim_node + dim_time, dim_out)
        # K, V: Neighbor Features + Edge Features + Time (dt)
        self.w_k = nn.Linear(dim_node + dim_edge + dim_time, dim_out)
        self.w_v = nn.Linear(dim_node + dim_edge + dim_time, dim_out)

        self.w_out = nn.Linear(dim_node + dim_out, dim_out)

        # self.ffn0 = nn.Sequential(
        #     nn.Linear(dim_out, dim_out * ffn_expansion),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_out * ffn_expansion, dim_out),
        #     nn.Dropout(dropout),
        # )

    def forward(self, b: DGLBlock) -> torch.Tensor:
        """
        b: DGL Block (MFG)
        b.srcdata['h']: Node features for both src and dst nodes.
        b.edata['f']: Edge features.
        b.edata['dt']: Time DyX (ts_interaction - ts_neighbor_last_update).
        """
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=b.device)

        # 1. Prepare Features
        dst_h = cast(torch.Tensor, b.srcdata["h"])[: b.num_dst_nodes()]  # Target nodes
        src_h = cast(torch.Tensor, b.srcdata["h"])[
            b.num_dst_nodes() :
        ]  # Neighbor nodes

        # Time Encoding
        if self.time_enc is not None:
            time_feat = self.time_enc(b.edata["dt"])
            zero_time_feat = self.time_enc(
                torch.zeros(b.num_dst_nodes(), device=b.device)
            )
        else:
            time_feat = None
            zero_time_feat = None

        # 2. Compute Q, K, V
        # Q comes from Target Nodes (DST)
        Q = self.w_q(_cat_optional(dst_h, zero_time_feat, dim=1))

        # K, V come from Source/Neighbor Nodes (SRC) + Edges
        # Note: 'src_h' needs to be mapped to edges.
        # In DGL blocks, src_h are the tail nodes of edges, dst_h are head nodes.
        # But we need to index them correctly.
        # w_k/w_v inputs: neighbor_feat || edge_feat || time_feat

        # Helper to gather neighbor features for every edge
        # b.edges()[0] gives indices in the SRC domain (which includes DST and SRC nodes in source side)
        # But src_h is only the [num_dst:] part of srcdata.
        # Actually, simpler way: use all srcdata for K/V construction on edges

        # Correct logic per TGN implementation:
        # Use b.srcdata['h'] directly mapped to edges
        all_src_h = cast(torch.Tensor, b.srcdata["h"])
        src_indices = b.edges()[0]  # Indices in b.srcdata
        dst_indices = b.edges()[1]  # Indices in b.dstdata (0 to num_dst)

        neighbor_h_on_edge = all_src_h[src_indices]

        kv_input = _cat_optional(
            neighbor_h_on_edge, b.edata.get("efeat"), time_feat, dim=1
        )
        K = self.w_k(kv_input)
        V = self.w_v(kv_input)

        # We need Q to be expanded to edges for attention computation
        Q_on_edge = Q[dst_indices]

        # 3. Multi-head Attention
        Q_reshaped = Q_on_edge.reshape(-1, self.num_head, self.head_dim)
        K_reshaped = K.reshape(-1, self.num_head, self.head_dim)
        V_reshaped = V.reshape(-1, self.num_head, self.head_dim)

        # (Edges, Heads)
        att_score = torch.sum(Q_reshaped * K_reshaped, dim=2)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)

        # Softmax over incoming edges for each dst node
        att_score = dgl.ops.edge_softmax(b, att_score)
        att_score = self.dropout(att_score)

        # Aggregate V
        V_weighted = V_reshaped * att_score.unsqueeze(-1)
        V_weighted = V_weighted.reshape(-1, self.dim_out)

        b.edata["v"] = V_weighted
        b.update_all(fn.copy_e("v", "m"), fn.sum("m", "h_agg"))

        aggregated_h = cast(torch.Tensor, b.dstdata["h_agg"])

        # 4. Residual + Output Project
        # TGN logic: Linear(cat(h_old, h_agg))
        out = self.w_out(torch.cat([dst_h, aggregated_h], dim=1))
        out = F.relu(self.dropout(out))
        out = self.layer_norm(out)
        # out = self.ffn0(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn1(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn2(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn3(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn4(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn5(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn6(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn7(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn8(out) + out  # Transformer-style FFN with residual connection
        # out = self.ffn9(out) + out  # Transformer-style FFN

        return cast(torch.Tensor, out)


class IdentityNormLayer(torch.nn.Module):
    def __init__(self, d_hid: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_hid)

    def forward(self, b: DGLBlock) -> torch.Tensor:
        return cast(torch.Tensor, self.norm(b.srcdata["h"]))
