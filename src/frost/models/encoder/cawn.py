"""CAWN encoder module.

Implements the Causal Anonymous Walks Network (CAWN) architecture adapted for
the FROST distributed training framework. Consists of:

* ``CachedPositionEncoder`` -- GPU-vectorized node appearance counter for anonymous walks.
* ``BiLSTMEncoder`` -- Encodes temporal and causal sequences along the walk.
* ``WalkEncoder`` -- Aggregates edge, node, time, and positioning via BiLSTM + Transformer.
* ``CAWN`` -- The top-level encoder pairing nodes appropriately and forming walk embeddings.
"""

import logging
from typing import List, Dict

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from frost.datatypes.graph import HierGraph
from frost.models.outputs import EncoderResult
from frost.models.modules.time_encoder import TimeEncoder

logger = logging.getLogger(__name__)


class CachedPositionEncoder(nn.Module):
    """
    Position encoder that computes each node's causal anonymous walk landing probability.
    Rewritten from DyGLib's Python dict representation into vectorized GPU chunking.
    """

    def __init__(self, position_feat_dim: int, walk_length: int):
        super().__init__()
        self.walk_length = walk_length
        self.position_encode_layer = nn.Sequential(
            nn.Linear(walk_length + 1, position_feat_dim),
            nn.ReLU(),
            nn.Linear(position_feat_dim, position_feat_dim),
        )

    def forward(
        self,
        query_nids: torch.Tensor,
        src_tree: List[torch.Tensor],
        dst_tree: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            query_nids: [B, num_walks, H] the nodes in the walk
            src_tree: List of length H=walk_length+1, each [B, S_hop]
            dst_tree: List of length H=walk_length+1, each [B, D_hop]
        Returns:
            position_features: [B, num_walks, H, position_feat_dim]
        """
        B, num_walks, H = query_nids.shape
        dev = query_nids.device

        # Flatten query nids for easier chunked counting across a single length dimension
        flat_query_nids = query_nids.view(B, num_walks * H)
        Q = flat_query_nids.shape[1]

        # [B, Q, 2, H]
        out_appearances = torch.zeros((B, Q, 2, H), device=dev, dtype=torch.float32)

        # Chunk over batch size to restrict max memory during boolean broadcasting
        max_hop_size = max([t.shape[1] for t in src_tree])
        chunk = max(1, (1 << 24) // (Q * max_hop_size))

        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            q_chunk = flat_query_nids[lo:hi]  # [C, Q]

            # ignores padding (-1)
            q_valid = q_chunk != -1

            for h in range(H):
                # source appearances
                s_hop = src_tree[h][lo:hi]  # [C, S_h]
                s_hop_valid = s_hop != -1
                match_s = q_chunk.unsqueeze(2) == s_hop.unsqueeze(1)
                match_s &= q_valid.unsqueeze(2)
                match_s &= s_hop_valid.unsqueeze(1)
                counts_s = match_s.sum(dim=2).float() / max(1, s_hop.shape[1])
                out_appearances[lo:hi, :, 0, h] = counts_s

                # destination appearances
                d_hop = dst_tree[h][lo:hi]  # [C, D_h]
                d_hop_valid = d_hop != -1
                match_d = q_chunk.unsqueeze(2) == d_hop.unsqueeze(1)
                match_d &= q_valid.unsqueeze(2)
                match_d &= d_hop_valid.unsqueeze(1)
                counts_d = match_d.sum(dim=2).float() / max(1, d_hop.shape[1])
                out_appearances[lo:hi, :, 1, h] = counts_d

        # [B, num_walks, H, 2, H]
        out_appearances = out_appearances.view(B, num_walks, H, 2, H)

        # MLP + sum over src/dst
        # [B, num_walks, H, 2, position_feat_dim]
        pos_feat = self.position_encode_layer(out_appearances)
        # [B, num_walks, H, position_feat_dim]
        pos_feat = pos_feat.sum(dim=-2)
        return pos_feat


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim_one_direction = hidden_dim // 2
        self.model_dim = self.hidden_dim_one_direction * 2
        self.bilstm_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim_one_direction,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode the inputs by BiLSTM based on valid lengths.
        Returns the encoding at the last valid position of each sequence.
        """
        B_W, seq_len, D = inputs.shape

        # Enforce seq_len >= lengths >= 1
        cpu_lengths = lengths.cpu().clamp(min=1, max=seq_len)

        packed_inputs = pack_padded_sequence(
            inputs, cpu_lengths, batch_first=True, enforce_sorted=False
        )
        encoded_features, _ = self.bilstm_encoder(packed_inputs)
        encoded_features, _ = pad_packed_sequence(
            encoded_features, batch_first=True, total_length=seq_len
        )

        # Retrieve the final state (lengths - 1)
        last_indices = (
            (cpu_lengths - 1)
            .to(inputs.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(B_W, 1, self.model_dim)
        )
        final_features = torch.gather(encoded_features, 1, last_indices).squeeze(1)

        return final_features


class TransformerEncoder(nn.Module):
    """
    Standard pre-norm Transformer encoder block used to aggregate random walks.
    """

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(attention_dim, 4 * attention_dim),
                nn.Linear(4 * attention_dim, attention_dim),
            ]
        )
        self.norm_layers = nn.ModuleList(
            [
                nn.LayerNorm(attention_dim),
                nn.LayerNorm(attention_dim),
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Pytorch MHA expects [seq, batch, dim]
        x = inputs.transpose(0, 1)
        normed = self.norm_layers[0](x)
        attn_out, _ = self.multi_head_attention(normed, normed, normed)
        x = inputs + self.dropout(attn_out.transpose(0, 1))

        ffn_in = self.norm_layers[1](x)
        ffn_out = self.linear_layers[1](
            self.dropout(torch.nn.functional.relu(self.linear_layers[0](ffn_in)))
        )
        return x + self.dropout(ffn_out)


class WalkEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        position_feat_dim: int,
        output_dim: int,
        num_walk_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.position_feat_dim = position_feat_dim
        self.attention_dim = self.input_dim // 2
        self.output_dim = output_dim
        self.num_walk_heads = num_walk_heads

        if self.attention_dim % self.num_walk_heads != 0:
            self.attention_dim += (
                self.num_walk_heads - self.attention_dim % self.num_walk_heads
            )

        self.feature_encoder = BiLSTMEncoder(
            input_dim=self.input_dim, hidden_dim=self.input_dim
        )
        self.position_encoder = BiLSTMEncoder(
            input_dim=self.position_feat_dim, hidden_dim=self.position_feat_dim
        )

        self.transformer_encoder = TransformerEncoder(
            attention_dim=self.attention_dim,
            num_heads=self.num_walk_heads,
            dropout=dropout,
        )

        self.projection_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.feature_encoder.model_dim
                    + self.position_encoder.model_dim,
                    out_features=self.attention_dim,
                ),
                nn.Linear(in_features=self.attention_dim, out_features=self.output_dim),
            ]
        )

    def forward(
        self,
        combined_raw_feat: torch.Tensor,
        neighbor_position_features: torch.Tensor,
        walks_valid_lengths: torch.Tensor,
    ):
        """
        combined_raw_feat: [B, num_walks, H, D_raw]
        neighbor_position_features: [B, num_walks, H, D_pos]
        walks_valid_lengths: [B, num_walks]
        """
        B, num_walks, H, D_raw = combined_raw_feat.shape
        D_pos = neighbor_position_features.shape[-1]

        # Flatten batch and walks
        combined_flat = combined_raw_feat.view(B * num_walks, H, D_raw)
        pos_flat = neighbor_position_features.view(B * num_walks, H, D_pos)
        len_flat = walks_valid_lengths.view(B * num_walks)

        feat_out = self.feature_encoder(combined_flat, len_flat)
        pos_out = self.position_encoder(pos_flat, len_flat)

        # [B * num_walks, feat_dim + pos_dim]
        combined_out = torch.cat([feat_out, pos_out], dim=-1)
        # [B, num_walks, ...]
        combined_out = combined_out.view(B, num_walks, -1)

        combined_out = self.projection_layers[0](combined_out)

        # Aggregate multiple random walks over num_walks dimension
        # [B, attention_dim]
        attn_out = self.transformer_encoder(combined_out).mean(dim=1)

        outputs = self.projection_layers[1](attn_out)
        return outputs


class CAWN(nn.Module):
    """
    CAWN encoder adapted for the FROST framework.
    """

    def __init__(
        self,
        ndim: int,
        edim: int,
        d_time: int,
        d_out: int,
        position_feat_dim: int,
        walk_length: int,
        num_walk_heads: int,
        n_neg: int,
        dropout: float = 0.1,
        enable_dummy_edge_features: bool = True,
        dummy_edge_dim: int = 1,
    ):
        super().__init__()
        self.ndim = ndim
        self.edim = edim
        self.d_time = d_time
        self.out_dim = d_out
        self.n_neg = n_neg
        self.position_feat_dim = position_feat_dim
        self.walk_length = walk_length

        if edim > 0:
            self.edge_feat_dim = edim
        elif enable_dummy_edge_features:
            self.edge_feat_dim = dummy_edge_dim
        else:
            self.edge_feat_dim = 0

        self.time_encoder = TimeEncoder(dim=d_time)
        self.position_encoder = CachedPositionEncoder(position_feat_dim, walk_length)

        walk_input_dim = (
            self.ndim + self.edge_feat_dim + self.d_time + self.position_feat_dim
        )
        self.walk_encoder = WalkEncoder(
            input_dim=walk_input_dim,
            position_feat_dim=self.position_feat_dim,
            output_dim=self.out_dim,
            num_walk_heads=num_walk_heads,
            dropout=dropout,
        )

    def forward(self, graph: HierGraph) -> EncoderResult:
        """
        Returns pair-aware embeddings laid out as:
          h[0 : n_src]                      = pos-pair src embeddings
          h[n_src : 2*n_src]                = pos dst embeddings
          h[2*n_src : 2*n_src + n_src*n_neg] = neg-pair src embeddings
          h[2*n_src + n_src*n_neg :]        = neg dst embeddings
        """
        assert graph.neighbors is not None, "CAWN requires neighbors (at least 1 hop)"
        assert graph.fanout is not None, "CAWN requires fanout information"
        assert len(graph.neighbors) == self.walk_length, (
            f"CAWN requires exactly {self.walk_length} hops"
        )

        n_root = graph.root_nid.shape[0]
        n_src = n_root // (2 + graph.n_neg)

        # 1. Build trees arrays sequentially
        src_tree = self._slice_tree(graph, 0, n_src, n_root, graph.fanout)
        pos_tree = self._slice_tree(graph, n_src, 2 * n_src, n_root, graph.fanout)
        neg_tree = self._slice_tree(graph, 2 * n_src, n_root, n_root, graph.fanout)

        # 2. Pass 1: (src, pos_dst)
        pos_src_emb, pos_dst_emb = self._forward_pair(src_tree, pos_tree, graph.fanout)

        # 3. Pass 2: (src, neg_dst)
        n_neg_actual = n_root - 2 * n_src
        if n_neg_actual == n_src:
            neg_src_emb, neg_dst_emb = self._forward_pair(
                src_tree, neg_tree, graph.fanout
            )
        else:
            # tile src sequences n_neg times
            rep = n_neg_actual // n_src
            src_tiled = {
                k: [sub.repeat_interleave(rep, dim=0) for sub in v]
                for k, v in src_tree.items()
            }
            neg_src_emb, neg_dst_emb = self._forward_pair(
                src_tiled, neg_tree, graph.fanout
            )

        return EncoderResult(
            torch.cat([pos_src_emb, pos_dst_emb, neg_src_emb, neg_dst_emb], dim=0)
        )

    def _slice_tree(
        self, graph: HierGraph, lo: int, hi: int, n_root: int, fanout: List[int]
    ) -> Dict[str, List[torch.Tensor]]:
        """Extract multi-hop ego-graphs for a slice of the batch."""
        res = {
            "nid": [graph.root_nid[lo:hi].unsqueeze(1)],
            "ts": [graph.root_ts[lo:hi].unsqueeze(1)],
        }
        if graph.root_nfeat is not None:
            res["nfeat"] = [graph.root_nfeat[lo:hi].unsqueeze(1)]
        else:
            dev = graph.root_nid.device
            res["nfeat"] = [torch.zeros((hi - lo, 1, self.ndim), device=dev)]

        dev = graph.root_nid.device
        res["efeat"] = [torch.zeros((hi - lo, 1, self.edge_feat_dim), device=dev)]

        # Neighbors
        if graph.neighbors is None:
            raise ValueError("CAWN requires sampled neighbors")
        neighbors = graph.neighbors
        for i, f in enumerate(fanout):
            chunk = 1
            for k in range(i + 1):
                chunk *= fanout[k]

            # Slice flattened representation via view
            neighbor = neighbors[i]
            nid_slice = neighbor.nid.view(n_root, chunk)[lo:hi]
            ts_slice = neighbor.ts.view(n_root, chunk)[lo:hi]
            res["nid"].append(nid_slice)
            res["ts"].append(ts_slice)

            if neighbor.nfeat is not None:
                nfeat_slice = neighbor.nfeat.view(n_root, chunk, -1)[lo:hi]
                res["nfeat"].append(nfeat_slice)
            else:
                res["nfeat"].append(
                    torch.zeros((hi - lo, chunk, self.ndim), device=dev)
                )

            if neighbor.efeat is not None:
                efeat_slice = neighbor.efeat.view(n_root, chunk, -1)[lo:hi]
                res["efeat"].append(efeat_slice)
            else:
                res["efeat"].append(
                    torch.zeros((hi - lo, chunk, self.edge_feat_dim), device=dev)
                )

        return res

    def _forward_pair(
        self,
        src_tree: Dict[str, List[torch.Tensor]],
        dst_tree: Dict[str, List[torch.Tensor]],
        fanout: List[int],
    ):
        """Processes pair of trees together to form the co-occurrence relationships and evaluate representations."""
        src_arr = self._tree_to_array(src_tree, fanout)
        dst_arr = self._tree_to_array(dst_tree, fanout)

        src_emb = self._encode_walk_array(src_arr, src_tree["nid"], dst_tree["nid"])
        dst_emb = self._encode_walk_array(dst_arr, dst_tree["nid"], src_tree["nid"])

        return src_emb, dst_emb

    def _tree_to_array(
        self, tree: Dict[str, List[torch.Tensor]], fanout: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Aligns tree shape (lists of varied dimension) into uniform paths array `[B, num_walks, H, D]`."""
        num_walks = 1
        for f in fanout:
            num_walks *= f
        B = tree["nid"][0].shape[0]
        H = len(fanout) + 1

        arr = {}
        for key, tensor_list in tree.items():
            first_shape = tensor_list[0].shape
            out_shape = (B, num_walks, H) + first_shape[2:]
            out = torch.empty(
                out_shape, dtype=tensor_list[0].dtype, device=tensor_list[0].device
            )

            out[:, :, 0] = tensor_list[0].repeat_interleave(num_walks, dim=1)

            num_paths_at_hop = 1
            for i in range(len(fanout)):
                h = i + 1
                f = fanout[i]
                num_paths_at_hop *= f
                repeats = num_walks // num_paths_at_hop
                out[:, :, h] = tensor_list[h].repeat_interleave(repeats, dim=1)
            arr[key] = out

        return arr

    def _encode_walk_array(
        self,
        arr: Dict[str, torch.Tensor],
        src_tree_nid: List[torch.Tensor],
        dst_tree_nid: List[torch.Tensor],
    ):
        """
        arr dict contains values of shape [B, num_walks, H, ...]
        """
        # Valid lengths
        walks_valid_lengths = (arr["nid"] != -1).sum(dim=-1)

        # Position encoding
        # [B, num_walks, H, pos_feat_dim]
        pos_feat = self.position_encoder(arr["nid"], src_tree_nid, dst_tree_nid)

        # Time difference relative to the target root node
        # [B, num_walks, H]
        delta_ts = arr["ts"][:, :, 0:1] - arr["ts"]
        time_feat = self.time_encoder(delta_ts.flatten(start_dim=1)).view(
            *delta_ts.shape, self.d_time
        )
        # Mask out padding with zeros
        pad_mask = (arr["nid"] == -1).unsqueeze(-1)
        time_feat = time_feat.masked_fill(pad_mask, 0.0)
        nfeat = arr["nfeat"].masked_fill(pad_mask, 0.0)
        efeat = arr["efeat"].masked_fill(pad_mask, 0.0)

        # Combine Raw Features
        combined = torch.cat([nfeat, time_feat, efeat, pos_feat], dim=-1)

        # Feed to WalkEncoder
        out = self.walk_encoder(combined, pos_feat, walks_valid_lengths)
        return out
