"""TCL (Temporal Causal Linkage) encoder module.

Implements the TCL architecture adapted for the FROST framework.  Each pair of
(src, dst) root nodes is processed by interleaved self-attention and
cross-attention Transformer blocks, then the root-position embedding is
projected to the output dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from frost.datatypes.graph import SeqGraph
from frost.models.modules.time_encoder import TimeEncoder


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block that supports both self-attention and
    cross-attention.  Q, K, V can come from different sequences, allowing
    the same block to be reused for the four attention operations per TCL
    layer (src-self, dst-self, src-cross, dst-cross).
    """

    def __init__(self, attn_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            attn_dim: Model dimensionality for Q, K, V projections and the FFN.
            num_heads: Number of attention heads.
            dropout: Dropout probability applied after attention and in the FFN.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, 4 * attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * attn_dim, attn_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q : [B, Lq, D]
        k : [B, Lk, D]
        v : [B, Lk, D]
        key_pad_mask : [B, Lk]  — True for positions to ignore
        """
        normed_q = self.norm1(q)
        normed_k = self.norm1(k)
        normed_v = self.norm1(v)
        attn_out, _ = self.attn(
            normed_q, normed_k, normed_v, key_padding_mask=key_pad_mask
        )
        x = q + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TCL(nn.Module):
    """
    TCL (Temporal Causal Linkage) encoder adapted for the FROST framework.

    Architecture (follows the reference closely):
      1. Sequence construction: for each root node, build a length-(K+1)
         sequence [root, ngh_1, …, ngh_K].  Four feature channels —
         node / edge / time / depth — are each projected to `ndim` and
         summed into a single sequence tensor.
      2. Per layer:
           src_self  = transformer(Q=src, K=src, V=src)
           dst_self  = transformer(Q=dst, K=dst, V=dst)
           src_cross = transformer(Q=src_self, K=dst_self, V=dst_self)
           dst_cross = transformer(Q=dst_self, K=src_self, V=src_self)
           src, dst  = src_cross, dst_cross
      3. Take position 0 (root) from each sequence → output_layer → d_out.

    Key differences from the original:
      - No internal feature store; nfeat/efeat come pre-fetched in Graph.
      - Interface: forward(graph: Graph) → Tensor[n_root, d_out].
      - Processed as (src, pos_dst) and (src, neg_dst) pairs, matching the
        reference training logic.
      - If the sampled fanout K > num_depths-1, neighbors are truncated.
        If K < num_depths-1, the sequence is zero-padded to num_depths.
    """

    def __init__(
        self,
        ndim: int,  # node feature dim   (${ndim})
        edim: int,  # edge feature dim   (${edim})
        d_time: int,  # time encoding dim  (${d_time})
        d_out: int,  # output dim         (${encoder.out_dim})
        num_depths: int = 21,  # K+1; should match fanout[0]+1
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        n_neg: int = 1,  # (${n_neg})
    ):
        super().__init__()
        self.ndim = ndim
        self.edim = edim
        self.d_time = d_time
        self.num_depths = num_depths
        self.n_neg = n_neg

        self.time_encoder = TimeEncoder(d_time)

        # Positional (depth) embedding: position 0 = root, 1..K = neighbors
        self.depth_embedding = nn.Embedding(num_depths, ndim)

        # Project each channel to ndim so they can be summed
        self.projection_layer = nn.ModuleDict(
            {
                "node": nn.Linear(ndim, ndim),
                "edge": nn.Linear(edim, ndim),
                "time": nn.Linear(d_time, ndim),
            }
        )

        # One TransformerBlock per layer; reused for all 4 attention ops
        self.transformers = nn.ModuleList(
            [TransformerBlock(ndim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.output_layer = nn.Linear(ndim, d_out)

    # ------------------------------------------------------------------
    # Main entry point (FROST interface)
    # ------------------------------------------------------------------

    def forward(self, graph: SeqGraph) -> torch.Tensor:
        """
        Returns h of shape [n_root, d_out] where
          h[0 : n_src]          = src embeddings
          h[n_src : 2*n_src]    = pos_dst embeddings
          h[2*n_src :]          = neg_dst embeddings
        """
        assert graph.neighbors is not None, "TCL requires 1-hop neighbors"
        assert graph.fanout is not None, "TCL requires fanout metadata"
        n_root = graph.root_nid.shape[0]
        K = graph.fanout[0]
        n_src = n_root // (2 + self.n_neg)

        # Build per-node sequences [n_root, L, ndim] -------------------
        seq, pad_mask = self._build_sequences(graph, n_root, K)
        # seq      : [n_root, L, ndim]
        # pad_mask : [n_root, L]  True = padded

        # Split into src / pos_dst / neg_dst groups --------------------
        src_seq = seq[0:n_src]
        pos_seq = seq[n_src : 2 * n_src]
        neg_seq = seq[2 * n_src :]

        src_mask = pad_mask[0:n_src]
        pos_mask = pad_mask[n_src : 2 * n_src]
        neg_mask = pad_mask[2 * n_src :]

        # Pass 1: (src, pos_dst) ----------------------------------------
        src_emb, pos_dst_emb = self._forward_pair(src_seq, pos_seq, src_mask, pos_mask)

        # Pass 2: (src, neg_dst) ----------------------------------------
        n_neg_actual = n_root - 2 * n_src
        if n_neg_actual == n_src:
            _, neg_dst_emb = self._forward_pair(src_seq, neg_seq, src_mask, neg_mask)
        else:
            rep = n_neg_actual // n_src
            src_seq_tiled = src_seq.repeat_interleave(rep, dim=0)
            src_mask_tiled = src_mask.repeat_interleave(rep, dim=0)
            _, neg_dst_emb = self._forward_pair(
                src_seq_tiled, neg_seq, src_mask_tiled, neg_mask
            )

        return torch.cat([src_emb, pos_dst_emb, neg_dst_emb], dim=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sequences(
        self, graph: SeqGraph, n_root: int, K: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the combined feature sequence for all root nodes.

        Returns
        -------
        seq      : [n_root, L, ndim]   summed node+edge+time+depth features
        pad_mask : [n_root, L]         True at padded positions (nid == -1)
        """
        assert graph.neighbors is not None, "TCL requires 1-hop neighbors"
        dev = graph.root_nid.device
        nbr = graph.neighbors[0]

        # Clamp to num_depths-1 neighbors (truncate if K > num_depths-1)
        use_K = min(K, self.num_depths - 1)
        L = use_K + 1  # actual sequence length (may be < num_depths)

        # --- nid and timestamps (for masking and time deltas) -----------
        ngh_nid = nbr.nid.reshape(n_root, K)[:, :use_K]  # [n_root, use_K]
        ngh_ts = nbr.ts.reshape(n_root, K)[:, :use_K]  # [n_root, use_K]

        seq_nid = torch.cat(
            [graph.root_nid.unsqueeze(1), ngh_nid], dim=1
        )  # [n_root, L]
        seq_ts = torch.cat([graph.root_ts.unsqueeze(1), ngh_ts], dim=1)  # [n_root, L]

        pad_mask = seq_nid == -1  # [n_root, L]

        # --- Node features ----------------------------------------------
        if graph.root_nfeat is not None and nbr.nfeat is not None:
            ngh_nfeat = nbr.nfeat.reshape(n_root, K, -1)[:, :use_K, :]
            seq_nfeat = torch.cat([graph.root_nfeat.unsqueeze(1), ngh_nfeat], dim=1)
        else:
            seq_nfeat = torch.zeros(n_root, L, self.ndim, device=dev)

        # --- Edge features (zero at position 0 — no edge to self) ------
        if nbr.efeat is not None:
            ngh_efeat = nbr.efeat.reshape(n_root, K, -1)[:, :use_K, :]
            root_ef0 = torch.zeros(n_root, 1, self.edim, device=dev)
            seq_efeat = torch.cat([root_ef0, ngh_efeat], dim=1)
        else:
            seq_efeat = torch.zeros(n_root, L, self.edim, device=dev)

        # --- Time encoding: delta_ts = root_ts - seq_ts -----------------
        dt = graph.root_ts.unsqueeze(1) - seq_ts  # [n_root, L]
        time_feat = self.time_encoder(dt)  # [n_root, L, d_time]

        # Zero out padded positions
        seq_nfeat = seq_nfeat.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        seq_efeat = seq_efeat.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        time_feat = time_feat.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # --- Project each channel to ndim and sum -----------------------
        node_proj = self.projection_layer["node"](seq_nfeat)  # [n_root, L, ndim]
        edge_proj = self.projection_layer["edge"](seq_efeat)  # [n_root, L, ndim]
        time_proj = self.projection_layer["time"](time_feat)  # [n_root, L, ndim]

        # Depth embedding: [L, ndim], broadcast over batch
        depth_feat = self.depth_embedding(torch.arange(L, device=dev)).unsqueeze(
            0
        )  # [1, L, ndim]

        seq = node_proj + edge_proj + time_proj + depth_feat  # [n_root, L, ndim]

        # Pad to num_depths if L < num_depths (so all batches have same length)
        if L < self.num_depths:
            pad_t = self.num_depths - L
            seq = F.pad(seq, (0, 0, 0, pad_t))  # [n_root, num_depths, ndim]
            pad_mask = F.pad(pad_mask, (0, pad_t), value=True)  # [n_root, num_depths]

        return seq, pad_mask

    def _forward_pair(
        self,
        src_seq: torch.Tensor,  # [B, L, ndim]
        dst_seq: torch.Tensor,  # [B, L, ndim]
        src_mask: torch.Tensor,  # [B, L]
        dst_mask: torch.Tensor,  # [B, L]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one TCL forward for a (src, dst) pair.
        Returns (src_emb, dst_emb), each [B, d_out].
        """
        s = src_seq
        d = dst_seq

        for transformer in self.transformers:
            # Self-attention
            s_self = transformer(s, s, s, key_pad_mask=src_mask)
            d_self = transformer(d, d, d, key_pad_mask=dst_mask)
            # Cross-attention
            s = transformer(s_self, d_self, d_self, key_pad_mask=dst_mask)
            d = transformer(d_self, s_self, s_self, key_pad_mask=src_mask)

        # Take the root-node position (index 0) as the final embedding
        src_emb = self.output_layer(s[:, 0, :])  # [B, d_out]
        dst_emb = self.output_layer(d[:, 0, :])  # [B, d_out]
        return src_emb, dst_emb
