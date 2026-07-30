"""TCL (Temporal Causal Linkage) encoder module.

Implements the TCL architecture adapted for the FROST framework, aligned with
the DyGLib reference implementation.  Each pair of (src, dst) root nodes is
processed by interleaved self-attention and cross-attention Transformer blocks,
then the root-position embedding is projected to the output dimension.

Key design choices matching DyGLib:
  - Post-norm Transformer blocks (LayerNorm after residual addition).
  - ReLU activation in the FFN (not GELU).
  - Configurable ``d_model`` for the internal hidden dimension, independent of
    the raw feature widths ``ndim`` / ``edim``.  DyGLib pads all features to
    172 and uses that as the attention dimension; FROST achieves the same
    effect by projecting raw features into ``d_model``.
  - Pair-aware output: returns ``[pos_src | pos_dst | neg_src | neg_dst]`` so
    that negative scoring uses the cross-attention-contextualized source
    embedding from the (src, neg_dst) pair, matching DyGLib exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from frost.datatypes.graph import HierGraph
from frost.models.outputs import EncoderResult
from frost.models.modules.time_encoder import TimeEncoder


class TransformerBlock(nn.Module):
    """Post-norm Transformer block matching DyGLib's ``TransformerEncoder``.

    Architecture per block:
      1. Multi-head attention (Q, K, V may differ for cross-attention).
      2. Residual connection + dropout + LayerNorm  (post-norm).
      3. FFN: Linear -> ReLU -> Dropout -> Linear.
      4. Residual connection + dropout + LayerNorm  (post-norm).
    """

    def __init__(self, attn_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(attn_dim, 4 * attn_dim)
        self.linear2 = nn.Linear(4 * attn_dim, attn_dim)
        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)

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
        key_pad_mask : [B, Lk]  -- True for positions to ignore
        """
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_pad_mask)
        x = self.norm1(q + self.dropout(attn_out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


class TCL(nn.Module):
    """TCL encoder aligned with the DyGLib reference implementation.

    Architecture (follows the DyGLib reference closely):
      1. Sequence construction: for each root node, build a length-(K+1)
         sequence ``[root, ngh_1, ..., ngh_K]``.  Four feature channels --
         node / edge / time / depth -- are each projected to ``d_model`` and
         summed into a single sequence tensor.
      2. Per layer::

             src_self  = transformer(Q=src, K=src, V=src)
             dst_self  = transformer(Q=dst, K=dst, V=dst)
             src_cross = transformer(Q=src_self, K=dst_self, V=dst_self)
             dst_cross = transformer(Q=dst_self, K=src_self, V=src_self)
             src, dst  = src_cross, dst_cross

      3. Take position 0 (root) from each sequence -> output_layer -> d_out.

    Output layout (pair-aware)::

        h = [pos_src_emb | pos_dst_emb | neg_src_emb | neg_dst_emb]

    This is normalized by ``RepresentationAdapter`` before scoring.
    """

    def __init__(
        self,
        ndim: int,
        edim: int,
        d_time: int,
        d_out: int,
        num_depths: int = 21,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        n_neg: int = 1,
        d_model: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model if d_model is not None else ndim
        self.ndim = ndim
        self.edim = edim
        self.d_time = d_time
        self.num_depths = num_depths
        self.n_neg = n_neg

        self.time_encoder = TimeEncoder(d_time)

        # Positional (depth) embedding: position 0 = root, 1..K = neighbors
        self.depth_embedding = nn.Embedding(num_depths, self.d_model)

        # Project each channel to d_model so they can be summed.
        # When ndim or edim is 0, the corresponding projection is omitted and
        # zeros are used instead.
        self.has_node_proj = ndim > 0
        self.has_edge_proj = edim > 0
        self.projection_layer = nn.ModuleDict({"time": nn.Linear(d_time, self.d_model)})
        if self.has_node_proj:
            self.projection_layer["node"] = nn.Linear(ndim, self.d_model)
        if self.has_edge_proj:
            self.projection_layer["edge"] = nn.Linear(edim, self.d_model)

        # One TransformerBlock per layer; reused for all 4 attention ops
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_layer = nn.Linear(self.d_model, d_out)

    # ------------------------------------------------------------------
    # Main entry point (FROST interface)
    # ------------------------------------------------------------------

    def forward(self, graph: HierGraph) -> EncoderResult:
        """Return embeddings in pair-aware layout.

        Output shape: ``[n_src * 2 * (1 + n_neg), d_out]`` laid out as
        ``[pos_src | pos_dst | neg_src | neg_dst]``.
        """
        assert graph.neighbors is not None, "TCL requires 1-hop neighbors"
        assert graph.fanout is not None, "TCL requires fanout metadata"
        n_root = graph.root_nid.shape[0]
        K = graph.fanout[0]
        n_src = n_root // (2 + graph.n_neg)

        # Build per-node sequences [n_root, L, d_model] -----------------
        seq, pad_mask = self._build_sequences(graph, n_root, K)

        # Split into src / pos_dst / neg_dst groups ----------------------
        src_seq = seq[0:n_src]
        pos_seq = seq[n_src : 2 * n_src]
        neg_seq = seq[2 * n_src :]

        src_mask = pad_mask[0:n_src]
        pos_mask = pad_mask[n_src : 2 * n_src]
        neg_mask = pad_mask[2 * n_src :]

        # Pass 1: (src, pos_dst) ----------------------------------------
        pos_src_emb, pos_dst_emb = self._forward_pair(
            src_seq, pos_seq, src_mask, pos_mask
        )

        # Pass 2: (src, neg_dst) ----------------------------------------
        n_neg_actual = n_root - 2 * n_src
        if n_neg_actual == n_src:
            neg_src_emb, neg_dst_emb = self._forward_pair(
                src_seq, neg_seq, src_mask, neg_mask
            )
        else:
            rep = n_neg_actual // n_src
            src_seq_tiled = src_seq.repeat_interleave(rep, dim=0)
            src_mask_tiled = src_mask.repeat_interleave(rep, dim=0)
            neg_src_emb, neg_dst_emb = self._forward_pair(
                src_seq_tiled, neg_seq, src_mask_tiled, neg_mask
            )

        # Pair-aware layout: [pos_src | pos_dst | neg_src | neg_dst]
        return EncoderResult(
            torch.cat([pos_src_emb, pos_dst_emb, neg_src_emb, neg_dst_emb], dim=0)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sequences(
        self, graph: HierGraph, n_root: int, K: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the combined feature sequence for all root nodes.

        Returns
        -------
        seq      : [n_root, L, d_model]  summed node+edge+time+depth features
        pad_mask : [n_root, L]           True at padded positions (nid == -1)
        """
        assert graph.neighbors is not None, "TCL requires 1-hop neighbors"
        dev = graph.root_nid.device
        nbr = graph.neighbors[0]

        # Clamp to num_depths-1 neighbors (truncate if K > num_depths-1)
        use_K = min(K, self.num_depths - 1)
        L = use_K + 1  # actual sequence length (may be < num_depths)

        # --- nid and timestamps (for masking and time deltas) -----------
        ngh_nid = nbr.nid.reshape(n_root, K)[:, :use_K]
        ngh_ts = nbr.ts.reshape(n_root, K)[:, :use_K]

        seq_nid = torch.cat([graph.root_nid.unsqueeze(1), ngh_nid], dim=1)
        seq_ts = torch.cat([graph.root_ts.unsqueeze(1), ngh_ts], dim=1)

        pad_mask = seq_nid == -1  # [n_root, L]

        # --- Node features ----------------------------------------------
        if (
            self.has_node_proj
            and graph.root_nfeat is not None
            and nbr.nfeat is not None
        ):
            ngh_nfeat = nbr.nfeat.reshape(n_root, K, -1)[:, :use_K, :]
            seq_nfeat = torch.cat([graph.root_nfeat.unsqueeze(1), ngh_nfeat], dim=1)
            seq_nfeat = seq_nfeat.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            node_proj = self.projection_layer["node"](seq_nfeat)
        else:
            node_proj = torch.zeros(n_root, L, self.d_model, device=dev)

        # --- Edge features (zero at position 0 -- no edge to self) ------
        if self.has_edge_proj and nbr.efeat is not None:
            ngh_efeat = nbr.efeat.reshape(n_root, K, -1)[:, :use_K, :]
            root_ef0 = torch.zeros(n_root, 1, self.edim, device=dev)
            seq_efeat = torch.cat([root_ef0, ngh_efeat], dim=1)
            seq_efeat = seq_efeat.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            edge_proj = self.projection_layer["edge"](seq_efeat)
        else:
            edge_proj = torch.zeros(n_root, L, self.d_model, device=dev)

        # --- Time encoding: delta_ts = root_ts - seq_ts -----------------
        dt = graph.root_ts.unsqueeze(1) - seq_ts
        time_feat = self.time_encoder(dt)
        time_feat = time_feat.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        time_proj = self.projection_layer["time"](time_feat)

        # --- Depth embedding: [L, d_model], broadcast over batch --------
        depth_feat = self.depth_embedding(torch.arange(L, device=dev)).unsqueeze(0)

        seq = node_proj + edge_proj + time_proj + depth_feat  # [n_root, L, d_model]

        # Pad to num_depths if L < num_depths
        if L < self.num_depths:
            pad_t = self.num_depths - L
            seq = F.pad(seq, (0, 0, 0, pad_t))
            pad_mask = F.pad(pad_mask, (0, pad_t), value=True)

        return seq, pad_mask

    def _forward_pair(
        self,
        src_seq: torch.Tensor,
        dst_seq: torch.Tensor,
        src_mask: torch.Tensor,
        dst_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one TCL forward for a (src, dst) pair.

        Returns (src_emb, dst_emb), each ``[B, d_out]``.
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
        src_emb = self.output_layer(s[:, 0, :])
        dst_emb = self.output_layer(d[:, 0, :])
        return src_emb, dst_emb
