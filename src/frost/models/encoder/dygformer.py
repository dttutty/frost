"""DyGFormer encoder module.

Implements the DyGFormer architecture adapted for the FROST distributed
training framework.  Consists of:

* ``NeighborCooccurrenceEncoder`` — GPU-vectorised neighbor co-occurrence counter.
* ``TransformerEncoder`` — Pre-norm Transformer block.
* ``DyGFormer`` — Top-level encoder that builds patch sequences and runs the
  joint Transformer over (src, pos_dst) and (src, neg_dst) pairs.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from frost.datatypes.graph import HierGraph
from frost.models.outputs import EncoderResult
from frost.models.modules.time_encoder import TimeEncoder

logger = logging.getLogger(__name__)


class NeighborCooccurrenceEncoder(nn.Module):
    """
    Encodes neighbor co-occurrence between a src node's sequence and a dst
    node's sequence.  For each position in src's sequence, computes:
      - how many times that node appears in src's sequence
      - how many times that node appears in the paired dst's sequence
    (and vice versa for dst).  The resulting 2-D count vector is mapped to a
    latent embedding via a 2-layer MLP.

    GPU-only implementation (no numpy / Python loops over batch positions).
    """

    def __init__(self, neighbor_co_occurrence_feat_dim: int):
        """
        Args:
            neighbor_co_occurrence_feat_dim: Output dimensionality of the MLP
                that maps each count value to a latent embedding.
        """
        super().__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(1, neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(neighbor_co_occurrence_feat_dim, neighbor_co_occurrence_feat_dim),
        )

    def _count_appearances(
        self, src_ids: torch.Tensor, dst_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized co-occurrence counting via chunked broadcasting.

        src_ids : [B, Ls]
        dst_ids : [B, Ld]

        For each position j in src_ids[i]:
          - count how many times src_ids[i, j] appears in src_ids[i, :]
          - count how many times src_ids[i, j] appears in dst_ids[i, :]
        (symmetrically for dst_ids).

        Returns
        -------
        src_app : [B, Ls, 2]
        dst_app : [B, Ld, 2]
        """
        B, Ls = src_ids.shape
        Ld = dst_ids.shape[1]
        dev = src_ids.device

        src_in_src = torch.zeros(B, Ls, device=dev, dtype=torch.float32)
        src_in_dst = torch.zeros(B, Ls, device=dev, dtype=torch.float32)
        dst_in_src = torch.zeros(B, Ld, device=dev, dtype=torch.float32)
        dst_in_dst = torch.zeros(B, Ld, device=dev, dtype=torch.float32)

        # Chunk along the batch dim so the [C, Ls, Ld] broadcast tensors stay
        # within ~256 MB. chunk_size = floor(256M_floats / (Ls + Ld)²) ≥ 1.
        chunk = max(1, (1 << 22) // ((Ls + Ld) * max(Ls, Ld)))
        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            s = src_ids[lo:hi]  # [C, Ls]
            d = dst_ids[lo:hi]  # [C, Ld]

            # [C, Ls, Ls]: s[c, i] == s[c, j]?  → count along j
            src_in_src[lo:hi] = (s.unsqueeze(2) == s.unsqueeze(1)).sum(dim=2).float()
            # [C, Ls, Ld]: s[c, i] == d[c, j]?  → count along j
            src_in_dst[lo:hi] = (s.unsqueeze(2) == d.unsqueeze(1)).sum(dim=2).float()
            # [C, Ld, Ls]
            dst_in_src[lo:hi] = (d.unsqueeze(2) == s.unsqueeze(1)).sum(dim=2).float()
            # [C, Ld, Ld]
            dst_in_dst[lo:hi] = (d.unsqueeze(2) == d.unsqueeze(1)).sum(dim=2).float()

        src_app = torch.stack([src_in_src, src_in_dst], dim=2)  # [B, Ls, 2]
        dst_app = torch.stack([dst_in_src, dst_in_dst], dim=2)  # [B, Ld, 2]

        src_app[src_ids == -1] = 0.0
        dst_app[dst_ids == -1] = 0.0
        return src_app, dst_app

    def forward(
        self, src_ids: torch.Tensor, dst_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        src_feat : [B, Ls, neighbor_co_occurrence_feat_dim]
        dst_feat : [B, Ld, neighbor_co_occurrence_feat_dim]
        """
        src_app, dst_app = self._count_appearances(src_ids, dst_ids)
        # unsqueeze(-1) → [B, L, 2, 1] → MLP → [B, L, 2, D] → sum over dim 2 → [B, L, D]
        src_feat = self.neighbor_co_occurrence_encode_layer(src_app.unsqueeze(-1)).sum(
            dim=2
        )
        dst_feat = self.neighbor_co_occurrence_encode_layer(dst_app.unsqueeze(-1)).sum(
            dim=2
        )
        return src_feat, dst_feat


class TransformerEncoder(nn.Module):
    """
    Pre-norm Transformer encoder block, identical in structure to the
    reference DyGFormer implementation.
    """

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            attention_dim: Model dimensionality (embedding dim for Q/K/V and
                the FFN).
            num_heads: Number of attention heads.
            dropout: Dropout probability applied after attention and in the FFN.
        """
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
        """Apply one pre-norm Transformer block.

        Args:
            inputs: Input tensor of shape ``[B, num_patches, attention_dim]``.

        Returns:
            Output tensor of shape ``[B, num_patches, attention_dim]``.
        """
        # inputs : [B, num_patches, attention_dim]
        # PyTorch MultiheadAttention wants [seq, B, dim]
        x = inputs.transpose(0, 1)  # [P, B, D]
        normed = self.norm_layers[0](x)
        attn_out, _ = self.multi_head_attention(normed, normed, normed)
        x = inputs + self.dropout(attn_out.transpose(0, 1))  # residual

        # FFN with pre-norm
        ffn_in = self.norm_layers[1](x)
        ffn_out = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](ffn_in)))
        )
        return x + self.dropout(ffn_out)


class DyGFormer(nn.Module):
    """
    DyGFormer encoder adapted for the FROST framework.

    Key differences from the original reference:
    - No internal feature store: nfeat / efeat come pre-fetched in the Graph.
    - Interface: forward(graph: Graph) → Tensor[n_root, d_out]
      where n_root = n_src * (2 + n_neg).
    - Processes (src, pos_dst) and (src, neg_dst) pairs in two passes so that
      neighbor co-occurrence is computed between the correct paired nodes,
      matching the reference training logic.
    """

    def __init__(
        self,
        ndim: int,  # node feature dim   ${ndim}
        edim: int,  # edge feature dim   ${edim}
        d_time: int,  # time encoding dim  ${d_time}
        d_out: int,  # output dim         ${encoder.out_dim}
        channel_embedding_dim: int,  # per-channel latent dim
        patch_size: int,
        transformer_layers: int,
        num_heads: int,
        dropout: float,
        max_input_sequence_length: int,
        n_neg: int,  # ${n_neg}
        enable_dummy_node_features: bool = False,
        dummy_node_dim: int | None = None,
        enable_dummy_edge_features: bool = True,
        dummy_edge_dim: int | None = None,
    ):
        super().__init__()
        self.ndim = ndim
        self.edim = edim
        self.d_time = d_time
        self.d_out = d_out
        self.channel_embedding_dim = C = channel_embedding_dim
        self.patch_size = patch_size
        self.max_input_sequence_length = max_input_sequence_length
        self.n_neg = n_neg
        self.enable_dummy_node_features = enable_dummy_node_features
        self.enable_dummy_edge_features = enable_dummy_edge_features
        if ndim > 0:
            self.node_feat_dim = ndim
            self.node_channel_enabled = True
            self.uses_dummy_node_features = False
        elif self.enable_dummy_node_features:
            if dummy_node_dim is None:
                raise ValueError(
                    "dummy_node_dim must be resolved before constructing DyGFormer "
                    "when enable_dummy_node_features=True and ndim=0."
                )
            if dummy_node_dim <= 0:
                raise ValueError(
                    "dummy_node_dim must be > 0 when DyGFormer synthesizes node features."
                )
            self.node_feat_dim = dummy_node_dim
            self.node_channel_enabled = True
            self.uses_dummy_node_features = True
            logger.info(
                "DyGFormer received ndim=0; synthesizing %d-D zero node features on GPU "
                "to mirror DyGLib's dummy-feature path.",
                self.node_feat_dim,
            )
        else:
            self.node_feat_dim = 0
            self.node_channel_enabled = False
            self.uses_dummy_node_features = False
            logger.info(
                "DyGFormer received ndim=0 with enable_dummy_node_features=False; "
                "disabling the node channel."
            )
        if edim > 0:
            self.edge_feat_dim = edim
            self.edge_channel_enabled = True
            self.uses_dummy_edge_features = False
        elif self.enable_dummy_edge_features:
            if dummy_edge_dim is None:
                raise ValueError(
                    "dummy_edge_dim must be resolved before constructing DyGFormer "
                    "when enable_dummy_edge_features=True and edim=0."
                )
            if dummy_edge_dim <= 0:
                raise ValueError(
                    "dummy_edge_dim must be > 0 when DyGFormer synthesizes edge features."
                )
            self.edge_feat_dim = dummy_edge_dim
            self.edge_channel_enabled = True
            self.uses_dummy_edge_features = True
            logger.info(
                "DyGFormer received edim=0; synthesizing %d-D zero edge features on GPU "
                "to mirror DyGLib's dummy-feature path.",
                self.edge_feat_dim,
            )
        else:
            self.edge_feat_dim = 0
            self.edge_channel_enabled = False
            self.uses_dummy_edge_features = False
            logger.info(
                "DyGFormer received edim=0 with enable_dummy_edge_features=False; "
                "disabling the dummy edge channel."
            )

        self.time_encoder = TimeEncoder(d_time)
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(C)

        self.projection_layer = nn.ModuleDict(
            {
                "time": nn.Linear(patch_size * d_time, C),
                "neighbor_co_occurrence": nn.Linear(patch_size * C, C),
            }
        )
        if self.node_channel_enabled:
            self.projection_layer["node"] = nn.Linear(
                patch_size * self.node_feat_dim, C
            )
        if self.edge_channel_enabled:
            self.projection_layer["edge"] = nn.Linear(
                patch_size * self.edge_feat_dim, C
            )

        attn_dim = 4 * C
        # num_channels == 4, including node feat, edge feat, time feat, neighbor co-occurrence feat
        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(attn_dim, num_heads, dropout)
                for _ in range(transformer_layers)
            ]
        )
        self.output_layer = nn.Linear(attn_dim, d_out)

    # ------------------------------------------------------------------
    # Main entry point (FROST interface)
    # ------------------------------------------------------------------

    def forward(self, graph: HierGraph) -> EncoderResult:
        """
        Returns pair-aware embeddings laid out as:
          h[0 : n_src]                      = pos-pair src embeddings
          h[n_src : 2*n_src]                = pos dst embeddings
          h[2*n_src : 2*n_src + n_src*n_neg] = neg-pair src embeddings
          h[2*n_src + n_src*n_neg :]        = neg dst embeddings
        """
        assert graph.neighbors is not None, "DyGFormer requires 1-hop neighbors"
        assert graph.fanout is not None, "DyGFormer requires fanout information"
        assert len(graph.neighbors) == 1, "DyGFormer only supports 1-hop neighbors"
        assert len(graph.fanout) == 1, "DyGFormer only supports 1-hop fanout"
        n_root = graph.root_nid.shape[0]
        K = graph.fanout[0]
        n_src = n_root // (2 + graph.n_neg)

        # Build per-node sequence tensors --------------------------------
        # seq_nid  : [n_root, K+1]  (root at pos 0, then sampled neighbors)
        # seq_ts   : [n_root, K+1]
        # seq_nfeat: [n_root, K+1, ndim]
        # seq_efeat: [n_root, K+1, edge_feat_dim]
        seq = self._build_sequences(graph, n_root, K)

        # Split into three groups ----------------------------------------
        def _slice(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
            return t[lo:hi]

        src_s = {k: _slice(v, 0, n_src) for k, v in seq.items()}
        pos_s = {k: _slice(v, n_src, 2 * n_src) for k, v in seq.items()}
        neg_s = {k: _slice(v, 2 * n_src, n_root) for k, v in seq.items()}

        # Pass 1: (src, pos_dst) ----------------------------------------
        pos_src_emb, pos_dst_emb = self._forward_pair(src_s, pos_s)

        # Pass 2: (src, neg_dst) ----------------------------------------
        # For n_neg > 1 src must be tiled to match neg_dst count.
        n_neg_actual = n_root - 2 * n_src  # = n_src * n_neg
        if n_neg_actual == n_src:
            neg_src_emb, neg_dst_emb = self._forward_pair(src_s, neg_s)
        else:
            # tile each src sequence n_neg times along the batch dim
            rep = n_neg_actual // n_src
            src_tiled = {k: v.repeat_interleave(rep, dim=0) for k, v in src_s.items()}
            neg_src_emb, neg_dst_emb = self._forward_pair(src_tiled, neg_s)

        # Assemble pair-aware output for DyGFormer-specific prediction ----
        return EncoderResult(
            torch.cat([pos_src_emb, pos_dst_emb, neg_src_emb, neg_dst_emb], dim=0)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sequences(
        self, graph: HierGraph, n_root: int, K: int
    ) -> dict[str, torch.Tensor]:
        """
        Builds truncated, patch-aligned sequences for every root node.

        Returns a dict with keys: nid, ts, root_ts, nfeat, efeat
        Each value has shape [n_root, keep_len, ...].
        """
        assert graph.neighbors is not None
        assert len(graph.neighbors) == 1
        dev = graph.root_nid.device
        nbr = graph.neighbors[0]

        # Compute keep_len first so we can slice before cat (avoids allocating
        # huge [n_root, K+1, D] intermediates when K >> max_input_sequence_length)
        raw_len = K + 1
        keep_len = min(raw_len, self.max_input_sequence_length)
        keep_len = keep_len - (keep_len % self.patch_size)
        keep_len = max(keep_len, self.patch_size)  # at least one patch
        use_K = keep_len - 1  # neighbor slots after root

        # The sampler emits fixed-width recent histories with valid neighbors
        # right-aligned inside the K slots. DyGLib truncates to the most recent
        # use_K neighbors first, then left-aligns valid positions after the root.
        start = K - use_K
        ngh_nid = nbr.nid.reshape(n_root, K)[:, start:]  # [n_root, use_K]
        ngh_ts = nbr.ts.reshape(n_root, K)[:, start:]
        valid_mask = ngh_nid != -1

        ngh_nid = self._left_align_valid_positions(
            ngh_nid, valid_mask=valid_mask, pad_value=-1
        )
        ngh_ts = self._left_align_valid_positions(
            ngh_ts, valid_mask=valid_mask, pad_value=0
        )

        seq_nid = torch.cat(
            [graph.root_nid.unsqueeze(1), ngh_nid], dim=1
        )  # [n_root, keep_len]
        seq_ts = torch.cat(
            [graph.root_ts.unsqueeze(1), ngh_ts], dim=1
        )  # [n_root, keep_len]

        # Node features --------------------------------------------------
        if graph.root_nfeat is not None and nbr.nfeat is not None:
            ngh_nfeat = nbr.nfeat.reshape(n_root, K, -1)[
                :, start:, :
            ]  # [n_root, use_K, ndim]
            ngh_nfeat = self._left_align_valid_positions(
                ngh_nfeat, valid_mask=valid_mask, pad_value=0
            )
            seq_nfeat = torch.cat(
                [graph.root_nfeat.unsqueeze(1), ngh_nfeat], dim=1
            )  # [n_root, keep_len, ndim]
        else:
            seq_nfeat = torch.zeros(
                n_root,
                keep_len,
                self.node_feat_dim,
                device=dev,
            )

        # Edge features (zero at root position — no edge to self). When the
        # dataset has no real edge features, we either synthesize a dummy
        # channel for DyGLib-style parity or leave the edge channel disabled.
        if not self.edge_channel_enabled:
            seq_efeat = torch.empty(n_root, keep_len, 0, device=dev)
        elif nbr.efeat is not None:
            ngh_efeat = nbr.efeat.reshape(n_root, K, -1)[
                :, start:, :
            ]  # [n_root, use_K, edge_feat_dim]
            ngh_efeat = self._left_align_valid_positions(
                ngh_efeat, valid_mask=valid_mask, pad_value=0
            )
            root_efeat_0 = torch.zeros(n_root, 1, self.edge_feat_dim, device=dev)
            seq_efeat = torch.cat(
                [root_efeat_0, ngh_efeat], dim=1
            )  # [n_root, keep_len, edge_feat_dim]
        else:
            seq_efeat = torch.zeros(n_root, keep_len, self.edge_feat_dim, device=dev)

        # Zero out padding positions (nid == -1) -------------------------
        pad_mask = (seq_nid == -1).unsqueeze(-1)  # [n_root, keep_len, 1]
        seq_nfeat = seq_nfeat.masked_fill(pad_mask, 0.0)
        seq_efeat = seq_efeat.masked_fill(pad_mask, 0.0)

        return {
            "nid": seq_nid,  # [n_root, keep_len]
            "ts": seq_ts,  # [n_root, keep_len]
            "root_ts": graph.root_ts,  # [n_root]
            "nfeat": seq_nfeat,  # [n_root, keep_len, ndim]
            "efeat": seq_efeat,  # [n_root, keep_len, edge_feat_dim]
        }

    def _left_align_valid_positions(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        pad_value: int | float,
    ) -> torch.Tensor:
        """Pack valid neighbor positions to the left within each row."""
        packed = torch.full_like(x, pad_value)
        if not valid_mask.any():
            return packed

        dst_pos = valid_mask.long().cumsum(dim=1) - 1
        row_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
        row_idx = row_idx.expand_as(valid_mask)

        if x.ndim == 2:
            packed[row_idx[valid_mask], dst_pos[valid_mask]] = x[valid_mask]
        else:
            packed[row_idx[valid_mask], dst_pos[valid_mask], :] = x[valid_mask]
        return packed

    def _forward_pair(
        self,
        src_s: dict[str, torch.Tensor],
        dst_s: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a full DyGFormer forward for one (src, dst) pairing.

        Both dicts have the same keys as returned by _build_sequences
        but indexed to their respective slice of n_root.

        Returns (src_emb, dst_emb), each [B, d_out].
        """
        # 1. Neighbour co-occurrence (fully on GPU) ----------------------
        src_co, dst_co = self.neighbor_co_occurrence_encoder(
            src_s["nid"], dst_s["nid"]
        )  # [B, L, C] each

        # 2. Build patch embeddings for each node group ------------------
        src_patches = self._seq_to_patches(src_s, src_co)  # [B, P, 4C]
        dst_patches = self._seq_to_patches(dst_s, dst_co)  # [B, P, 4C]

        # 3. Joint transformer over (src ++ dst) patches -----------------
        P_src = src_patches.shape[1]
        combined = torch.cat([src_patches, dst_patches], dim=1)  # [B, P_src+P_dst, 4C]
        for transformer in self.transformers:
            combined = transformer(combined)

        # 4. Mean-pool each side, then project to d_out ------------------
        src_out = combined[:, :P_src, :].mean(dim=1)  # [B, 4C]
        dst_out = combined[:, P_src:, :].mean(dim=1)  # [B, 4C]
        return self.output_layer(src_out), self.output_layer(dst_out)

    def _seq_to_patches(
        self,
        seq: dict[str, torch.Tensor],
        co_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build patch embeddings for one group (src or dst).

        seq keys: nid [B,L], ts [B,L], root_ts [B], nfeat [B,L,ndim], efeat [B,L,edge_feat_dim]
        co_feat : [B, L, C]

        Returns : [B, num_patches, 4*C]
        """
        nid = seq["nid"]  # [B, L]
        ts = seq["ts"]  # [B, L]
        root_ts = seq["root_ts"]  # [B]
        nfeat = seq["nfeat"]  # [B, L, ndim]
        efeat = seq["efeat"]  # [B, L, edim]

        # Time-delta features; zero out at padding positions
        delta_ts = root_ts.unsqueeze(1) - ts  # [B, L]
        time_feat = self.time_encoder(delta_ts)  # [B, L, d_time]
        time_feat = time_feat.masked_fill((nid == -1).unsqueeze(-1), 0.0)

        # Project each channel after patchifying
        p_time = self.projection_layer["time"](self._to_patches(time_feat))  # [B, P, C]
        B_p, P_p = p_time.shape[0], p_time.shape[1]
        if self.node_channel_enabled:
            p_node = self.projection_layer["node"](self._to_patches(nfeat))  # [B, P, C]
        else:
            p_node = p_time.new_zeros(B_p, P_p, self.channel_embedding_dim)
        if self.edge_channel_enabled:
            p_edge = self.projection_layer["edge"](self._to_patches(efeat))  # [B, P, C]
        else:
            p_edge = p_time.new_zeros(B_p, P_p, self.channel_embedding_dim)
        p_co = self.projection_layer["neighbor_co_occurrence"](
            self._to_patches(co_feat)
        )  # [B, P, C]

        return torch.cat([p_node, p_edge, p_time, p_co], dim=2)  # [B, P, 4C]

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] → [B, L // patch_size, patch_size * D]"""
        B, L, D = x.shape
        return x.reshape(B, L // self.patch_size, self.patch_size * D)
