"""GraphMixer encoder module.

Implements the GraphMixer architecture adapted for the FROST framework.
Consists of:

* ``FeedForwardNet`` — Two-layer MLP with GELU activation used inside MLP-Mixer.
* ``MLPMixer`` — MLP-Mixer block with token-mixing and channel-mixing FFNs.
* ``GraphMixer`` — Top-level encoder combining a link encoder (MLP-Mixer over
  neighbor edge/time features) and a node encoder (softmax-weighted mean of
  neighbor node features).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from frost.datatypes.graph import HierGraph
from frost.models.outputs import EncoderResult
from frost.models.modules.time_encoder import TimeEncoder

logger = logging.getLogger(__name__)


class FeedForwardNet(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(
        self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input and output dimensionality.
            dim_expansion_factor: Multiplier for the hidden layer size
                (``hidden_dim = int(dim_expansion_factor * input_dim)``).
            dropout: Dropout probability applied in the FFN.
        """
        super().__init__()
        hidden_dim = int(dim_expansion_factor * input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer FFN.

        Args:
            x: Input tensor of arbitrary shape ``[..., input_dim]``.

        Returns:
            Output tensor of the same shape ``[..., input_dim]``.
        """
        return self.ffn(x)


class MLPMixer(nn.Module):
    """
    MLP-Mixer block: alternates token-mixing and channel-mixing FFNs,
    each with a LayerNorm pre-norm and residual connection.

    input : [B, num_tokens, num_channels]
    output: [B, num_tokens, num_channels]
    """

    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(
            num_tokens, token_dim_expansion_factor, dropout
        )
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(
            num_channels, channel_dim_expansion_factor, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one MLP-Mixer block.

        Args:
            x: Input tensor of shape ``[B, num_tokens, num_channels]``.

        Returns:
            Output tensor of shape ``[B, num_tokens, num_channels]``.
        """
        # x: [B, T, C]
        # token mixing: operate over T, for each channel independently
        h = self.token_norm(x.permute(0, 2, 1))  # [B, C, T]
        h = self.token_feedforward(h).permute(0, 2, 1)  # [B, T, C]
        x = x + h

        # channel mixing: operate over C, for each token independently
        h = self.channel_norm(x)
        h = self.channel_feedforward(h)
        return x + h


class GraphMixer(nn.Module):
    """
    GraphMixer encoder adapted for the FROST framework.

    Architecture (follows the reference paper/code closely):
      1. Link encoder: for each root node, project its K sampled neighbors'
         (edge_feat ++ time_encoding) → num_channels, run through num_layers
         MLPMixer blocks, mean-pool over tokens → [n_root, num_channels].
      2. Node encoder: softmax-weighted mean of the same K neighbors' node
         features, plus the root node's own features → [n_root, ndim].
      3. Output layer: Linear(num_channels + ndim, d_out).

    Key differences from the original:
      - No internal feature store; nfeat/efeat come pre-fetched in Graph.
      - Interface: forward(graph: Graph) → Tensor[n_root, d_out].
      - Single neighbor set (graph.neighbors[0]) used for both encoders.
      - If the sampled fanout K < num_tokens, link_feat is zero-padded to
        num_tokens before MLPMixer so the LayerNorm dimension is consistent.
    """

    def __init__(
        self,
        ndim: int,  # node feature dim   (${ndim})
        edim: int,  # edge feature dim   (${edim})
        d_time: int,  # time encoding dim  (${d_time})
        d_out: int,  # output dim         (${encoder.out_dim})
        num_tokens: int = 20,
        num_channels: int = 172,
        num_layers: int = 2,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.1,
        freeze_time_encoder: bool = False,
        align_dyglib_zero_features: bool = False,
        dyglib_feature_dim: int | None = None,
    ):
        super().__init__()
        self.raw_ndim = ndim
        self.raw_edim = edim
        self.d_time = d_time
        self.num_tokens = num_tokens
        self.num_channels = num_channels
        self.align_dyglib_zero_features = align_dyglib_zero_features

        if self.align_dyglib_zero_features:
            if dyglib_feature_dim is None or dyglib_feature_dim <= 0:
                raise ValueError(
                    "dyglib_feature_dim must be > 0 when "
                    "align_dyglib_zero_features=True."
                )
            self.ndim = dyglib_feature_dim
            self.edim = dyglib_feature_dim
            logger.info(
                "GraphMixer received raw ndim=%d, edim=%d; using %d-D DyGLib-style "
                "zero node/edge features instead.",
                self.raw_ndim,
                self.raw_edim,
                dyglib_feature_dim,
            )
        else:
            self.ndim = ndim
            self.edim = edim

        self.time_encoder = TimeEncoder(d_time)
        if freeze_time_encoder:
            for p in self.time_encoder.parameters():
                p.requires_grad = False

        # Link encoder: project (edim + d_time) → num_channels
        self.projection_layer = nn.Linear(self.edim + d_time, num_channels)

        self.mlp_mixers = nn.ModuleList(
            [
                MLPMixer(
                    num_tokens=num_tokens,
                    num_channels=num_channels,
                    token_dim_expansion_factor=token_dim_expansion_factor,
                    channel_dim_expansion_factor=channel_dim_expansion_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection: concat link output and node output → d_out
        self.output_layer = nn.Linear(num_channels + self.ndim, d_out)

    def forward(self, graph: HierGraph) -> EncoderResult:
        """
        Returns h of shape [n_root, d_out].
        n_root = n_src * (2 + n_neg); all root nodes are processed independently.
        """
        assert graph.neighbors is not None, "GraphMixer requires 1-hop neighbors"
        assert graph.fanout is not None, "GraphMixer requires fanout metadata"
        nbr = graph.neighbors[0]
        n_root = graph.root_nid.shape[0]
        K = graph.fanout[0]
        use_K = min(K, self.num_tokens)

        # ------------------------------------------------------------------ #
        # Link encoder                                                        #
        # ------------------------------------------------------------------ #
        # The sampler emits fixed-width recent histories with valid neighbors
        # right-aligned in the K slots. DyGLib's direct num_neighbors=use_K call
        # is equivalent to taking the tail use_K block from a time_gap=K recent
        # sample, then left-aligning valid positions inside that block.
        start = K - use_K
        ngh_nid = nbr.nid.reshape(n_root, K)[:, start:]  # [n_root, use_K]
        ngh_ts = nbr.ts.reshape(n_root, K)[:, start:]  # [n_root, use_K]
        valid_mask = ngh_nid != -1

        ngh_nid = self._left_align_valid_positions(
            ngh_nid, valid_mask=valid_mask, pad_value=-1
        )
        ngh_ts = self._left_align_valid_positions(
            ngh_ts, valid_mask=valid_mask, pad_value=0
        )
        pad_mask = ngh_nid == -1  # [n_root, use_K]

        # Time delta: root_ts - neighbor_ts
        dt = graph.root_ts.unsqueeze(1) - ngh_ts  # [n_root, use_K]
        time_feat = self.time_encoder(dt)  # [n_root, use_K, d_time]

        # Zero out padding positions
        time_feat = time_feat.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        if self.align_dyglib_zero_features:
            ngh_efeat = torch.zeros(
                n_root,
                use_K,
                self.edim,
                device=time_feat.device,
                dtype=time_feat.dtype,
            )
            link_in = torch.cat([ngh_efeat, time_feat], dim=-1)
        elif nbr.efeat is not None and self.edim > 0:
            ngh_efeat = nbr.efeat.reshape(n_root, K, -1)[
                :, start:, :
            ]  # [n_root, use_K, edim]
            ngh_efeat = self._left_align_valid_positions(
                ngh_efeat, valid_mask=valid_mask, pad_value=0
            )
            ngh_efeat = ngh_efeat.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            link_in = torch.cat([ngh_efeat, time_feat], dim=-1)
        else:
            link_in = time_feat  # [n_root, use_K, d_time]

        link_feat = self.projection_layer(link_in)  # [n_root, use_K, num_channels]

        # Pad token dimension to num_tokens if K < num_tokens
        if use_K < self.num_tokens:
            pad_t = self.num_tokens - use_K
            link_feat = F.pad(
                link_feat, (0, 0, 0, pad_t)
            )  # [n_root, num_tokens, num_channels]

        for mixer in self.mlp_mixers:
            link_feat = mixer(link_feat)  # [n_root, num_tokens, num_channels]

        link_out = link_feat.mean(dim=1)  # [n_root, num_channels]

        # ------------------------------------------------------------------ #
        # Node encoder                                                        #
        # ------------------------------------------------------------------ #
        # DyGLib calls the recent sampler twice for GraphMixer:
        # num_neighbors for the link encoder and time_gap for the node encoder.
        # When the sampler is "recent", a single K=time_gap sample is enough to
        # emulate both calls: the node encoder uses all K slots, while the link
        # encoder above uses the tail num_tokens block from the same sample.
        if self.align_dyglib_zero_features:
            node_out = torch.zeros(
                (n_root, self.ndim), device=link_out.device, dtype=link_out.dtype
            )
        elif graph.root_nfeat is not None and nbr.nfeat is not None:
            ngh_nfeat_all = nbr.nfeat.reshape(n_root, K, -1)  # [n_root, K, ndim]

            # Build mask over full neighbor set
            pad_mask_all = nbr.nid.reshape(n_root, K) == -1  # [n_root, K]
            logits_all = torch.zeros(
                (n_root, K), device=ngh_nfeat_all.device, dtype=torch.float32
            )
            logits_all = logits_all.masked_fill(pad_mask_all, -1e10)
            scores_all = torch.softmax(logits_all, dim=1)  # [n_root, K]

            node_agg = (ngh_nfeat_all * scores_all.unsqueeze(-1)).mean(dim=1)
            node_out = node_agg + graph.root_nfeat  # [n_root, ndim]
        else:
            # No node features — produce zeros so output_layer still works
            node_out = torch.zeros(
                (n_root, self.ndim), device=link_out.device, dtype=link_out.dtype
            )

        # ------------------------------------------------------------------ #
        # Output                                                              #
        # ------------------------------------------------------------------ #
        return EncoderResult(
            self.output_layer(torch.cat([link_out, node_out], dim=-1))
        )  # [n_root, d_out]

    def _left_align_valid_positions(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        pad_value: int | float,
    ) -> torch.Tensor:
        """Left-pack valid positions inside a fixed-width recent-history block."""
        order = torch.argsort((~valid_mask).to(torch.int64), dim=1, stable=True)
        sorted_x = torch.gather(
            x,
            dim=1,
            index=order.unsqueeze(-1).expand_as(x) if x.ndim == 3 else order,
        )

        sorted_valid_mask = torch.gather(valid_mask, dim=1, index=order)
        if x.ndim == 3:
            sorted_valid_mask = sorted_valid_mask.unsqueeze(-1)

        sorted_x.masked_fill_(~sorted_valid_mask, pad_value)
        return sorted_x
