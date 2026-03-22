"""GraphMixer encoder module.

Implements the GraphMixer architecture adapted for the FROST framework.
Consists of:

* ``FeedForwardNet`` — Two-layer MLP with GELU activation used inside MLP-Mixer.
* ``MLPMixer`` — MLP-Mixer block with token-mixing and channel-mixing FFNs.
* ``GraphMixer`` — Top-level encoder combining a link encoder (MLP-Mixer over
  neighbor edge/time features) and a node encoder (softmax-weighted mean of
  neighbor node features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from frost.datatypes.graph import SeqGraph
from frost.models.modules.time_encoder import TimeEncoder


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
        d_out: int,  # output dim         (${final_emb_dim})
        num_tokens: int = 20,
        num_channels: int = 100,
        num_layers: int = 2,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ndim = ndim
        self.edim = edim
        self.d_time = d_time
        self.num_tokens = num_tokens
        self.num_channels = num_channels

        self.time_encoder = TimeEncoder(d_time)

        # Link encoder: project (edim + d_time) → num_channels
        self.projection_layer = nn.Linear(edim + d_time, num_channels)

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
        self.output_layer = nn.Linear(num_channels + ndim, d_out)

    def forward(self, graph: SeqGraph) -> torch.Tensor:
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
        # Slice neighbors to use_K before reshape to avoid large intermediates
        ngh_nid = nbr.nid.reshape(n_root, K)[:, :use_K]  # [n_root, use_K]
        ngh_ts = nbr.ts.reshape(n_root, K)[:, :use_K]  # [n_root, use_K]
        assert nbr.efeat is not None, "GraphMixer requires edge features"
        ngh_efeat = nbr.efeat.reshape(n_root, K, -1)[
            :, :use_K, :
        ]  # [n_root, use_K, edim]

        pad_mask = ngh_nid == -1  # [n_root, use_K]

        # Time delta: root_ts - neighbor_ts
        dt = graph.root_ts.unsqueeze(1) - ngh_ts  # [n_root, use_K]
        time_feat = self.time_encoder(dt)  # [n_root, use_K, d_time]

        # Zero out padding positions
        time_feat = time_feat.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        ngh_efeat = ngh_efeat.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        link_in = torch.cat(
            [ngh_efeat, time_feat], dim=-1
        )  # [n_root, use_K, edim+d_time]
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
        # Softmax-weighted mean of neighbor node features + root node feature
        assert graph.root_nfeat is not None, "GraphMixer requires node features"
        assert nbr.nfeat is not None, "GraphMixer requires neighbor node features"

        ngh_nfeat = nbr.nfeat.reshape(n_root, K, -1)[
            :, :use_K, :
        ]  # [n_root, use_K, ndim]

        # Build mask: valid neighbors get 0, padding gets -1e10 (avoids nan from softmax)
        logits = torch.where(
            pad_mask,
            torch.full_like(dt[:, :use_K], -1e10),
            torch.zeros_like(dt[:, :use_K]),
        )
        scores = torch.softmax(logits, dim=1)  # [n_root, use_K]

        # Weighted mean over neighbor features
        node_agg = (ngh_nfeat * scores.unsqueeze(-1)).mean(dim=1)  # [n_root, ndim]

        # Add root node features
        node_out = node_agg + graph.root_nfeat  # [n_root, ndim]

        # ------------------------------------------------------------------ #
        # Output                                                              #
        # ------------------------------------------------------------------ #
        return self.output_layer(
            torch.cat([link_out, node_out], dim=-1)
        )  # [n_root, d_out]
