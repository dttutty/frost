"""TGAT (Temporal Graph Attention Network) encoder module.

Supports single- and multi-layer temporal attention by stacking
``TemporalGraphAttentionLayer`` blocks and aggregating neighborhood
information in a bottom-up fashion.
"""

import logging
from typing import cast

import torch
import torch.nn as nn

from frost.datatypes.graph import HierGraph
from frost.models.outputs import EncoderResult
from frost.models.modules.temporal_graph_attention_layer import (
    TemporalGraphAttentionLayer,
)

logger = logging.getLogger(__name__)


class TGAT(nn.Module):
    """
    TGAT Model (Multi-layer).
    """

    def __init__(
        self,
        ndim: int,  # from hydra config
        edim: int,  # from hydra config
        d_time: int,  # from hydra config
        d_hid: int,  # from hydra config
        fanout: list[int],  # from hydra config
        heads: int = 2,  # from hydra config
        dropout: float = 0.1,  # from hydra config
        enable_dummy_node_features: bool = True,
        dummy_node_dim: int | None = None,
        enable_dummy_edge_features: bool = True,
        dummy_edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = len(fanout)
        self.dim_hid = d_hid
        self.layers = nn.ModuleList()
        self.enable_dummy_node_features = enable_dummy_node_features
        self.enable_dummy_edge_features = enable_dummy_edge_features

        if ndim > 0:
            self.node_feat_dim = ndim
            self.uses_dummy_node_features = False
        elif self.enable_dummy_node_features:
            if dummy_node_dim is None:
                raise ValueError(
                    "dummy_node_dim must be resolved before constructing TGAT "
                    "when enable_dummy_node_features=True and ndim=0."
                )
            if dummy_node_dim <= 0:
                raise ValueError(
                    "dummy_node_dim must be > 0 when TGAT synthesizes node features."
                )
            self.node_feat_dim = dummy_node_dim
            self.uses_dummy_node_features = True
            logger.info(
                "TGAT received ndim=0; synthesizing %d-D zero node features on GPU.",
                self.node_feat_dim,
            )
        else:
            self.node_feat_dim = 0
            self.uses_dummy_node_features = False
            logger.info(
                "TGAT received ndim=0 with enable_dummy_node_features=False; "
                "disabling the node-feature channel."
            )

        if edim > 0:
            self.edge_feat_dim = edim
            self.uses_dummy_edge_features = False
        elif self.enable_dummy_edge_features:
            if dummy_edge_dim is None:
                raise ValueError(
                    "dummy_edge_dim must be resolved before constructing TGAT "
                    "when enable_dummy_edge_features=True and edim=0."
                )
            if dummy_edge_dim <= 0:
                raise ValueError(
                    "dummy_edge_dim must be > 0 when TGAT synthesizes edge features."
                )
            self.edge_feat_dim = dummy_edge_dim
            self.uses_dummy_edge_features = True
            logger.info(
                "TGAT received edim=0; synthesizing %d-D zero edge features on GPU.",
                self.edge_feat_dim,
            )
        else:
            self.edge_feat_dim = 0
            self.uses_dummy_edge_features = False
            logger.info(
                "TGAT received edim=0 with enable_dummy_edge_features=False; "
                "disabling the edge-feature channel."
            )

        # --- Layer 1: Input (dim_node) -> Hidden (dim_hid) ---
        self.layers.append(
            TemporalGraphAttentionLayer(
                dim_node=self.node_feat_dim,
                dim_edge=self.edge_feat_dim,
                dim_time=d_time,
                dim_out=d_hid,
                num_head=heads,
                dropout=dropout,
            )
        )

        # --- Layer 2..N: Hidden target and neighbor representations -> Hidden ---
        for _ in range(self.num_layers - 1):
            self.layers.append(
                TemporalGraphAttentionLayer(
                    dim_node=d_hid,
                    dim_neighbor=d_hid,
                    dim_edge=self.edge_feat_dim,
                    dim_time=d_time,
                    dim_out=d_hid,
                    num_head=heads,
                    dropout=dropout,
                )
            )

    @staticmethod
    def _resolve_feature_dtype(*tensors: torch.Tensor | None) -> torch.dtype:
        for tensor in tensors:
            if tensor is not None and tensor.is_floating_point():
                return tensor.dtype
        return torch.float32

    def _node_features(
        self,
        *,
        node_features: torch.Tensor | None,
        num_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if node_features is not None:
            return node_features
        if self.node_feat_dim > 0:
            return torch.zeros(
                num_rows,
                self.node_feat_dim,
                device=device,
                dtype=dtype,
            )
        return None

    def _edge_features_for_hop(
        self,
        *,
        edge_features: torch.Tensor | None,
        num_targets: int,
        fanout: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if edge_features is not None:
            return edge_features.view(num_targets, fanout, -1)
        if self.edge_feat_dim > 0:
            return torch.zeros(
                num_targets,
                fanout,
                self.edge_feat_dim,
                device=device,
                dtype=dtype,
            )
        return None

    def forward(self, graph: HierGraph) -> EncoderResult:
        """
        Forward pass using the pure PyTorch HierGraph structure.
        """
        assert graph.neighbors is not None, "Graph must have neighbors for TGAT."
        assert graph.fanout is not None, "TGAT requires fanout metadata."
        assert len(graph.neighbors) >= self.num_layers, (
            f"Expected at least {self.num_layers} hops of neighbors, "
            f"but got {len(graph.neighbors)}."
        )

        # Extract root features
        h_root = self._node_features(
            node_features=graph.root_nfeat,
            num_rows=graph.root_nid.numel(),
            device=graph.root_ts.device,
            dtype=self._resolve_feature_dtype(graph.root_nfeat, graph.root_mem),
        )
        assert h_root is not None, (
            "TGAT requires node features. Provide node features or enable "
            "dummy zero node features."
        )
        if graph.root_mem is not None:
            # Combine node features and memory if both exist
            h_root = h_root + graph.root_mem if h_root is not None else graph.root_mem

        # Fast path for single-layer TGAT
        if self.num_layers == 1:
            N = graph.root_nid.numel()
            S = graph.fanout[0]
            nbr = graph.neighbors[0]

            # Prepare neighbor features
            h_nbr = self._node_features(
                node_features=nbr.nfeat,
                num_rows=nbr.nid.numel(),
                device=h_root.device,
                dtype=self._resolve_feature_dtype(nbr.nfeat, nbr.mem, h_root),
            )
            assert h_nbr is not None, (
                "TGAT requires neighbor node features. Provide node features or "
                "enable dummy zero node features."
            )
            h_nbr = h_nbr.view(N, S, -1)
            if nbr.mem is not None:
                h_nbr = h_nbr + nbr.mem.view(N, S, -1)

            # Prepare edge features and time deltas
            e_feat = self._edge_features_for_hop(
                edge_features=nbr.efeat,
                num_targets=N,
                fanout=S,
                device=h_nbr.device,
                dtype=h_nbr.dtype,
            )
            dt = graph.root_ts.unsqueeze(1) - nbr.ts.view(N, S)
            mask = nbr.nid.view(N, S) != -1

            # Compute attention
            out = self.layers[0](
                target_feat=h_root,
                neighbor_feat=h_nbr,
                edge_feat=e_feat,
                dt=dt,
                mask=mask,
            )
            return EncoderResult(cast(torch.Tensor, out))

        # Multi-layer path (bottom-up aggregation)
        # We need to maintain a list of hidden states for each hop level
        hidden_states: list[torch.Tensor] = []

        # 1. Initialize features for all required hops
        # Hop 0 is root, Hop 1 is neighbors[0], Hop 2 is neighbors[1], etc.
        hidden_states.append(h_root)

        for i in range(self.num_layers):
            nbr = graph.neighbors[i]
            h_n = self._node_features(
                node_features=nbr.nfeat,
                num_rows=nbr.nid.numel(),
                device=h_root.device,
                dtype=self._resolve_feature_dtype(nbr.nfeat, nbr.mem, h_root),
            )
            assert h_n is not None, (
                "TGAT requires neighbor node features. Provide node features or "
                "enable dummy zero node features."
            )
            if nbr.mem is not None:
                h_n = h_n + nbr.mem if h_n is not None else nbr.mem
            hidden_states.append(h_n)

        # 2. Apply every layer to every still-live hop level.  For a two-layer
        # model, layer 0 embeds both roots from hop 1 and hop-1 nodes from hop
        # 2; layer 1 then embeds roots from those two first-layer outputs.  This
        # matches TGL's pair of nested message-flow blocks.
        for layer_idx in range(self.num_layers):
            next_hidden_states: list[torch.Tensor] = []
            target_level_count = self.num_layers - layer_idx
            for hop_idx in range(target_level_count):
                target_feat = hidden_states[hop_idx]
                num_targets = target_feat.shape[0]
                fanout = graph.fanout[hop_idx]
                nbr_level = graph.neighbors[hop_idx]
                neighbor_feat = hidden_states[hop_idx + 1].view(num_targets, fanout, -1)
                e_feat = self._edge_features_for_hop(
                    edge_features=nbr_level.efeat,
                    num_targets=num_targets,
                    fanout=fanout,
                    device=neighbor_feat.device,
                    dtype=neighbor_feat.dtype,
                )
                target_ts = (
                    graph.root_ts if hop_idx == 0 else graph.neighbors[hop_idx - 1].ts
                )
                dt = target_ts.unsqueeze(1) - nbr_level.ts.view(num_targets, fanout)
                mask = nbr_level.nid.view(num_targets, fanout) != -1
                next_hidden_states.append(
                    self.layers[layer_idx](
                        target_feat=target_feat,
                        neighbor_feat=neighbor_feat,
                        edge_feat=e_feat,
                        dt=dt,
                        mask=mask,
                    )
                )
            hidden_states = next_hidden_states

        # The final representation of the root nodes is at index 0
        return EncoderResult(cast(torch.Tensor, hidden_states[0]))
