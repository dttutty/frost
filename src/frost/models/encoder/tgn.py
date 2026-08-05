"""TGN (Temporal Graph Network) encoder module."""
# worker/trainer.py

import logging

import torch

from frost.datatypes.graph import HierGraph
from frost.models.outputs import EncoderResult
from frost.models.modules.temporal_graph_attention_layer import (
    TemporalGraphAttentionLayer,
)

logger = logging.getLogger(__name__)


class TGN(torch.nn.Module):
    """Temporal Graph Network encoder (1-hop).

    Combines pre-fetched node memory with optional static node features,
    then applies one layer of temporal graph attention over the 1-hop
    neighborhood to produce root-node embeddings.

    Args:
        ndim: Dimensionality of raw node features.  Set to 0 if no node
            features are available.
        edim: Dimensionality of edge features.
        d_mem: Dimensionality of node memory (typically tied to ``encoder.out_dim``).
        d_time: Dimensionality of the time encoding.
        d_hid: Dimensionality of the attention output (hidden layer).
        heads: Number of attention heads.
        dropout: Dropout probability.
        attention_layout: ``dense`` or ``packed`` to force one kernel layout;
            ``auto`` selects from the input shape and projected neighbor width.
        auto_pack_min_projected_elements: Minimum theoretical dense projection
            work before ``auto`` uses packed attention.
    """

    def __init__(
        self,
        ndim: int,  # from hydra config
        edim: int,  # from hydra config
        d_mem: int,  # from hydra config
        d_time: int,  # from hydra config
        d_hid: int,  # from hydra config
        heads: int,  # from hydra config
        dropout: float,  # from hydra config
        attention_layout: str = "dense",
        auto_pack_min_projected_elements: int = 50_000_000,
    ) -> None:
        super().__init__()
        self.dim_node = ndim
        self.dim_edge = edim
        self.dim_memory = d_mem
        if attention_layout not in {"auto", "dense", "packed"}:
            raise ValueError(
                "attention_layout must be 'auto', 'dense', or 'packed', got "
                f"{attention_layout!r}"
            )
        if auto_pack_min_projected_elements <= 0:
            raise ValueError("auto_pack_min_projected_elements must be positive")
        self.attention_layout = attention_layout
        self.auto_pack_min_projected_elements = auto_pack_min_projected_elements
        # The dense path projects every fanout slot.  This width captures its
        # dominant per-slot K/V and time-encoding work without inspecting mask
        # values on the GPU (which would introduce a stream synchronization).
        self._projected_neighbor_width = d_mem + edim + d_time

        self.n_feat_mapper: torch.nn.Linear | None = None
        if ndim > 0 and ndim != d_mem:
            self.n_feat_mapper = torch.nn.Linear(ndim, d_mem)

        assert d_time >= 0, "must have time encoding if memory is used"

        self.encoder = TemporalGraphAttentionLayer(
            dim_node=d_mem,
            dim_edge=edim,
            dim_time=d_time,
            dim_out=d_hid,
            num_head=heads,
            dropout=dropout,
        )

    def forward(
        self,
        graph: HierGraph,
    ) -> EncoderResult:
        """Compute root-node embeddings via 1-hop temporal graph attention.

        Args:
            graph: Inflated graph with ``root_mem``, ``root_nfeat``,
                ``root_ts``, ``root_nid``, and ``neighbors[0]`` populated.

        Returns:
            Embedding tensor of shape ``[n_root, d_hid]``.
        """
        assert graph.neighbors is not None, "TGN requires neighbors for memory update"
        assert graph.fanout is not None, "TGN requires fanout metadata"
        assert graph.root_mem is not None, "TGN requires root memory"
        # 1. Prepare Target Features (Roots)
        root_mem = graph.root_mem
        if self.dim_node > 0 and graph.root_nfeat is not None:
            root_nfeat_emb = (
                self.n_feat_mapper(graph.root_nfeat)
                if self.n_feat_mapper
                else graph.root_nfeat
            )
            target_feat = root_mem + root_nfeat_emb
        else:
            target_feat = root_mem

        # 2. Prepare Neighbor Features (Hop 0)
        # Note: We assume 1-hop TGN here.
        nbr_data = graph.neighbors[0]
        assert nbr_data.mem is not None, "TGN requires neighbor memory"
        nbr_mem = nbr_data.mem
        if self.dim_node > 0 and nbr_data.nfeat is not None:
            if self.n_feat_mapper is not None:
                # The sampled layout is fanout-padded.  MAG has very sparse
                # historical neighborhoods, so applying the wide feature
                # projection to every padding row wastes most of this GEMM.
                # Keep the differentiable projection in the model, but run it
                # only for semantic neighbor rows and scatter the compact
                # result back to the dense attention layout.
                valid_neighbors = nbr_data.nid != -1
                nbr_nfeat_emb = torch.zeros_like(nbr_mem)
                nbr_nfeat_emb[valid_neighbors] = self.n_feat_mapper(
                    nbr_data.nfeat[valid_neighbors]
                )
            else:
                nbr_nfeat_emb = nbr_data.nfeat
            neighbor_combined_feat = nbr_mem + nbr_nfeat_emb
        else:
            neighbor_combined_feat = nbr_mem

        # 3. Reshape inputs to dense tensors [N, S, D] for Attention Layer
        N = graph.root_nid.numel()
        S = graph.fanout[0]

        neighbor_feat = neighbor_combined_feat.view(N, S, -1)
        edge_feat = (
            nbr_data.efeat.view(N, S, -1) if nbr_data.efeat is not None else None
        )

        # 4. Compute delta time (dt)
        neighbor_ts = nbr_data.ts.view(N, S)
        dt = graph.root_ts.unsqueeze(1) - neighbor_ts

        # 5. Create boolean mask (True for valid edges, False for padding)
        neighbor_nid = nbr_data.nid.view(N, S)
        mask = neighbor_nid != -1

        # 6. Pass into the attention layer
        use_packed = self._use_packed_attention(mask.numel())
        attention = self.encoder.forward_packed if use_packed else self.encoder
        out = attention(
            target_feat=target_feat,
            neighbor_feat=neighbor_feat,
            edge_feat=edge_feat,
            dt=dt,
            mask=mask,
        )

        return EncoderResult(out)

    def _use_packed_attention(self, dense_slots: int) -> bool:
        if self.attention_layout == "packed":
            return True
        if self.attention_layout == "dense":
            return False
        projected_elements = dense_slots * self._projected_neighbor_width
        return projected_elements >= self.auto_pack_min_projected_elements
