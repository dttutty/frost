"""Pure temporal link-prediction computation graph."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from frost.datatypes.meta import Meta
from frost.models.contracts import RepresentationKind, RootExpansion
from frost.models.outputs import EncoderResult, RepresentationAdapter
from frost.models.protocols import MemoryUpdaterProto


class TemporalLinkModel(nn.Module):
    """Compose state update, graph encoding, representation, and scoring.

    This module owns only differentiable model computation. State publication,
    queues, CUDA events, P2P routing, and shared-memory writeback belong to the
    worker runtime.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        mem_updater: MemoryUpdaterProto | None,
        head: nn.Module,
        representation: RepresentationKind,
        fanout: tuple[int, ...],
        root_expansion: RootExpansion,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mem_updater = mem_updater
        self.head = head
        self.representation_adapter = RepresentationAdapter(representation)
        self.fanout = fanout
        self.root_expansion = root_expansion

    def forward(self, meta: Meta) -> torch.Tensor:
        flat_graph = meta.flat_graph
        if flat_graph is None:
            raise RuntimeError("Batch metadata is missing flat_graph.")

        if self.mem_updater is not None:
            flat_graph = self.mem_updater(
                flat_graph,
                meta.n_src * 2,
                n_root_nodes=meta.n_src * (2 + meta.n_neg),
            )
            meta.flat_graph = flat_graph

        graph = flat_graph.unflatten(
            n_src=meta.n_src,
            fanout=list(self.fanout),
            n_neg=meta.n_neg,
            root_expansion=self.root_expansion,
        )
        encoder_result = cast(EncoderResult, self.encoder(graph=graph))
        edge_representations = self.representation_adapter(
            encoder_result,
            n_src=meta.n_src,
            n_neg=meta.n_neg,
        )
        return cast(torch.Tensor, self.head(edge_representations))
