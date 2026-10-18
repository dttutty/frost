"""APAN (Asynchronous Propagation Attention Network) encoder module.

APAN's GNN component is intentionally minimal: after the ``TransformerMemoryUpdater``
has already integrated mailbox messages into node memory, the encoder simply
applies LayerNorm to the updated memory to produce the final node embedding.
"""

from typing import cast

import torch
import torch.nn as nn

from frost.datatypes.graph import HierGraph
from frost.models.outputs import EncoderResult

class APAN(nn.Module):
    """
    Simplified APAN Model.
    Architecture:
    1. Memory Update (Transformer over Mailbox)
    2. GNN (Identity + LayerNorm)
    3. Edge Prediction
    """

    def __init__(self, d_mem: int = 100) -> None:
        super().__init__()
        self.gnn_norm = nn.LayerNorm(d_mem)

    def forward(
        self,
        graph: HierGraph,
    ) -> EncoderResult:
        """
        Forward pass using HierGraph.
        """
        # Ensure root memory exists
        assert graph.root_mem is not None, "root_mem is missing in HierGraph"

        # Apply LayerNorm to target node memory
        embedding = self.gnn_norm(graph.root_mem)

        return EncoderResult(cast(torch.Tensor, embedding))
