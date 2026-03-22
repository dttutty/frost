"""APAN (Asynchronous Propagation Attention Network) encoder module.

APAN's GNN component is intentionally minimal: after the ``TransformerMemoryUpdater``
has already integrated mailbox messages into node memory, the encoder simply
applies LayerNorm to the updated memory to produce the final node embedding.
"""

import logging
from typing import cast

import torch
import torch.nn as nn

from frost.datatypes.graph import SeqGraph

logger = logging.getLogger(__name__)


class APAN(nn.Module):
    """
    Simplified APAN Model.
    Architecture:
    1. Memory Update (Transformer over Mailbox)
    2. GNN (Identity + LayerNorm)
    3. Edge Prediction
    """

    def __init__(
        self,
        d_mem: int = 100,  # from hydra config
        n_neg: int = 1,  # from hydra config
    ) -> None:
        super().__init__()
        self.dim_memory = d_mem
        self.neg_samples = n_neg
        self.gnn_norm = nn.LayerNorm(d_mem)

    def forward(
        self,
        graph: SeqGraph,
    ) -> torch.Tensor:
        """
        Forward pass using SeqGraph.
        """
        # Ensure root memory exists
        assert graph.root_mem is not None, "root_mem is missing in SeqGraph"

        # Apply LayerNorm to target node memory
        embedding = self.gnn_norm(graph.root_mem)

        return cast(torch.Tensor, embedding)
