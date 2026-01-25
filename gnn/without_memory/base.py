import logging

import torch
import torch.nn as nn

from schemas.datatypes.misc import Meta


logger = logging.getLogger(__name__)


class GNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        header: nn.Module,
    ) -> None:
        super().__init__()
        self.gnn = gnn
        self.header = header

    def forward(self, meta: Meta) -> torch.Tensor:
        h = self.gnn(meta.gnn_mfgs)
        return self.header(h)
