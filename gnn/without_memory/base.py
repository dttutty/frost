import torch.nn as nn

from schemas.datatypes.misc import ForwardCtx


class GNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        predictor: nn.Module,
    ):
        super().__init__()
        self.gnn = gnn
        self.predictor = predictor

    def forward(self, ctx: ForwardCtx):
        h = self.gnn(ctx.blockss)
        return self.predictor(h)
