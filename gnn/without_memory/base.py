import torch.nn as nn

from schemas.datatypes.misc import ForwardCtx


class GNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        header: nn.Module,
    ):
        super().__init__()
        self.gnn = gnn
        self.header = header

    def forward(self, ctx: ForwardCtx):
        h = self.gnn(ctx.blockss)
        return self.header(h)
