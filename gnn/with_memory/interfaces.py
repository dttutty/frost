from torch import nn


class MemoryUpdater(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
