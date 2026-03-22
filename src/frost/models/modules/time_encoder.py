import logging

import numpy as np
import torch

# worker/trainer.py
import torch.nn as nn

logger = logging.getLogger(__name__)


class TimeEncoder(nn.Module):
    """
    Time Encoding as described in TGN/JODIE.
    Formula: cos(w * t + b) where w is determined by 1 / 10^linspace(0, 9, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)

        # Original TGN implementation
        seq = np.linspace(0, 9, dim, dtype=np.float32)

        # Initialize weights
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10**seq)).reshape(dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Add a feature dimension at the end. [N, S] -> [N, S, 1]
        t = t.unsqueeze(-1).float()

        # Apply linear transformation and cosine
        output = torch.cos(self.w(t))

        return output
