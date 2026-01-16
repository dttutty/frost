from typing import List

import torch.nn as nn
from dgl.heterograph import DGLBlock


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
        dim_memory=100,
        neg_samples=1,
    ):
        super(APAN, self).__init__()
        self.dim_memory = dim_memory
        self.neg_samples = neg_samples
        self.gnn_norm = nn.LayerNorm(dim_memory)

    def forward(
        self,
        mfgs: List[List[DGLBlock]],
    ):
        """
        mfgs: List of Message Flow Graphs (DGL Blocks).
              mfgs[0] is the Target Nodes block.
              It must contain 'mem', 'mem_input' (mailbox), 'ts', 'mail_ts'.
        """
        block = mfgs[0][0]

        embedding = self.gnn_norm(block.srcdata["updated_mem"])

        return embedding
