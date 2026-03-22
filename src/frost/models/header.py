"""Prediction header modules for link prediction and node classification.

This module provides ``LinkPredictor``, the standard link-prediction header
used across all FROST encoder models, and ``NodeClassificationHeader`` for
node-level tasks.
"""
import logging
from typing import cast

import torch

# worker/trainer.py
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LinkPredictor(nn.Module):
    """
    Predicts probability of an edge existing.
    Logic: MLP(h_src + h_dst) -> score
    """

    def __init__(
        self,
        d_in: int,  # from hydra config
        n_neg: int = 1,  # from hydra config
    ) -> None:
        super().__init__()
        self.dim_in = d_in
        self.neg_samples = n_neg
        self.src_fc = nn.Linear(d_in, d_in)
        self.dst_fc = nn.Linear(d_in, d_in)
        self.out_fc = nn.Linear(d_in, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute link-prediction logits from node embeddings.

        Expects ``h`` to be the concatenation of src, pos_dst, and neg_dst
        embeddings in that order.  Returns concatenated positive and negative
        edge scores.

        Args:
            h: Node embedding tensor of shape
               ``[n_src * (2 + n_neg), d_in]``.

        Returns:
            Logit tensor of shape ``[n_src * (1 + n_neg), 1]`` with positive
            edge scores first, followed by negative edge scores.
        """
        # h contains: [True Edges Src, True Edges Dst, Neg Edges Dst]
        num_edge = h.shape[0] // (self.neg_samples + 2)

        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge : 2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge :])

        h_pos_edge = F.relu(h_src + h_pos_dst)
        # Tile src for negative samples: [src1, src2] -> [src1, src1, src2, src2] if neg=2
        h_neg_edge = F.relu(h_src.tile(self.neg_samples, 1) + h_neg_dst)

        ret_1 = self.out_fc(h_pos_edge)
        ret_2 = self.out_fc(h_neg_edge)

        return torch.cat([ret_1, ret_2])


class NodeClassificationHeader(torch.nn.Module):
    """Two-layer MLP head for node classification tasks.

    Args:
        dim_in: Input embedding dimensionality.
        d_hid: Hidden layer dimensionality.
        num_class: Number of output classes.
    """

    def __init__(
        self,
        dim_in: int,  # from hydra config
        d_hid: int,  # from hydra config
        num_class: int,  # from hydra config
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim_in, d_hid)
        self.fc2 = torch.nn.Linear(d_hid, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify nodes from embeddings.

        Args:
            x: Node embedding tensor of shape ``[N, dim_in]``.

        Returns:
            Class logit tensor of shape ``[N, num_class]``.
        """
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return cast(torch.Tensor, x)
