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

    def forward(self, h: torch.Tensor, n_neg: int | None = None) -> torch.Tensor:
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
        runtime_n_neg = self.neg_samples if n_neg is None else n_neg
        if runtime_n_neg <= 0:
            raise ValueError(f"n_neg must be positive, got {runtime_n_neg}")
        root_width = runtime_n_neg + 2
        if h.shape[0] % root_width != 0:
            raise ValueError(
                f"Embedding rows ({h.shape[0]}) must be divisible by 2 + n_neg "
                f"({root_width})."
            )
        num_edge = h.shape[0] // root_width

        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge : 2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge :])

        h_pos_edge = F.relu(h_src + h_pos_dst)
        # Negatives are row-major per positive edge:
        # [src1-neg1, src1-neg2, src2-neg1, src2-neg2].
        h_neg_src = h_src.repeat_interleave(runtime_n_neg, dim=0)
        h_neg_edge = F.relu(h_neg_src + h_neg_dst)

        ret_1 = self.out_fc(h_pos_edge)
        ret_2 = self.out_fc(h_neg_edge)

        return torch.cat([ret_1, ret_2])


class MergeLayerPredictor(nn.Module):
    """Concat-based link predictor matching DyGLib's MergeLayer.

    Logic: MLP(concat(h_src, h_dst)) -> score
    Used by GraphMixer / TCL to match the original DyGLib implementation.
    """

    def __init__(
        self,
        d_in: int,
        n_neg: int = 1,
    ) -> None:
        super().__init__()
        self.neg_samples = n_neg
        self.fc1 = nn.Linear(2 * d_in, d_in)
        self.fc2 = nn.Linear(d_in, 1)

    def forward(self, h: torch.Tensor, n_neg: int | None = None) -> torch.Tensor:
        runtime_n_neg = self.neg_samples if n_neg is None else n_neg
        if runtime_n_neg <= 0:
            raise ValueError(f"n_neg must be positive, got {runtime_n_neg}")
        root_width = runtime_n_neg + 2
        if h.shape[0] % root_width != 0:
            raise ValueError(
                f"Embedding rows ({h.shape[0]}) must be divisible by 2 + n_neg "
                f"({root_width})."
            )
        num_edge = h.shape[0] // root_width

        h_src = h[:num_edge]
        h_pos_dst = h[num_edge : 2 * num_edge]
        h_neg_dst = h[2 * num_edge :]

        h_pos = torch.cat([h_src, h_pos_dst], dim=1)
        h_neg_src = h_src.repeat_interleave(runtime_n_neg, dim=0)
        h_neg = torch.cat([h_neg_src, h_neg_dst], dim=1)

        pos_score = self.fc2(F.relu(self.fc1(h_pos)))
        neg_score = self.fc2(F.relu(self.fc1(h_neg)))

        return torch.cat([pos_score, neg_score])


class DyGFormerLinkPredictor(nn.Module):
    """Pair-aware predictor for DyGFormer.

    DyGFormer jointly encodes each (src, dst) pair, so negative edges must use
    the source embedding produced from the negative pairing instead of reusing
    the positive-pair source embedding.
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

    def forward(self, h: torch.Tensor, n_neg: int | None = None) -> torch.Tensor:
        """Compute logits from DyGFormer's pair-specific embedding layout.

        Expects ``h`` laid out as:
        ``[pos_src | pos_dst | neg_src | neg_dst]``
        with lengths ``[n_src, n_src, n_src*n_neg, n_src*n_neg]``.
        """
        runtime_n_neg = self.neg_samples if n_neg is None else n_neg
        if runtime_n_neg <= 0:
            raise ValueError(f"n_neg must be positive, got {runtime_n_neg}")
        pair_width = 2 * (runtime_n_neg + 1)
        if h.shape[0] % pair_width != 0:
            raise ValueError(
                f"Embedding rows ({h.shape[0]}) must be divisible by "
                f"2 * (1 + n_neg) ({pair_width})."
            )
        num_edge = h.shape[0] // pair_width
        neg_span = num_edge * runtime_n_neg

        h_pos_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge : 2 * num_edge])
        h_neg_src = self.src_fc(h[2 * num_edge : 2 * num_edge + neg_span])
        h_neg_dst = self.dst_fc(h[2 * num_edge + neg_span :])

        h_pos_edge = F.relu(h_pos_src + h_pos_dst)
        h_neg_edge = F.relu(h_neg_src + h_neg_dst)

        ret_1 = self.out_fc(h_pos_edge)
        ret_2 = self.out_fc(h_neg_edge)
        return torch.cat([ret_1, ret_2])


class PairAwareMergeLayerPredictor(nn.Module):
    """Concat-based pair-aware link predictor matching DyGLib's MergeLayer.

    Like ``MergeLayerPredictor`` but expects the pair-aware embedding layout
    ``[pos_src | pos_dst | neg_src | neg_dst]`` so that negative scoring uses
    the cross-attention-contextualized source embedding from the (src, neg_dst)
    pair, matching DyGLib's TCL evaluation exactly.
    """

    def __init__(
        self,
        d_in: int,
        n_neg: int = 1,
    ) -> None:
        super().__init__()
        self.neg_samples = n_neg
        self.fc1 = nn.Linear(2 * d_in, d_in)
        self.fc2 = nn.Linear(d_in, 1)

    def forward(self, h: torch.Tensor, n_neg: int | None = None) -> torch.Tensor:
        """Compute logits from pair-aware embedding layout.

        Expects ``h`` laid out as
        ``[pos_src | pos_dst | neg_src | neg_dst]``
        with lengths ``[n_src, n_src, n_src*n_neg, n_src*n_neg]``.
        """
        runtime_n_neg = self.neg_samples if n_neg is None else n_neg
        if runtime_n_neg <= 0:
            raise ValueError(f"n_neg must be positive, got {runtime_n_neg}")
        pair_width = 2 * (runtime_n_neg + 1)
        if h.shape[0] % pair_width != 0:
            raise ValueError(
                f"Embedding rows ({h.shape[0]}) must be divisible by "
                f"2 * (1 + n_neg) ({pair_width})."
            )
        num_edge = h.shape[0] // pair_width
        neg_span = num_edge * runtime_n_neg

        h_pos_src = h[:num_edge]
        h_pos_dst = h[num_edge : 2 * num_edge]
        h_neg_src = h[2 * num_edge : 2 * num_edge + neg_span]
        h_neg_dst = h[2 * num_edge + neg_span :]

        h_pos = torch.cat([h_pos_src, h_pos_dst], dim=1)
        h_neg = torch.cat([h_neg_src, h_neg_dst], dim=1)

        pos_score = self.fc2(F.relu(self.fc1(h_pos)))
        neg_score = self.fc2(F.relu(self.fc1(h_neg)))

        return torch.cat([pos_score, neg_score])


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
