"""Task heads over structured model representations."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from frost.models.outputs import EdgeRepresentations


class AdditiveLinkHead(nn.Module):
    """Score edges with an MLP over transformed source/destination sums."""

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.src_fc = nn.Linear(d_in, d_in)
        self.dst_fc = nn.Linear(d_in, d_in)
        self.out_fc = nn.Linear(d_in, 1)

    def forward(self, edges: EdgeRepresentations) -> torch.Tensor:
        positive = F.relu(
            self.src_fc(edges.positive_source) + self.dst_fc(edges.positive_destination)
        )
        negative = F.relu(
            self.src_fc(edges.negative_source) + self.dst_fc(edges.negative_destination)
        )
        return torch.cat([self.out_fc(positive), self.out_fc(negative)])


class ConcatLinkHead(nn.Module):
    """Score edges with a DyGLib-style MLP over concatenated endpoints."""

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * d_in, d_in)
        self.fc2 = nn.Linear(d_in, 1)

    def forward(self, edges: EdgeRepresentations) -> torch.Tensor:
        positive = torch.cat([edges.positive_source, edges.positive_destination], dim=1)
        negative = torch.cat([edges.negative_source, edges.negative_destination], dim=1)
        positive_score = self.fc2(F.relu(self.fc1(positive)))
        negative_score = self.fc2(F.relu(self.fc1(negative)))
        return torch.cat([positive_score, negative_score])


class NodeClassificationHeader(nn.Module):
    """Two-layer MLP head for node classification tasks."""

    def __init__(self, dim_in: int, d_hid: int, num_class: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim_in, d_hid)
        self.fc2 = nn.Linear(d_hid, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return cast(torch.Tensor, x)
