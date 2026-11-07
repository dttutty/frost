# Adapted from lxd99/TPNet and lxd99/TGB_TPNet; see NOTICE.md and LICENSE.
import numpy as np
import torch
from torch import nn


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (
                torch.from_numpy(
                    1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
                )
            ).reshape(time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class LinkPredictor_v1(nn.Module):
    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        hidden_dim: int,
        output_dim: int,
        random_projections,
        not_encode,
    ):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.random_projections = random_projections
        self.not_encode = not_encode
        if self.random_projections is None:
            self.random_feature_dim = 0
        else:
            self.random_feature_dim = self.random_projections.pair_wise_feature_dim
        self.fc1 = nn.Linear(
            input_dim1 + input_dim2 + self.random_feature_dim, hidden_dim
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        src_node_embeddings: torch.Tensor,
        dst_node_embeddings: torch.Tensor,
    ):
        """
        merge and project the inputs
        :param dst_node_ids: ndarray, shape (batch_size)
        :param src_node_ids: ndarray, shape (batch_size)
        :param src_node_embeddings: Tensor, shape (batch_size, input_dim1)
        :param dst_node_embeddings: Tensor, shape (batch_size, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        if self.not_encode:
            src_node_embeddings = torch.zeros_like(src_node_embeddings)
            dst_node_embeddings = torch.zeros_like(dst_node_embeddings)
        if self.random_projections is None:
            x = torch.cat([src_node_embeddings, dst_node_embeddings], dim=1)
        else:
            pair_wise_features = self.random_projections.get_pair_wise_feature(
                src_node_ids=src_node_ids, dst_node_ids=dst_node_ids
            )
            x = torch.cat(
                [src_node_embeddings, dst_node_embeddings, pair_wise_features], dim=1
            )
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h
