import logging
from typing import cast

import torch
import torch.nn as nn
from dgl.heterograph import DGLBlock

from ..components import TemporalGraphAttentionLayer


logger = logging.getLogger(__name__)


class TGAT(nn.Module):
    """
    TGAT Model (Multi-layer).
    """

    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int = 100,
        dim_hid: int = 100,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim_hid = dim_hid
        self.layers = nn.ModuleList()

        # --- Layer 1: Input (dim_node) -> Hidden (dim_hid) ---
        self.layers.append(
            TemporalGraphAttentionLayer(
                dim_node=dim_node,
                dim_edge=dim_edge,
                dim_time=dim_time,
                dim_out=dim_hid,
                num_head=heads,
                dropout=dropout,
            )
        )

        # --- Layer 2..N: Hidden (dim_hid) -> Hidden (dim_hid) ---
        for _ in range(num_layers - 1):
            self.layers.append(
                TemporalGraphAttentionLayer(
                    dim_node=dim_hid,  # Input is previous layer's output
                    dim_edge=dim_edge,
                    dim_time=dim_time,
                    dim_out=dim_hid,
                    num_head=heads,
                    dropout=dropout,
                )
            )

    def forward(
        self,
        mfgs: list[list[DGLBlock]],
    ) -> torch.Tensor:
        """
        mfgs: List of Message Flow Graphs (Blocks).
              Length must equal num_layers.
              mfgs[0] takes raw features.
              mfgs[1] takes output of mfgs[0].
        """
        assert len(mfgs) == self.num_layers, (
            f"Expected {self.num_layers} blocks, got {len(mfgs)}"
        )

        # Initial input feature comes from the first block's srcdata
        # Assuming data loader puts 'nfeat' into srcdata
        h = mfgs[0][0].srcdata["nfeat"]
        del mfgs[0][0].srcdata["nfeat"]

        for i in range(self.num_layers):
            # Prepare block data
            # The block needs 'h' in srcdata and dstdata
            # For the \first layer, h is raw features.
            # For subsequent layers, h is the hidden state from previous layer.

            # Map h to block
            mfgs[i][0].srcdata["h"] = h
            mfgs[i][0].dstdata["h"] = h[: mfgs[i][0].num_dst_nodes()]

            # Compute Layer
            h = self.layers[i](mfgs[i][0])

            # h is now the output for the *target* nodes of this layer.
            # It becomes the input for the *source* nodes of the next layer
            # (if we follow standard DGL block pipelining where block[i+1] srcs are subset of block[i] dsts)

        return cast(torch.Tensor, h)
