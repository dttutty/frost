# worker/trainer.py

from typing import List

import torch
from dgl.heterograph import DGLBlock

from gnn.components import TemporalGraphAttentionLayer


class TGN(torch.nn.Module):
    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_memory: int = 100,
        dim_time: int = 100,
        dim_out: int = 100,
        att_h: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # parse from cfg
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_memory = dim_memory

        if dim_node > 0 and dim_node != dim_memory:
            self.n_feat_mapper = torch.nn.Linear(dim_node, dim_memory)
        else:
            self.n_feat_mapper = None

        assert dim_time >= 0, "must have time encoding if memory is used"

        self.encoder = TemporalGraphAttentionLayer(
            dim_node=dim_memory,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=att_h,
            dropout=dropout,
        )

    def forward(
        self,
        mfgs: List[List[DGLBlock]],
    ):
        block = mfgs[0][0]
        updated_mem: torch.Tensor = block.srcdata["updated_mem"]
        # fill in updated memory to blocks
        if self.dim_node > 0:
            nfeat = block.srcdata["nfeat"]
            if self.n_feat_mapper:
                node_feat_emb = self.n_feat_mapper(nfeat)
            else:
                node_feat_emb = nfeat
            h_combined = updated_mem + node_feat_emb
        else:
            h_combined = updated_mem
        del block.srcdata["nfeat"]
        block.srcdata["h"] = h_combined

        return self.encoder(block)
