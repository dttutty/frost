# worker/trainer.py

import logging

import torch
from dgl.heterograph import DGLBlock

from gnn.components import TemporalGraphAttentionLayer
from gnn.sample_mixin import SampleMixin
from schemas.datatypes.misc import DGLBlockMeta
from schemas.datatypes.plan import BatchNodeStateIO
from tensor_set import TensorSet


logger = logging.getLogger(__name__)


class TGN(torch.nn.Module, SampleMixin):
    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_memory: int = 100,
        dim_time: int = 100,
        dim_out: int = 100,
        att_h: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # parse from cfg
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_memory = dim_memory
        self.n_feat_mapper: torch.nn.Linear | None = None
        if dim_node > 0 and dim_node != dim_memory:
            self.n_feat_mapper = torch.nn.Linear(dim_node, dim_memory)

        assert dim_time >= 0, "must have time encoding if memory is used"

        self.encoder = TemporalGraphAttentionLayer(
            dim_node=dim_memory,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=att_h,
            dropout=dropout,
        )

    @staticmethod
    def analyze_block(mfg_meta: DGLBlockMeta) -> tuple[TensorSet, TensorSet]:
        """
        Analyzes the provided list of DGLBlockMeta to determine the maximum
        number of source and destination nodes across the last n_history blocks.

        Args:
            mfg_meta_list (List[DGLBlockMeta]): List of DGLBlockMeta objects.

        Returns:
            Tuple[set, set]: A tuple containing the set of all   source nodes
                            and the set of all root nodes.
        """

        nodes = mfg_meta.nodes
        num_roots = mfg_meta.dim_out
        all_nodes = TensorSet(nodes)
        root_nodes = TensorSet(nodes[:num_roots])

        return all_nodes, root_nodes

    @staticmethod
    def analyze_dynam(
        blockss: list[DGLBlockMeta],
        src: torch.Tensor,
        pos_dst: torch.Tensor,
        neg_dst: torch.Tensor,
    ) -> BatchNodeStateIO:
        assert len(blockss) == 1, "TGN only supports 1 layer currently"

        all_nodes, root_nodes = TGN.analyze_block(blockss[0])
        mem_mailbox_will_update_nodes = TensorSet(src) + TensorSet(pos_dst)
        return BatchNodeStateIO(
            mem_needed=all_nodes,
            mem_updated=mem_mailbox_will_update_nodes,
            mailbox_needed=all_nodes,
            mailbox_updated=mem_mailbox_will_update_nodes,
            nfeat_needed=all_nodes,
        )

    def forward(
        self,
        mfgs: list[list[DGLBlock]],
    ) -> torch.Tensor:
        block = mfgs[0][0]
        updated_mem: torch.Tensor = block.srcdata["updated_mem"]
        # fill in updated memory to blocks
        if self.dim_node > 0:
            nfeat = block.srcdata["nfeat"]
            node_feat_emb = self.n_feat_mapper(nfeat) if self.n_feat_mapper else nfeat
            h_combined = updated_mem + node_feat_emb
        else:
            h_combined = updated_mem
        del block.srcdata["nfeat"]
        block.srcdata["h"] = h_combined

        return self.encoder(block)
