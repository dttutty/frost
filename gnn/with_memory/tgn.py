# worker/trainer.py

import logging
from typing import cast

import torch
from dgl.heterograph import DGLBlock

from gnn.components import TemporalGraphAttentionLayer
from schemas.datatypes.misc import DGLBlockMeta
from schemas.datatypes.plan import BatchNodeStateIO
from tensor_set import TensorSet


logger = logging.getLogger(__name__)


def analyze_all_n_root(
    mfg_meta: DGLBlockMeta, n_src: int
) -> tuple[TensorSet, TensorSet]:
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
    all_nodes = TensorSet(nodes)
    root_nodes = TensorSet(nodes[: n_src * 2])

    return all_nodes, root_nodes


def analyze_need(
    all_nodes: TensorSet,
    src: torch.Tensor,
    pos_dst: torch.Tensor,
) -> BatchNodeStateIO:
    mem_mailbox_will_update_nodes = TensorSet(src) + TensorSet(pos_dst)
    return BatchNodeStateIO(
        mem_needed=all_nodes,
        mem_updated=mem_mailbox_will_update_nodes,
        mailbox_needed=all_nodes,
        mailbox_updated=mem_mailbox_will_update_nodes,
        nfeat_needed=all_nodes,
    )


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

    def forward(
        self,
        mfgs: list[list[DGLBlock]],
    ) -> torch.Tensor:
        block = mfgs[0][0]
        # show block info(datatype of each srcdata field)
        # print(block)

        updated_mem: torch.Tensor = cast(torch.Tensor, block.srcdata["updated_mem"])
        # fill in updated memory to blocks
        if self.dim_node > 0:
            nfeat = block.srcdata["nfeat"]
            node_feat_emb = self.n_feat_mapper(nfeat) if self.n_feat_mapper else nfeat
            h_combined = updated_mem + node_feat_emb
        else:
            h_combined = updated_mem
        del block.srcdata["nfeat"]
        block.srcdata["h"] = h_combined
        # block.srcdata["h"] = block.srcdata["nfeat"]

        return cast(torch.Tensor, self.encoder(block))
