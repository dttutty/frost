from typing import List, Tuple

import torch
import torch.nn as nn
from dgl.heterograph import DGLBlock

from schemas.datatypes import DGLBlockMeta, SampleReturn
from schemas.plan import BatchNodeStateIO
from utils import extract_raw_data


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

    @staticmethod
    def analyze_block(mfg_meta: DGLBlockMeta) -> Tuple[set, set]:
        """
        Analyzes the provided DGLBlockMeta to determine the number of source and destination nodes.

        Args:
            mfg_meta (DGLBlockMeta): The DGLBlockMeta object to analyze.
        Returns:
            Tuple[set, set]: A tuple containing the set of all source nodes and the set of all root nodes.
        """
        all_nodes = set()
        root_nodes = set()
        nodes = mfg_meta.nodes.numpy()
        num_roots = mfg_meta.dim_in
        all_nodes.update(nodes.tolist())

        root_nodes.update(nodes[:num_roots].tolist())
        return all_nodes, root_nodes

    @staticmethod
    def analyze_need(mdb, neg_dst):
        mailbox_will_update_nodes, src_n_posdst = APAN.analyze_block(mdb)
        neg_nodes = set(neg_dst.numpy().tolist())
        all_nodes = mailbox_will_update_nodes + neg_nodes
        root_nodes = src_n_posdst + neg_nodes
        return BatchNodeStateIO(
            mem_needed=root_nodes,
            mem_updated=src_n_posdst,
            mailbox_needed=all_nodes,
            mailbox_updated=mailbox_will_update_nodes,
            nfeat_needed=root_nodes,
        )

    @staticmethod
    def sample(
        ngh_sampler,
        src: torch.Tensor,
        dst: torch.Tensor,
        neg_dst: torch.Tensor,
        ts: torch.Tensor,
    ):
        pos_root_np = torch.cat([src, dst]).numpy()
        pos_ts_np = torch.cat([ts, ts]).numpy()
        ngh_sampler.sample(pos_root_np, pos_ts_np)
        ret = ngh_sampler.get_ret()
        root2ngh_mfg = extract_raw_data(ret)[0]
        batch_node_state_io = APAN.analyze_need(root2ngh_mfg, neg_dst)
        return SampleReturn(
            all_nodes=None,
            root_nodes=None,
            ngh2root_mfg=None,
            root2ngh_mfg=root2ngh_mfg,
            batch_node_state_io=batch_node_state_io,
        )

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
