import logging

import torch
import torch.nn as nn
from dgl.heterograph import DGLBlock

from gnn.sample_mixin import extract_raw_data
from sampler_core import ParallelSampler
from schemas.datatypes.misc import DGLBlockMeta
from schemas.datatypes.plan import BatchNodeStateIO
from tensor_set import TensorSet


logger = logging.getLogger(__name__)


def analyze_all_n_root(mfg_meta: DGLBlockMeta) -> tuple[TensorSet, TensorSet]:
    """
    Analyzes the provided DGLBlockMeta to determine the number of source and destination nodes.

    Args:
        mfg_meta (DGLBlockMeta): The DGLBlockMeta object to analyze.
    Returns:
        Tuple[TensorSet, TensorSet]: A tuple containing the set of all source nodes and the set of all root nodes.
    """

    nodes = mfg_meta.nodes
    num_roots = mfg_meta.dim_in
    all_nodes = TensorSet(nodes)
    root_nodes = TensorSet(nodes[:num_roots])

    return all_nodes, root_nodes


def analyze_need(
    mailbox_will_update_nodes: TensorSet, src_n_posdst: TensorSet, neg_dst: torch.Tensor
) -> BatchNodeStateIO:
    all_nodes = mailbox_will_update_nodes + TensorSet(neg_dst)
    root_nodes = src_n_posdst + TensorSet(neg_dst)
    return BatchNodeStateIO(
        mem_needed=root_nodes,
        mem_updated=src_n_posdst,
        mailbox_needed=all_nodes,
        mailbox_updated=mailbox_will_update_nodes,
        nfeat_needed=root_nodes,
    )


def sample(
    ngh_sampler: ParallelSampler,
    src: torch.Tensor,
    dst: torch.Tensor,
    ts: torch.Tensor,
) -> DGLBlockMeta:
    pos_root_np = torch.cat([src, dst]).numpy()
    pos_ts_np = ts.repeat(2).numpy()
    ngh_sampler.sample(pos_root_np, pos_ts_np)
    ret = ngh_sampler.get_ret()
    root2ngh_mfg = extract_raw_data(ret)[0]

    return root2ngh_mfg


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
        dim_memory: int = 100,
        neg_samples: int = 1,
    ) -> None:
        super().__init__()
        self.dim_memory = dim_memory
        self.neg_samples = neg_samples
        self.gnn_norm = nn.LayerNorm(dim_memory)

    def forward(
        self,
        mfgs: list[list[DGLBlock]],
    ) -> torch.Tensor:
        """
        mfgs: List of Message Flow Graphs (DGL Blocks).
              mfgs[0] is the Target Nodes block.
              It must contain 'mem', 'mem_input' (mailbox), 'ts', 'mail_ts'.
        """
        block = mfgs[0][0]

        embedding = self.gnn_norm(block.srcdata["updated_mem"])

        return embedding
