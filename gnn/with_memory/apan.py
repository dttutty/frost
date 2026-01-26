import logging

import torch
import torch.nn as nn
from dgl.heterograph import DGLBlock

from gnn.sample_mixin import SampleMixin
from sampler_core import ParallelSampler
from schemas.datatypes.misc import DGLBlockMeta, SampleReturn
from schemas.datatypes.plan import BatchNodeStateIO


logger = logging.getLogger(__name__)


class APAN(nn.Module, SampleMixin):
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

    @staticmethod
    def analyze_block(mfg_meta: DGLBlockMeta) -> tuple[set, set]:
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

    @classmethod
    def analyze_need(cls, mdb: DGLBlockMeta, neg_dst: torch.Tensor) -> BatchNodeStateIO:
        mailbox_will_update_nodes, src_n_posdst = cls.analyze_block(mdb)
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

    @classmethod
    def sample(
        cls,
        ngh_sampler: ParallelSampler,
        src: torch.Tensor,
        dst: torch.Tensor,
        neg_dst: torch.Tensor,
        ts: torch.Tensor,
        sample_history: int = 1,  # <--- 必须加回这个参数，并给个默认值
    ) -> SampleReturn:
        pos_root_np = torch.cat([src, dst]).numpy()
        pos_ts_np = ts.repeat(2).numpy()
        ngh_sampler.sample(pos_root_np, pos_ts_np)
        ret = ngh_sampler.get_ret()
        root2ngh_mfg = cls.extract_raw_data(ret)[0]
        batch_node_state_io = cls.analyze_need(root2ngh_mfg, neg_dst)
        sample_ret = SampleReturn(
            ngh2root_mfgs_raw=None,
            root2ngh_mfg_raw=root2ngh_mfg,
            batch_node_state_io=batch_node_state_io,
        )
        return sample_ret

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
