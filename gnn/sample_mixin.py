from __future__ import annotations

import logging

import torch

from sampler_core import ParallelSampler
from schemas.datatypes.misc import DGLBlockMeta, SampleReturn
from schemas.datatypes.plan import BatchNodeStateIO


logger = logging.getLogger(__name__)


class SampleMixin:
    @classmethod
    def extract_raw_data(cls, ret: list) -> list[DGLBlockMeta]:
        # 用简单的字典或者 NamedTuple 存储纯数据
        mfg_meta_list = []
        for tgb in ret:
            # 1. 提取 Numpy 数据并转为 CPU Tensor
            # 此时不要去 GPU，也不要创建 DGLBlock
            mfg_meta: DGLBlockMeta = DGLBlockMeta(
                nodes=torch.from_numpy(tgb.nodes()),
                ts=torch.from_numpy(tgb.ts()),
                eid=torch.from_numpy(tgb.eid()),
                dts=torch.from_numpy(tgb.dts()),
                col=torch.from_numpy(tgb.col()),
                row=torch.from_numpy(tgb.row()),
                dim_in=tgb.dim_in(),
                dim_out=tgb.dim_out(),
            )
            mfg_meta_list.append(mfg_meta)
        return mfg_meta_list

    @classmethod
    def analyze_blocks(
        cls, mfg_meta_list: list[DGLBlockMeta], n_history: int
    ) -> tuple[set, set]:
        all_nodes = set()
        root_nodes = set()
        for index, mfg_meta in enumerate(mfg_meta_list):
            nodes = mfg_meta.nodes.numpy()
            num_roots = mfg_meta.dim_out
            all_nodes.update(nodes.tolist())
            if index < n_history:
                root_nodes.update(nodes[:num_roots].tolist())
        return all_nodes, root_nodes

    @classmethod
    def analyze_need(
        cls, mfg_meta_list: list[DGLBlockMeta], n_history: int
    ) -> BatchNodeStateIO:
        """
        Analyzes the provided list of DGLBlockMeta to determine the maximum
        number of source and destination nodes across the last n_history blocks.

        Args:
            mfg_meta_list (List[DGLBlockMeta]): List of DGLBlockMeta objects.

        Returns:
            BatchNodeStateIO: An object containing the sets of all source nodes
                            and the set of all root nodes.
        """
        all_nodes, root_nodes = cls.analyze_blocks(mfg_meta_list, n_history)
        batch_node_state_io = BatchNodeStateIO(
            mem_needed=all_nodes,
            mem_updated=root_nodes,
            mailbox_needed=all_nodes,
            mailbox_updated=root_nodes,
            nfeat_needed=all_nodes,
        )
        return batch_node_state_io

    @classmethod
    def sample(
        cls,
        ngh_sampler: ParallelSampler,
        src: torch.Tensor,
        dst: torch.Tensor,
        neg_dst: torch.Tensor,
        ts: torch.Tensor,
        sample_history: int = 1,
    ) -> SampleReturn:
        roots = torch.cat([src, dst, neg_dst])
        ts_np = ts.repeat(3).numpy()
        roots_np = roots.numpy()
        ngh_sampler.sample(roots_np, ts_np)
        ret = ngh_sampler.get_ret()
        gnn_mfgs_raws = cls.extract_raw_data(ret)
        batch_node_state_io = cls.analyze_need(gnn_mfgs_raws, sample_history)

        sample_ret = SampleReturn(
            ngh2root_mfgs_raw=gnn_mfgs_raws,
            root2ngh_mfg_raw=None,
            batch_node_state_io=batch_node_state_io,
        )
        return sample_ret
