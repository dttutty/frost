import logging

import torch

from frost.datatypes.access import Access
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


class APANAnalyzer:
    def __init__(self, n_neg: int):
        self.n_neg = n_neg  # from hydra config

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
    ):
        assert fanout is not None, "APAN requires fanout for neighbor sampling"
        assert len(fanout) == 1, "APAN only supports 1-hop sampling"
        n_src = src.shape[0]
        ngh_start = n_src * 3
        all_nid = all_nid.clone()  # 克隆一份，避免修改原始输入
        pos_ngh = all_nid[ngh_start : (ngh_start + fanout[0] * n_src * 2)]

        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)

        return Access(
            mem_need=all_nodes_set,
            mem_update=pos_root,
            box_need=all_nodes_set,
            box_update=pos_root + TensorSet(pos_ngh),
            nfeat_need=all_nodes_set,
        )


class TGNAnalyzer:
    def __init__(self, n_neg: int):
        self.n_neg = n_neg  # from hydra config

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
    ):
        all_nid = all_nid.clone()  # 克隆一份，避免修改原始输入
        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)

        return Access(
            mem_need=all_nodes_set,
            mem_update=pos_root,
            box_need=all_nodes_set,
            box_update=pos_root,
            nfeat_need=all_nodes_set,
        )


class JODIEAnalyzer:
    def __init__(self, n_neg: int):
        self.n_neg = n_neg  # from hydra config

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
    ):
        all_nid = all_nid.clone()  # 克隆一份，避免修改原始输入
        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)  # this includes src, dst, and neg_dst

        return Access(
            mem_need=all_nodes_set,
            mem_update=pos_root,
            box_need=all_nodes_set,
            box_update=pos_root,
            nfeat_need=all_nodes_set,
        )
