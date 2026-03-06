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
        valid_all: torch.Tensor | None = None,
    ):
        assert valid_all is not None, "APAN requires valid_all for access counting"
        assert fanout is not None, "APAN requires fanout for neighbor sampling"
        assert len(fanout) == 1, "APAN only supports 1-hop sampling"
        n_src = src.shape[0]
        # Root layout: [src(n_src), pos_dst(n_src), neg_dst(n_neg*n_src)]
        # Neighbors (sample_4_neg=False): only src + pos_dst have neighbors
        ngh_start = n_src * (2 + self.n_neg)
        pos_ngh_raw = valid_all[ngh_start : (ngh_start + fanout[0] * n_src * 2)]
        pos_ngh = pos_ngh_raw[pos_ngh_raw != -1]  # remove padding slots

        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            required_box=all_nodes_set,
            box_update=TensorSet(pos_ngh),
            required_nfeat=all_nodes_set,
        )


class NoMemAnalyzer:
    def __init__(self, n_neg: int):
        self.n_neg = n_neg  # from hydra config

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ):
        all_nid = all_nid.clone()  # 克隆一份，避免修改原始输入
        all_nodes_set = TensorSet(all_nid)

        return Access(
            required_nfeat=all_nodes_set,
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
        valid_all: torch.Tensor | None = None,
    ):
        all_nid = all_nid.clone()  # 克隆一份，避免修改原始输入
        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            required_box=all_nodes_set,
            box_update=pos_root,
            required_nfeat=all_nodes_set,
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
        valid_all: torch.Tensor | None = None,
    ):
        all_nid = all_nid.clone()  # 克隆一份，避免修改原始输入
        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)  # this includes src, dst, and neg_dst

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            required_box=all_nodes_set,
            box_update=pos_root,
            required_nfeat=all_nodes_set,
        )
