from typing import List

import torch
from dgl.heterograph import DGLBlock

from schemas.datatypes.misc import Meta, NFeatNid
from schemas.datatypes.plan import TensorStPlan


class StAssembler:
    """
    静态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    nfeat_delta: NFeatNid | None = None

    def __init__(
        self,
        shared_nfeat: torch.Tensor | None,
        shared_efeat: torch.Tensor | None,
    ):
        self.nfeat_delta = None
        self.shared_nfeat = shared_nfeat
        self.shared_efeat = shared_efeat
        super().__init__()

    def fill_block_with_efeat(
        self, gnn_mfgs: List[List[DGLBlock]], efeat: torch.Tensor
    ):
        for blocks in gnn_mfgs:
            for block in blocks:
                global_eid: torch.Tensor = block.edata["ID"]
                block.edata["efeat"] = efeat[global_eid]

    def fill_block_with_nfeat(self, gnn_mfgs_0: List[DGLBlock], nfeat: NFeatNid):
        assert nfeat.is_sorted  # make sure nfeat.nid is sorted

        if nfeat is not None:
            for block in gnn_mfgs_0:
                global_nid: torch.Tensor = block.srcdata["ID"]
                permute_nidx = torch.searchsorted(nfeat.nid, global_nid)
                block.srcdata["nfeat"] = nfeat.nfeat[permute_nidx]  # 这里放memory

    def fill(self, meta: Meta, plan: TensorStPlan | None = None) -> Meta:
        assert meta.x is not None
        # 这一部分是负责nfeat重用的,不管是mfg版本还是no_mfg版本,只要有nfeat都要做
        if self.shared_nfeat is not None:
            assert plan is not None
            to_cache, fetch = plan.to_cache, plan.fetch
            nfeat = NFeatNid(nid=fetch, nfeat=self.shared_nfeat[fetch])
            if self.nfeat_delta is not None:
                nfeat = nfeat.merge(self.nfeat_delta)

            nfeat = nfeat.sort_by_nid()  # 你的 sort

            if to_cache.numel() > 0:
                self.nfeat_delta = nfeat.filter_by_nid(to_cache)
            else:
                self.nfeat_delta = None

        if meta.gnn_mfgs is not None:
            if self.shared_nfeat is not None:
                self.fill_block_with_nfeat(meta.gnn_mfgs[0], nfeat)
            if self.shared_efeat is not None:
                self.fill_block_with_efeat(meta.gnn_mfgs, self.shared_efeat)
        else:
            if self.shared_nfeat is not None:
                meta.x.nfeat = nfeat
                # 其实对于mfg版本,不需要x里面有nfeat的(但是x需要有mem和box)
                # 对于非mfg版本,才需要x里面有nfeat的
        return meta
