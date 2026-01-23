import logging

import torch
from dgl.heterograph import DGLBlock

from schemas.datatypes.misc import NFeatNid
from schemas.datatypes.plan import TensorStPlan


logger = logging.getLogger(__name__)


class StAssembler:
    """
    静态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    nfeat_delta: NFeatNid | None = None

    def __init__(
        self,
        shared_nfeat: torch.Tensor | None,
        shared_efeat: torch.Tensor | None,
    ) -> None:
        self.nfeat_delta = None
        self.shared_nfeat = shared_nfeat
        self.shared_efeat = shared_efeat
        super().__init__()

    def fill_block_with_efeat(
        self, gnn_mfgs: list[list[DGLBlock]], efeat: torch.Tensor
    ) -> list[list[DGLBlock]]:
        for blocks in gnn_mfgs:
            for block in blocks:
                global_eid: torch.Tensor = block.edata["ID"]
                block.edata["efeat"] = efeat[global_eid]
        return gnn_mfgs

    def fill_block_with_nfeat(
        self, gnn_mfgs_0: list[DGLBlock], nfeat: NFeatNid
    ) -> list[DGLBlock]:
        assert nfeat.is_sorted  # make sure nfeat.nid is sorted

        for block in gnn_mfgs_0:
            global_nid: torch.Tensor = block.srcdata["ID"]
            permute_nidx = torch.searchsorted(nfeat.nid, global_nid)
            block.srcdata["nfeat"] = nfeat.nfeat[permute_nidx]  # 这里放memory

        return gnn_mfgs_0

    def prepare_nfeat_cache(
        self,
        nfeat_plan: TensorStPlan,
    ) -> NFeatNid | None:
        if self.shared_nfeat is not None:
            to_cache, fetch = nfeat_plan.to_cache, nfeat_plan.fetch
            nfeat = NFeatNid(nid=fetch, nfeat=self.shared_nfeat[fetch])
            if self.nfeat_delta is not None:
                nfeat = nfeat.merge(self.nfeat_delta)

            nfeat = nfeat.sort_by_nid()  # 你的 sort

            if to_cache.numel() > 0:
                self.nfeat_delta = nfeat.filter_by_nid(to_cache)
            else:
                self.nfeat_delta = None
        return nfeat

    def fill_mfg_with_efeat_n_nfeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        nfeat: NFeatNid | None,
    ) -> list[list[DGLBlock]]:
        # 这一部分是负责nfeat重用的,不管是mfg版本还是no_mfg版本,只要有nfeat都要做

        if self.shared_nfeat is not None:
            assert nfeat is not None
            gnn_mfgs[0] = self.fill_block_with_nfeat(gnn_mfgs[0], nfeat)
        if self.shared_efeat is not None:
            gnn_mfgs = self.fill_block_with_efeat(gnn_mfgs, self.shared_efeat)

        return gnn_mfgs
