import torch

from schemas.datatypes.misc import RootStX
from schemas.datatypes.plan import TensorStPlan


class RootXStAssembler:
    """
    只能填充nfeat,这个是为了identity gnn设计的
    This can only fill nfeat, which is designed for identity GNNs.
    """

    root_x_DyX: RootStX | None = None

    def __init__(self, dram_data: torch.Tensor):
        self.root_x_DyX = None
        self.dram_data = dram_data

    def assemble(self, plan: TensorStPlan, root_nid) -> RootStX:
        to_cache, fetch = plan.to_cache, plan.fetch
        root_x_part = RootStX(nid=fetch, nfeat=self.dram_data[fetch])
        if self.root_x_DyX is not None:
            root_x = root_x_part.merge_with_another_rootx(self.root_x_DyX)

        else:
            root_x = root_x_part

        root_x = root_x.unzip_by_root_nid(root_nid)

        if to_cache.numel() > 0:
            self.root_x_DyX = root_x.filter_by_root_nid(to_cache)
        else:
            self.root_x_DyX = None
        return root_x
