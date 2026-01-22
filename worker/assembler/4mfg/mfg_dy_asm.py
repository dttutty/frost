import threading

import torch
from dgl.heterograph import DGLBlock

from schemas.datatypes.misc import DyX


class MFGDyAssembler(threading.Thread):
    """
    动态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    def __init__(self):
        super().__init__(daemon=True)

    def fill_mem_mail(self, block: DGLBlock, x: DyX):
        assert x.is_sorted  # make sure x.nid is sorted
        global_nid: torch.Tensor = block.srcdata["ID"]
        permute_nidx = torch.searchsorted(x.nid, global_nid)

        if x.mem is not None:
            block.srcdata["mem"] = x.mem.data[permute_nidx]  # 这里放memory
            block.srcdata["mem_ts"] = x.mem.ts[permute_nidx]
        if x.box is not None:
            block.srcdata["box"] = x.box.data[permute_nidx]
            block.srcdata["box_ts"] = x.box.ts[permute_nidx]

    def logic(self, out_layer_mfgs, x_part: DyX, DyX: DyX | None):
        assert x_part.mem is not None
        assert x_part.box is not None
        if DyX is not None:
            x_part.mem = x_part.mem.merge_with_another_mem(DyX.mem)
            x_part.box = x_part.box.merge_with_another_box(DyX.box)

        x_full = x_part
        sorted_x = x_full.sort_by_nid()  # 你的 sort
        for block in out_layer_mfgs:
            self.fill_mem_mail(block, sorted_x)

        return out_layer_mfgs
