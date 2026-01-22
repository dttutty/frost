import os
import threading
import traceback  # 记得导入

import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock

from schemas.datatypes.data_to_sync import DyX
from schemas.datatypes.misc import DyX, RootDyX, StX


class MFGStAssembler(threading.Thread):
    """
    动态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    def __init__(self):
        super().__init__(daemon=True)

        self.stream = torch.cuda.Stream()

    def mailbox_assemble(self, new_mails, old_mailbox):
        if old_mailbox is None:
            return new_mails
        else:
            # 合并新旧 mailbox，假设按时间戳排序
            combined_data = torch.cat((old_mailbox.data, new_mails.data), dim=0)
            combined_ts = torch.cat((old_mailbox.ts, new_mails.ts), dim=0)
            combined_id = torch.cat((old_mailbox.id, new_mails.id), dim=0)
            # 这里可以添加排序逻辑，确保按时间戳排序
            return DyX.Box(data=combined_data, ts=combined_ts, id=combined_id)

    def fill_block(self, block: DGLBlock, x: StX):
        # print(f"fill_blocks block.srcdata['pos']: {block.srcdata['pos']}")
        assert x.is_sorted  # make sure x.nid is sorted
        global_nid: torch.Tensor = block.srcdata["ID"]
        # step 1: find index of each x.nid in _ID
        permute_nidx = torch.searchsorted(x.nid, global_nid)
        # need to make sure _ID is subset of x.nid
        # assert torch.equal(x.nid[permute_nidx], global_nid), (
        #     "Data missing: Block IDs not found in Assembler X data"
        # )
        if x.mem is not None:
            block.srcdata["mem"] = x.mem.data[permute_nidx]  # 这里放memory
            block.srcdata["mem_ts"] = x.mem.ts[permute_nidx]
        if x.box is not None:
            block.srcdata["box"] = x.box.data[permute_nidx]
            block.srcdata["box_ts"] = x.box.ts[permute_nidx]

    def fill_root_x(self, root_info, x: StX) -> StX:
        assert x.is_sorted  # make sure x.nid is sorted
        global_nid: torch.Tensor = root_info.nid
        permute_nidx = torch.searchsorted(x.nid, global_nid)
        root_x = RootDyX()
        if x.mem is not None:
            root_x.mem = x.mem.data[permute_nidx]
            root_x.mem_ts = x.mem.ts[permute_nidx]

        return root_x

    def logic(self):
        meta, x_part, plan_set, ready_event = self.xpart_q.get(block=True)

        DyX = self.DyX_q.get(block=True)

        torch.cuda.current_stream().wait_event(ready_event)

        x_full = self._assemble_dy_data(x_part, DyX)  # 你的 _assemble_dy_data
        sorted_x = x_full.sort_by_nid()  # 你的 sort
        if meta.gnn_mfgs is not None:
            for block in meta.gnn_mfgs[0]:
                self.fill_block(block, sorted_x)
        else:
            meta.root_x = self.fill_root_x(meta.root_info, sorted_x)
        # 3) 把结果交给主线程（主线程取时再 wait_stream）
        self.xfull_q.put((meta, plan_set, sorted_x, self.stream), block=True)

    def run(self):
        try:  # 【新增】包裹整个 run 逻辑
            with torch.cuda.stream(self.stream), torch.no_grad():
                while not self.stop_ev.is_set():
                    self.logic()
        except Exception:
            print(f"!!! Assembler Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)

    @staticmethod
    def _assemble_dy_data(x: DyX, DyX: DyX | None) -> DyX:
        if DyX is None:
            return x

        if DyX.mem:
            assert x.mem
            x.mem.data = torch.cat((x.mem.data, DyX.mem.data), dim=0)
            x.mem.id = torch.cat((x.mem.id, DyX.mem.id), dim=0)
            x.mem.ts = torch.cat((x.mem.ts, DyX.mem.ts), dim=0)

        if DyX.box:
            assert x.box
            x.box.data = torch.cat((x.box.data, DyX.box.data), dim=0)
            x.box.id = torch.cat((x.box.id, DyX.box.id), dim=0)
            x.box.ts = torch.cat((x.box.ts, DyX.box.ts), dim=0)

        return x
