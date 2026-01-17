import os
import threading
import traceback  # 记得导入

import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock

from schemas.datatypes import Delta, X


class Assembler(threading.Thread):
    def __init__(self, xpart_q, delta_q, xfull_q, stop_ev):
        super().__init__(daemon=True)
        self.xpart_q = xpart_q
        self.delta_q = delta_q
        self.xfull_q = xfull_q
        self.stop_ev = stop_ev
        self.stream = torch.cuda.Stream()

    def fill_block(self, block: DGLBlock, x: X):
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

    def run(self):
        try:  # 【新增】包裹整个 run 逻辑
            with torch.cuda.stream(self.stream), torch.no_grad():
                while not self.stop_ev.is_set():
                    meta, x_part, plan_set, ready_event = self.xpart_q.get(block=True)

                    delta = self.delta_q.get(block=True)

                    torch.cuda.current_stream().wait_event(ready_event)

                    x_full = self._assemble_data(x_part, delta)  # 你的 _assemble_data
                    sorted_x = x_full.sort_by_nid()  # 你的 sort
                    for block in meta.blockss[0]:
                        self.fill_block(block, sorted_x)
                    # 3) 把结果交给主线程（主线程取时再 wait_stream）
                    self.xfull_q.put(
                        (meta, plan_set, sorted_x, self.stream), block=True
                    )
        except Exception:
            print(f"!!! Assembler Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)

    @staticmethod
    def _assemble_data(x: X, delta: Delta | None) -> X:
        if delta is None:
            return x

        if delta.mem:
            assert x.mem
            x.mem.data = torch.cat((x.mem.data, delta.mem.data), dim=0)
            x.mem.id = torch.cat((x.mem.id, delta.mem.id), dim=0)
            x.mem.ts = torch.cat((x.mem.ts, delta.mem.ts), dim=0)

        if delta.box:
            assert x.box
            x.box.data = torch.cat((x.box.data, delta.box.data), dim=0)
            x.box.id = torch.cat((x.box.id, delta.box.id), dim=0)
            x.box.ts = torch.cat((x.box.ts, delta.box.ts), dim=0)

        return x
