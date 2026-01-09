import threading
import traceback  # 记得导入

import torch
import torch.distributed as dist

from schemas.datatypes import Delta, X


class Assembler(threading.Thread):
    def __init__(self, xpart_q, delta_q, xfull_q, stop_ev):
        super().__init__(daemon=True)
        self.xpart_q = xpart_q
        self.delta_q = delta_q
        self.xfull_q = xfull_q
        self.stop_ev = stop_ev
        self.stream = torch.cuda.Stream()

    def run(self):
        try:  # 【新增】包裹整个 run 逻辑
            with torch.cuda.stream(self.stream), torch.no_grad():
                while not self.stop_ev.is_set():
                    # 1) 尽量都读一读（非阻塞/短超时），避免某一边饿死
                    meta, x_part, plan_set, ready_event = self.xpart_q.get(block=True)
                    print("Assembler: got x_part", meta.macro_bid)

                    delta = self.delta_q.get(block=True)
                    print("Assembler: got delta", meta.macro_bid)

                    # 2) 找到可组装的 bid

                    torch.cuda.current_stream().wait_event(ready_event)

                    x_full = self._assemble_data(x_part, delta)  # 你的 _assemble_data
                    sorted_x = x_full.sort_by_nid()  # 你的 sort

                    # 3) 把结果交给主线程（主线程取时再 wait_stream）
                    self.xfull_q.put(
                        (meta, plan_set, sorted_x, self.stream), block=True
                    )
                    print("Assembler: put x_full", meta.macro_bid)
        except Exception:
            print(f"!!! Assembler Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()  # 打印详细报错
            self.stop_ev.set()  # 通知其他线程退出
            # 可选：强制杀掉进程防止死锁等待
            import os

            os._exit(1)

    @staticmethod
    def _assemble_data(x: X, delta: Delta | None) -> X:
        if delta is None:
            return x

        ids = x.nfeat.id

        if delta.mem:
            assert x.mem
            x.mem.data = torch.cat((x.mem.data, delta.mem.data), dim=0)
            x.mem.id = torch.cat((x.mem.id, delta.mem.id), dim=0)
            x.mem.ts = torch.cat((x.mem.ts, delta.mem.ts), dim=0)
            sorted_nid, idx = torch.sort(x.mem.id)
            x.mem.data = x.mem.data[idx]
            x.mem.ts = x.mem.ts[idx]
            x.mem.id = sorted_nid
            idx = x.mem.id

            has_dup = torch.any(ids[1:] == ids[:-1])
            assert not has_dup, "Duplicate node IDs found in nfeat after merging."

        if delta.box:
            assert x.box
            x.box.data = torch.cat((x.box.data, delta.box.data), dim=0)
            x.box.id = torch.cat((x.box.id, delta.box.id), dim=0)
            x.box.ts = torch.cat((x.box.ts, delta.box.ts), dim=0)
            sorted_nid, idx = torch.sort(x.box.id)
            x.box.data = x.box.data[idx]
            x.box.ts = x.box.ts[idx]
            x.box.id = sorted_nid
            idx = x.box.id
            has_dup = torch.any(ids[1:] == ids[:-1])
            assert not has_dup, "Duplicate node IDs found in nfeat after merging."

        return x
