import copy
import threading
import traceback

import torch
import torch.distributed as dist

from schemas.datatypes.env import Env


class HostWorkerSyncer(threading.Thread):
    def __init__(self, env: Env, gloo, meta_q, stop_ev, cmd_q):
        super().__init__(daemon=True, name="receiver")
        self.env: Env = env
        self.gloo = gloo
        self.meta_q = meta_q
        self.stream = torch.cuda.Stream()
        self.stop_ev = stop_ev
        self.cmd_q = cmd_q  # 新增一个属性用于存放命令队列

    def logic(self):
        next_meta_buf = [None]

        # 这一步是阻塞的，直到 host 发来数据
        dist.scatter_object_list(
            next_meta_buf, None, src=self.env.host_rank, group=self.gloo
        )

        next_meta = next_meta_buf[0]

        # 【重要】需要一个退出机制
        # 约定：如果 host 发来 None 或者特殊的结束信号，就退出循环
        if next_meta is None:
            print(f"Rank {self.env.local_rank}: Received stop. Exiting receiver.")
            break
        elif isinstance(next_meta, str):
            self.cmd_q.put(next_meta)
            continue
        else:
            self.meta_q.put(copy.deepcopy(next_meta), block=True)

    def run(self):
        while not self.stop_ev.is_set():
            try:
                """这是在该线程中一直跑的循环"""
                with torch.cuda.stream(self.stream), torch.no_grad():
                    self.logic()
            except Exception:
                # 【修改】这里原来写的是 Assembler Thread Crashed，改为 Receiver
                print(f"!!! Receiver Thread Crashed on Rank {dist.get_rank()} !!!")
                traceback.print_exc()
                self.stop_ev.set()
                import os

                os._exit(1)
