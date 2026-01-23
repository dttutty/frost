import copy
import logging

import torch
import torch.distributed as dist

from schemas.datatypes.env import Env
from thread_base_with_stream import StreamThreadBase


logger = logging.getLogger(__name__)


class HostWorkerSyncer(StreamThreadBase):
    def __init__(self, env: Env, gloo, meta_q, stop_ev, cmd_q):
        super().__init__(name="receiver")
        self.env: Env = env
        self.gloo = gloo
        self.meta_q = meta_q
        self.cmd_q = cmd_q  # 新增一个属性用于存放命令队列
        self.stop_ev = stop_ev

    @torch.no_grad()
    def logic(self):
        while not self.stop_ev.is_set():
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
