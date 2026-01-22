import os
import threading
import traceback  # 记得导入
from queue import Queue

import torch.distributed as dist

from schemas.alias import MetaBundle
from schemas.datatypes.misc import Meta


class MetaBatcher(threading.Thread):
    def __init__(
        self,
        n_gpus: int,
        in_q: Queue[Meta | None],
        out_q: Queue[MetaBundle | None],
        stop_ev: threading.Event,
    ):
        super().__init__(daemon=True)
        self.n_gpus = n_gpus
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev

    def logic(self):
        meta_list = []
        while len(meta_list) < self.n_gpus:
            meta = self.in_q.get(block=True)
            if meta is None:
                self.out_q.put(None)
            else:
                meta_list.append(meta)
        self.out_q.put(meta_list)

    def run(self):
        try:
            while not self.stop_ev.is_set():
                self.logic()

        except Exception:
            print(f"!!! MetaBatcher Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)
