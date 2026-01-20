import os
import threading
import traceback  # 记得导入
from queue import Queue  # 这是给多线程用的
from typing import List, Tuple

import torch.distributed as dist

from schemas import Meta
from schemas.datatypes import ScatterData
from schemas.plan import PlanSet


class Scatter(threading.Thread):
    def __init__(
        self,
        sem,
        n_gpus,
        comm,
        send_q: Queue[Tuple[List[Meta], List[PlanSet]]],
        mem_plan_out_q,
        mail_plan_out_q,
        nfeat_plan_out_q,
        stop_ev,
    ):
        super().__init__(daemon=True)
        self.sem = sem
        self.n_gpus = n_gpus
        self.comm = comm
        self.send_q = send_q
        self.mem_plan_out_q = mem_plan_out_q
        self.mail_plan_out_q = mail_plan_out_q
        self.nfeat_plan_out_q = nfeat_plan_out_q
        self.stop_ev = stop_ev

    def run(self):
        try:
            while not self.stop_ev.is_set():
                meta_2_scatter = self.send_q.get(block=True)
                if meta_2_scatter is None:
                    self.comm.scatter_send(None)
                    self.sem.release()
                else:
                    assert len(meta_2_scatter) == self.n_gpus
                    scatter_datas = []
                    mem_plans = mail_plans = nfeat_plans = None
                    if self.mem_plan_out_q is not None:
                        mem_plans = self.mem_plan_out_q.get(block=True)
                    if self.mail_plan_out_q is not None:
                        mail_plans = self.mail_plan_out_q.get(block=True)
                    if self.nfeat_plan_out_q is not None:
                        nfeat_plans = self.nfeat_plan_out_q.get(block=True)
                    for i in range(self.n_gpus):
                        planset = PlanSet(
                            mem=mem_plans[i] if mem_plans is not None else None,
                            box=mail_plans[i] if mail_plans is not None else None,
                            nfeat=nfeat_plans[i] if nfeat_plans is not None else None,
                        )
                        planset = planset.tensorfy()
                        planset = planset.sort()
                        scatter_data = ScatterData(
                            metas=meta_2_scatter[i],
                            planset=planset,
                        )
                    scatter_datas.append(scatter_data)

                    self.comm.scatter_send(scatter_datas)

        except Exception:
            print(f"!!! Scatter Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)
