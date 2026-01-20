import os
import threading
import traceback  # 记得导入

import torch.distributed as dist


class ScatterSender(threading.Thread):
    def __init__(
        self,
        sem,
        n_gpus,
        comm,
        send_q,
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
                batch.plan_set = batch.plan_set.tensorfy()
                batch.plan_set = batch.plan_set.sort()
                if meta_2_scatter is None:
                    self.comm.scatter_send(None)
                    self.sem.release()
                else:
                    assert len(meta_2_scatter) == self.n_gpus
                    self.comm.scatter_send(meta_2_scatter)

        except Exception:
            print(f"!!! Assembler Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)
