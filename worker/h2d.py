import queue
import threading
import traceback

import torch
import torch.distributed as dist

from schemas.datatypes import (
    BoxTsNid,
    EdgeFeat,
    MemTsNid,
    Meta,
    NodeFeat,
    Shared,
    TensorPlanSet,
    X,
)
from utils import _deduplicate_first_layer_src, blocks_to_cuda


class H2D(threading.Thread):
    def __init__(
        self,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
        shared: Shared | None,
        merge_src: bool,
        meta_q: queue.Queue,
        xpart_q: queue.Queue,
        stop_ev: threading.Event,
        h2d_sem: threading.Semaphore,
    ):
        super().__init__(daemon=True, name="h2d")
        self.nfeat = nfeat
        self.efeat = efeat
        self.shared = shared
        self.merge_src = merge_src
        self.meta_q = meta_q
        self.xpart_q = xpart_q
        self.stop_ev = stop_ev
        self.h2d_sem = h2d_sem
        self.stream = torch.cuda.Stream()

    def run(self):
        try:
            with torch.cuda.stream(self.stream), torch.no_grad():
                while not self.stop_ev.is_set():
                    self.h2d_sem.acquire()
                    self.h2d_sem.acquire()
                    print("H2D: acquired semaphore, starting H2D operation")
                    meta: Meta = self.meta_q.get(block=True)
                    x = X()
                    plan_set = meta.plan_set
                    assert plan_set is not None, "plan_set should not be None after H2D"

                    if self.efeat is not None:
                        x.pos_efeat = self.efeat[meta.pos_edge_id].cuda()

                    # move feat to gpu
                    if plan_set is not None:
                        assert type(plan_set) is TensorPlanSet
                        if plan_set.nfeat_prefetch is not None:
                            assert self.nfeat is not None
                            nid = plan_set.nfeat_prefetch
                            x.nfeat = NodeFeat(
                                id=nid, data=self.nfeat[plan_set.nfeat_prefetch]
                            ).cuda()
                        if plan_set.efeat_prefetch is not None:
                            assert self.efeat is not None
                            x.efeat = EdgeFeat(
                                id=plan_set.efeat_prefetch,
                                data=self.efeat[plan_set.efeat_prefetch],
                            ).cuda()
                        if self.shared is not None:
                            assert plan_set.box is not None
                            assert plan_set.mem is not None

                            mem_prefetch_nids = plan_set.mem.prefetch
                            x.mem = MemTsNid(
                                id=mem_prefetch_nids,
                                data=self.shared.mem[mem_prefetch_nids],
                                ts=self.shared.mem_ts[mem_prefetch_nids],
                            ).cuda()
                            mail_prefetch_nids = plan_set.box.prefetch
                            x.box = BoxTsNid(
                                id=mail_prefetch_nids,
                                data=self.shared.box[mail_prefetch_nids],
                                ts=self.shared.box_ts[mail_prefetch_nids],
                            ).cuda()
                        # move plan to gpu
                        # even there is no next batch, we still need plan_set for writeback
                        if self.merge_src:
                            meta.blockss[0] = _deduplicate_first_layer_src(
                                meta.blockss[0]
                            )

                        meta.blockss = blocks_to_cuda(meta.blockss)

                        if meta.mdt:
                            meta.mdt = meta.mdt.cuda()

                    assert plan_set is not None, "plan_set should not be None after H2D"

                    ready_event = torch.cuda.Event()
                    ready_event.record(torch.cuda.current_stream())
                    self.xpart_q.put((meta, x, plan_set, ready_event), block=True)

        except Exception:
            print(f"!!! Assembler Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()  # 打印详细报错
            self.stop_ev.set()  # 通知其他线程退出
            # 可选：强制杀掉进程防止死锁等待
            import os

            os._exit(1)
