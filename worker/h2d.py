import queue
import threading
import traceback

import torch
import torch.distributed as dist

from schemas import (
    BoxTsNid,
    MemTsNid,
    Meta,
    Shared,
    TensorPlanSet,
    X,
)
from utils import _deduplicate_first_layer_src


class H2D(threading.Thread):
    def __init__(
        self,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
        shared: Shared | None,
        gnn_arch: str,
        merge_src: bool,
        meta_q: queue.Queue[Meta],
        destination_q: queue.Queue,
        stop_ev: threading.Event,
        h2d_sem: threading.Semaphore | None,
    ):
        super().__init__(daemon=True, name="h2d")
        self.nfeat = nfeat
        self.efeat = efeat
        self.shared = shared
        self.merge_src = merge_src
        self.gnn_arch = gnn_arch
        self.meta_q = meta_q
        self.destination_q = destination_q
        self.stop_ev = stop_ev
        self.h2d_sem = h2d_sem
        self.stream = torch.cuda.Stream()

    def run(self):
        try:
            with torch.cuda.stream(self.stream), torch.no_grad():
                while not self.stop_ev.is_set():
                    if self.h2d_sem is not None:
                        self.h2d_sem.acquire()
                        self.h2d_sem.acquire()
                    print("H2D: acquired semaphore, starting H2D operation")
                    meta: Meta = self.meta_q.get(block=True)

                    x = X()
                    plan_set = meta.plan_set

                    if self.efeat is not None:
                        x.pos_efeat = self.efeat[meta.pos_edge_id].cuda()
                        if self.gnn_arch != "identity":
                            for blocks in meta.            gnn_mfgs_raw=gnn_mfgs_raw,:
                                for block in blocks:
                                    eid = block.edata["ID"]
                                    block.edata["efeat"] = self.efeat[eid]

                    if self.nfeat is not None:
                        nfeat = self.nfeat[meta.unique_nodes]
                        id_list = meta.unique_nodes
                        for block in meta.            gnn_mfgs_raw=gnn_mfgs_raw,[0]:
                            nid = block.srcdata["ID"]
                            node_idx = torch.searchsorted(id_list, nid)
                            block.srcdata["nfeat"] = nfeat[node_idx]

                    # move feat to gpu
                    if plan_set is not None:
                        assert type(plan_set) is TensorPlanSet
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
                            meta.            gnn_mfgs_raw=gnn_mfgs_raw,[0] = _deduplicate_first_layer_src(
                                meta.            gnn_mfg=gnn_mfgs_raw,[0]
                            )

                    if meta.mdt:
                        meta.mdt = meta.mdt.cuda()
                    meta.            gnn_mfgs_raw=gnn_mfgs_raw, = [
                        [block.to("cuda:0") for block in blocks]
                        for blocks in meta.            gnn_mfgs_raw=gnn_mfgs_raw,
                    ]

                    print("meta.blockss[0][0].device:", meta.            gnn_mfg=gnn_mfgs_raw,[0][0].device)

                    ready_event = torch.cuda.Event()
                    ready_event.record(torch.cuda.current_stream())
                    self.destination_q.put((meta, x, plan_set, ready_event), block=True)

        except Exception:
            print(f"!!! Assembler Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()  # 打印详细报错
            self.stop_ev.set()  # 通知其他线程退出
            # 可选：强制杀掉进程防止死锁等待
            import os

            os._exit(1)
