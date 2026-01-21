import queue
import threading
import traceback
from dataclasses import dataclass

import torch
import torch.distributed as dist

from schemas.datatypes.data_to_sync import (
    BoxTsNid,
    MemTsNid,
)
from schemas.datatypes.misc import (
    Meta,
    RootX,
    Shared,
    X,
)
from schemas.datatypes.plan import TensorPlanSet
from utils import _deduplicate_first_layer_src, collate_and_build_graph


@dataclass(slots=True, kw_only=True)
class StaticCache:
    nid: torch.Tensor | None
    nfeat: torch.Tensor | None


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
        history: int = 1,
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
        self.history = history

        self.stream = torch.cuda.Stream()

    def logic(self):
        if self.h2d_sem is not None:
            self.h2d_sem.acquire()
            self.h2d_sem.acquire()
        print("H2D: acquired semaphore, starting H2D operation")
        x = X()
        meta: Meta = self.meta_q.get(block=True)
        plan_set = meta.plan_set

        # make mfg from raw data
        if meta.gnn_mfgs_raw is not None:
            meta.gnn_mfgs = collate_and_build_graph(meta.gnn_mfgs_raw, self.history)

            # we don't reuse efeat always
            if self.efeat is not None:
                x.pos_efeat = self.efeat[meta.pos_edge_id].cuda()
                if self.gnn_arch != "identity":
                    for blocks in meta.gnn_mfgs[0]:
                        for block in blocks:
                            eid = block.edata["ID"]
                            block.edata["efeat"] = self.efeat[eid]

            # this version not reuse nfeat
            if self.nfeat is not None:
                for block in meta.gnn_mfgs[0]:
                    # inner node appear in outer gnn, so we only do this for outer gnn
                    nid = block.srcdata["ID"]
                    block.srcdata["nfeat"] = self.nfeat[nid]

        # move feat to gpu
        if plan_set is not None:
            assert type(plan_set) is TensorPlanSet
            if self.shared is not None:
                assert plan_set.box is not None
                assert plan_set.mem is not None

                mem_fetch_nids = plan_set.mem.fetch
                x.mem = MemTsNid(
                    id=mem_fetch_nids,
                    data=self.shared.mem[mem_fetch_nids],
                    ts=self.shared.mem_ts[mem_fetch_nids],
                ).cuda()
                mail_fetch_nids = plan_set.box.fetch
                x.box = BoxTsNid(
                    id=mail_fetch_nids,
                    data=self.shared.box[mail_fetch_nids],
                    ts=self.shared.box_ts[mail_fetch_nids],
                ).cuda()
            # move plan to gpu
            # even there is no next batch, we still need plan_set for writeback
            if self.merge_src:
                meta.gnn_mfgs[0] = _deduplicate_first_layer_src(meta.gnn_mfgs[0])
        else:
            meta.roots.x = RootX()
            if self.nfeat is not None:
                nid = meta.roots.nid
                meta.roots.x.nfeat = self.nfeat[nid].cuda()

        if meta.gnn_mfgs is not None:
            for blocks in meta.gnn_mfgs:
                for block in blocks:
                    block = block.to("cuda:0")

        print(
            "meta.gnn_mfgs[0][0].device:",
            meta.gnn_mfgs[0][0].device,
        )

        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())
        self.destination_q.put((meta, x, plan_set, ready_event), block=True)

    def run(self):
        try:
            with torch.cuda.stream(self.stream), torch.no_grad():
                while not self.stop_ev.is_set():
                    self.logic()

        except Exception:
            print(f"!!! H2D Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()  # 打印详细报错
            self.stop_ev.set()  # 通知其他线程退出
            # 可选：强制杀掉进程防止死锁等待
            import os

            os._exit(1)
