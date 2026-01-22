import os
import queue
import threading
import traceback
from typing import List

import dgl
import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock

# In utils.py
from schemas.datatypes.misc import DGLBlockMeta, Meta, Shared, X
from utils import _deduplicate_first_layer_src
from worker.assembler.dy_asm import prep_before_dy_asm
from worker.assembler.st_asm import StAssembler


def collate_and_build_graph(
    mfg_meta_list: List[DGLBlockMeta], history: int
) -> List[List[DGLBlock]]:
    blocks = []

    for mfg_meta in mfg_meta_list:
        # 1. 在 Worker 进程组装 DGLBlock
        # 此时数据已经在 Worker 的内存空间了
        block = dgl.create_block(
            (mfg_meta.col, mfg_meta.row),
            num_src_nodes=mfg_meta.dim_in,
            num_dst_nodes=mfg_meta.dim_out,
        )

        # num_dst_nodes = tgb.dim_out()少
        # num_src_nodes = tgb.dim_in()多

        # 2. 绑定特征
        num_roots = mfg_meta.dim_out
        all_nodes = mfg_meta.nodes
        block.srcdata["ID"] = all_nodes
        block.dstdata["ID"] = all_nodes[:num_roots]
        block.srcdata["ts"] = mfg_meta.ts
        block.dstdata["ts"] = mfg_meta.ts[:num_roots]
        block.edata["dt"] = mfg_meta.dts[num_roots:]
        block.edata["ID"] = mfg_meta.eid
        # 3. 统一上 GPU (由 Worker 决定何时上 GPU)
        # block = block.to("cuda:0")
        blocks.append(block)

    # 4. 处理历史分组逻辑
    grouped_blocks = list(map(list, zip(*[iter(blocks)] * history)))
    grouped_blocks.reverse()

    return grouped_blocks


class H2D(threading.Thread):
    def __init__(
        self,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
        shared: Shared | None,
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
        self.meta_q = meta_q
        self.destination_q = destination_q
        self.stop_ev = stop_ev
        self.h2d_sem = h2d_sem
        self.history = history
        self.st_asm = StAssembler(shared_nfeat=nfeat, shared_efeat=efeat)

        self.stream = torch.cuda.Stream()

    def logic(self):
        if self.h2d_sem is not None:
            self.h2d_sem.acquire()
            self.h2d_sem.acquire()
        print("H2D: acquired semaphore, starting H2D operation")

        meta: Meta = self.meta_q.get(block=True)

        # move pos_efeat to gpu
        if self.efeat is not None:
            meta.pos_efeat = self.efeat[meta.pos_edge_id].cuda()

        # make mfg from raw data
        if meta.gnn_mfgs_raw is not None:
            meta.gnn_mfgs = collate_and_build_graph(meta.gnn_mfgs_raw, self.history)

        # init x
        meta.x = X()

        # move plan to gpu
        if meta.plan_set is not None:
            meta.plan_set = meta.plan_set.cuda()

        # move nfeat and efeat to gpu
        if self.nfeat is not None or self.efeat is not None:
            if meta.plan_set is not None and meta.plan_set.nfeat is not None:
                self.st_asm.fill(meta, meta.plan_set.nfeat)
            else:
                self.st_asm.fill(meta)

        if self.shared is not None:
            assert meta.plan_set is not None
            meta.x.mem, meta.x.box = prep_before_dy_asm(self.shared, meta.plan_set)

        # even there is no next batch, we still need plan_set for writeback
        if self.merge_src:
            assert meta.gnn_mfgs is not None
            meta.gnn_mfgs[0] = _deduplicate_first_layer_src(meta.gnn_mfgs[0])

        if meta.gnn_mfgs is not None:
            for blocks in meta.gnn_mfgs:
                for block in blocks:
                    block = block.to("cuda:0")

        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())
        self.destination_q.put((meta, ready_event), block=True)

    def run(self):
        try:
            with torch.cuda.stream(self.stream), torch.no_grad():
                while not self.stop_ev.is_set():
                    self.logic()

        except Exception:
            print(f"!!! H2D Thread Crashed on Rank {dist.get_rank()} !!!")
            traceback.print_exc()  # 打印详细报错
            self.stop_ev.set()  # 通知其他线程退出

            os._exit(1)
