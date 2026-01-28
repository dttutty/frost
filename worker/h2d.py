import logging
import threading
from queue import Queue as Q
from typing import Literal

import dgl
import torch
from dgl.heterograph import DGLBlock

# In utils.py
from buffer import Buffers
from schemas.datatypes.misc import DGLBlockMeta, Meta, Shared
from thread_base_with_stream import StreamThreadBase
from utils import _deduplicate_first_layer_src
from worker.assembler.dy_asm import prep_part_mem_n_box
from worker.assembler.st_asm import StAssembler


logger = logging.getLogger(__name__)


@torch.no_grad()
def collate_and_build_graph(
    mfg_meta_list: list[DGLBlockMeta], history: int
) -> list[list[DGLBlock]]:
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
    def chunked(iterable: list, n: int) -> list:
        return [iterable[i : i + n] for i in range(0, len(iterable), n)]

    grouped_blocks = chunked(blocks, history)

    grouped_blocks.reverse()

    return grouped_blocks


class H2D(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
        shared: Shared | None,
        merge_src: bool,
        in_q: Q[Meta | None | Literal["end_of_epoch"]],
        out_q: Q[tuple[Meta, torch.cuda.Event] | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
        h2d_sem: threading.Semaphore | None,
        history: int = 1,
        n_neg: int = 1,
    ) -> None:
        super().__init__(name="h2d", stop_ev=stop_ev)
        self.buf = buf
        self.nfeat = nfeat
        self.efeat = efeat
        self.shared = shared
        self.merge_src = merge_src
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.h2d_sem = h2d_sem
        self.history = history
        self.n_neg = n_neg  # TODO: make it configurable
        self.st_asm = StAssembler(shared_nfeat=nfeat, shared_efeat=efeat)

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            logger.debug("H2D: acquired semaphore, starting H2D operation")

            meta = self.in_q.get(block=True)
            if meta is None:
                self.out_q.put(None, block=True)
                break
            elif meta == "end_of_epoch":
                self.out_q.put("end_of_epoch", block=True)
                continue
            # move pos_efeat to gpu
            if self.efeat is not None:
                if (
                    self.buf.pos_efeat_h2d is not None
                    and meta.pos_edge_id.numel() <= self.buf.pos_efeat_h2d.shape[0]
                ):
                    n = meta.pos_edge_id.numel()
                    buf_view = self.buf.pos_efeat_h2d[:n]
                    torch.index_select(self.efeat, 0, meta.pos_edge_id, out=buf_view)
                    meta.pos_efeat = buf_view.to("cuda", non_blocking=True)
                else:
                    meta.pos_efeat = self.efeat[meta.pos_edge_id].cuda()

            # make mfg from raw data
            if meta.n2r_meta is not None:
                meta.n2r_mfgs = collate_and_build_graph(meta.n2r_meta, self.history)

            # init x
            n_pos_edges = meta.pos_edge.shape[0]
            meta.labels = torch.cat(
                [
                    torch.ones(n_pos_edges, device="cuda"),
                    torch.zeros(n_pos_edges * self.n_neg, device="cuda"),
                ],
            )

            # move plan to gpu
            if meta.plan_set is not None:
                if meta.plan_set.mem is not None and meta.plan_set.box is None:
                    meta.plan_set.box = meta.plan_set.mem
                meta.plan_set = meta.plan_set

            # move nfeat and efeat to gpu
            if self.nfeat is not None or self.efeat is not None:
                # prepare if has n_feat
                if meta.plan_set is not None and meta.plan_set.nfeat is not None:
                    meta.nfeat = self.st_asm.prepare_nfeat_cache(
                        nfeat_plan=meta.plan_set.nfeat, pin_nfeat_buf=self.buf.nfeat_h2d
                    )

                # can run no matter has nfeat or not
                if meta.n2r_mfgs is not None:
                    meta.n2r_mfgs = self.st_asm.fill_mfg_with_efeat_n_nfeat(
                        gnn_mfgs=meta.n2r_mfgs,
                        nfeat=meta.nfeat,
                        pin_efeat_buf=self.buf.efeat_h2d,
                    )

                # mfg version do not need nfeat output
                if meta.n2r_mfgs is not None:
                    del meta.nfeat  # mfg版本不需要nfeat输出

            # move mem and box to gpu
            if self.shared is not None:
                assert self.buf.mem_h2d is not None
                assert self.buf.box_h2d is not None
                assert self.h2d_sem is not None
                self.h2d_sem.acquire()
                self.h2d_sem.acquire()
                assert meta.plan_set is not None
                assert meta.plan_set.mem is not None
                assert meta.plan_set.box is not None
                meta.mem_n_box = prep_part_mem_n_box(
                    self.shared,
                    meta.plan_set.mem,
                    meta.plan_set.box,
                    pin_mem_buf=self.buf.mem_h2d,
                    pin_box_buf=self.buf.box_h2d,
                )
            if meta.plan_set is not None:
                if meta.plan_set.mem is not None and meta.plan_set.box is None:
                    meta.plan_set.box = meta.plan_set.mem
                # 使用 non_blocking 即可，无需 buffer
                meta.plan_set = meta.plan_set.cuda()

            # even there is no next batch, we still need plan_set for writeback
            if self.merge_src:
                assert meta.n2r_mfgs is not None
                meta.n2r_mfgs[0] = _deduplicate_first_layer_src(meta.n2r_mfgs[0])

            # move gnn_mfgs to gpu
            if meta.n2r_mfgs is not None:
                for blocks in meta.n2r_mfgs:
                    for block in blocks:
                        block = block.to("cuda:0")

            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            self.out_q.put((meta, ready_event), block=True)
