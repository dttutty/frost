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
        device: torch.device,
        history: int = 1,
        n_neg: int = 1,
    ) -> None:
        super().__init__(name="h2d", stop_ev=stop_ev)
        logger.debug(
            f"H2D: Initializing, history={history}, n_neg={n_neg}, merge_src={merge_src}"
        )
        self.buf = buf
        self.nfeat = nfeat
        self.efeat = efeat
        self.shared = shared
        self.device = device
        self.merge_src = merge_src
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.h2d_sem = h2d_sem
        self.history = history
        self.n_neg = n_neg  # TODO: make it configurable
        self.st_asm = StAssembler(
            shared_nfeat=nfeat, shared_efeat=efeat, device=self.device
        )

        logger.debug(
            f"H2D: nfeat={'enabled' if nfeat is not None else 'disabled'}, shape={nfeat.shape if nfeat is not None else None}"
        )
        logger.debug(
            f"H2D: efeat={'enabled' if efeat is not None else 'disabled'}, shape={efeat.shape if efeat is not None else None}"
        )
        logger.debug(f"H2D: shared={'enabled' if shared is not None else 'disabled'}")
        logger.debug(
            f"H2D: Buffers - pos_efeat_h2d={'enabled' if buf.pos_efeat_h2d is not None else 'disabled'}, "
            f"nfeat_h2d={'enabled' if buf.nfeat_h2d is not None else 'disabled'}, "
            f"efeat_h2d={'enabled' if buf.efeat_h2d is not None else 'disabled'}, "
            f"mem_h2d={'enabled' if buf.mem_h2d is not None else 'disabled'}, "
            f"box_h2d={'enabled' if buf.box_h2d is not None else 'disabled'}"
        )
        logger.debug("H2D: Initialized")

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0
        logger.debug("H2D: starting logic loop")

        while not self.stop_ev.is_set():
            logger.debug(
                f"H2D: waiting for meta from in_q (queue size: {self.in_q.qsize()})"
            )

            meta = self.in_q.get(block=True)
            logger.debug(f"H2D: received meta type: {type(meta).__name__}")

            if meta is None:
                logger.debug("H2D: received None, putting to out_q and exiting")
                self.out_q.put(None, block=True)
                logger.debug(f"H2D: Processed {batch_count} batches before exit")
                break
            elif meta == "end_of_epoch":
                logger.debug("H2D: received end_of_epoch, forwarding to out_q")
                self.out_q.put("end_of_epoch", block=True)
                continue

            batch_count += 1
            logger.debug(
                f"H2D: processing batch #{batch_count}, macro_bid={meta.m_state.cur_macro_of_epoch}"
            )

            if meta.plan_set is not None:
                if meta.plan_set.mem is not None and meta.plan_set.box is None:
                    logger.debug(
                        f"H2D: batch #{batch_count}, copying mem plan to box (final check)"
                    )
                    meta.plan_set.box = meta.plan_set.mem
                # 使用 non_blocking 即可，无需 buffer
                logger.debug(
                    f"H2D: batch #{batch_count}, transferring plan_set to cuda"
                )
                meta.plan_set = meta.plan_set.to(self.device, non_blocking=False)
                logger.debug(f"H2D: batch #{batch_count}, plan_set on GPU")

            # move pos_efeat to gpu
            if self.efeat is not None:
                logger.debug(
                    f"H2D: batch #{batch_count}, transferring pos_efeat, size={meta.pos_edge_id.numel()}"
                )
                if (
                    self.buf.pos_efeat_h2d is not None
                    and meta.pos_edge_id.numel() <= self.buf.pos_efeat_h2d.shape[0]
                ):
                    n = meta.pos_edge_id.numel()
                    logger.debug(
                        f"H2D: batch #{batch_count}, using pinned buffer for pos_efeat, n={n}, buffer capacity={self.buf.pos_efeat_h2d.shape[0]}"
                    )
                    buf_view = self.buf.pos_efeat_h2d[:n]
                    if __debug__:
                        # 1. 检查索引是否为空，防止 buf_view 的尺寸冲突警告
                        assert meta.pos_edge_id.numel() > 0, "pos_edge_id is empty"

                        # 2. 核心检查：索引是否越界 (关键！)
                        # 确保所有索引都在 [0, efeat.size(0)) 范围内
                        max_idx = meta.pos_edge_id.max().item()
                        min_idx = meta.pos_edge_id.min().item()
                        assert max_idx < self.efeat.size(0), (
                            f"Edge index {max_idx} out of bounds for efeat size {self.efeat.size(0)}"
                        )
                        assert min_idx >= 0, f"Negative edge index {min_idx} detected"

                        # 3. 检查设备一致性
                        assert self.efeat.device == meta.pos_edge_id.device, (
                            "Device mismatch between efeat and indices"
                        )
                    torch.index_select(self.efeat, 0, meta.pos_edge_id, out=buf_view)
                    meta.pos_efeat = buf_view.to(self.device, non_blocking=False)
                    logger.debug(
                        f"H2D: batch #{batch_count}, pos_efeat transferred via pinned buffer"
                    )
                else:
                    logger.debug(
                        f"H2D: batch #{batch_count}, direct cuda transfer for pos_efeat (buffer insufficient or disabled)"
                    )
                    meta.pos_efeat = self.efeat[meta.pos_edge_id].to(
                        self.device, non_blocking=False
                    )
            else:
                logger.debug(
                    f"H2D: batch #{batch_count}, no efeat available, skipping pos_efeat transfer"
                )

            # make mfg from raw data
            if meta.n2r_meta is not None:
                logger.debug(
                    f"H2D: batch #{batch_count}, building MFGs, num_blocks={len(meta.n2r_meta)}, history={self.history}"
                )
                meta.n2r_mfgs = collate_and_build_graph(meta.n2r_meta, self.history)
                logger.debug(
                    f"H2D: batch #{batch_count}, MFGs built, num_groups={len(meta.n2r_mfgs)}"
                )
                # Log details about each group
                for idx, group in enumerate(meta.n2r_mfgs):
                    logger.debug(
                        f"H2D: batch #{batch_count}, MFG group {idx} has {len(group)} blocks"
                    )
            else:
                logger.debug(
                    f"H2D: batch #{batch_count}, no n2r_meta, skipping MFG construction"
                )

            eids: list[list[torch.Tensor]] = []
            if meta.n2r_mfgs is not None:
                logger.debug(
                    f"H2D: batch #{batch_count}, transferring {len(meta.n2r_mfgs)} MFG groups to GPU"
                )
                for group_idx, blocks in enumerate(meta.n2r_mfgs):
                    eid: list[torch.Tensor] = [
                        torch.empty((block.num_edges(),), dtype=torch.long)
                        for block in blocks
                    ]
                    for block_idx, block in enumerate(blocks):
                        eid[block_idx] = block.edata["ID"]
                    eids.append(eid)
                    logger.debug(
                        f"H2D: batch #{batch_count}, MFG group {group_idx} transferred to GPU"
                    )
                logger.debug(f"H2D: batch #{batch_count}, all MFGs on GPU")
            else:
                logger.debug(f"H2D: batch #{batch_count}, no MFGs to transfer")

            # move gnn_mfgs to gpu
            if meta.n2r_mfgs is not None:
                logger.debug(
                    f"H2D: batch #{batch_count}, transferring {len(meta.n2r_mfgs)} MFG groups to GPU"
                )
                for group_idx, blocks in enumerate(meta.n2r_mfgs):
                    for block_idx, block in enumerate(blocks):
                        blocks[block_idx] = block.to(self.device, non_blocking=False)
                    logger.debug(
                        f"H2D: batch #{batch_count}, MFG group {group_idx} transferred to GPU"
                    )
                logger.debug(f"H2D: batch #{batch_count}, all MFGs on GPU")
            else:
                logger.debug(f"H2D: batch #{batch_count}, no MFGs to transfer")

            # can run no matter has nfeat or not
            if meta.n2r_mfgs is not None:
                logger.debug(f"H2D: batch #{batch_count}, filling MFGs with features")
                meta.n2r_mfgs = self.st_asm.fill_mfg_with_efeat(
                    gnn_mfgs=meta.n2r_mfgs,
                    pin_efeat_buf=self.buf.efeat_h2d,
                    eids=eids,
                )
                logger.debug(f"H2D: batch #{batch_count}, MFGs filled with features")

            # init x
            n_pos_edges = meta.pos_edge.shape[0]
            logger.debug(
                f"H2D: batch #{batch_count}, creating labels, n_pos={n_pos_edges}, n_neg_per_pos={self.n_neg}, total={n_pos_edges * (1 + self.n_neg)}"
            )
            meta.labels = torch.cat(
                [
                    torch.ones(n_pos_edges, device=self.device),
                    torch.zeros(n_pos_edges * self.n_neg, device=self.device),
                ],
            )
            logger.debug(
                f"H2D: batch #{batch_count}, labels created, shape={meta.labels.shape}"
            )

            # move plan to gpu
            # if meta.plan_set is not None:
            #     logger.debug(f"H2D: batch #{batch_count}, processing plan_set")
            #     if meta.plan_set.mem is not None and meta.plan_set.box is None:
            #         logger.debug(
            #             f"H2D: batch #{batch_count}, copying mem plan to box (box is None)"
            #         )
            #         meta.plan_set.box = meta.plan_set.mem
            #     meta.plan_set = meta.plan_set
            # else:
            #     logger.debug(f"H2D: batch #{batch_count}, no plan_set provided")

            # move nfeat and efeat to gpu
            if self.nfeat is not None or self.efeat is not None:
                logger.debug(f"H2D: batch #{batch_count}, processing nfeat/efeat")
                # prepare if has n_feat
                if meta.plan_set is not None and meta.plan_set.nfeat is not None:
                    logger.debug(f"H2D: batch #{batch_count}, preparing nfeat cache")
                    meta.nfeat = self.st_asm.prepare_nfeat_cache(
                        nfeat_plan=meta.plan_set.nfeat,
                        pin_nfeat_buf=self.buf.nfeat_h2d,
                    )
                    if meta.nfeat is not None:
                        logger.debug(
                            f"H2D: batch #{batch_count}, nfeat prepared, nid count={meta.nfeat.nid.numel()}, feature shape={meta.nfeat.nfeat.shape}"
                        )
                    else:
                        logger.debug(
                            f"H2D: batch #{batch_count}, nfeat preparation returned None"
                        )
                else:
                    logger.debug(
                        f"H2D: batch #{batch_count}, no nfeat plan, skipping nfeat cache"
                    )

                # can run no matter has nfeat or not
                if meta.n2r_mfgs is not None:
                    logger.debug(
                        f"H2D: batch #{batch_count}, filling MFGs with features"
                    )
                    meta.n2r_mfgs = self.st_asm.fill_mfg_with_nfeat(
                        gnn_mfgs=meta.n2r_mfgs,
                        nfeat=meta.nfeat,
                    )
                    logger.debug(
                        f"H2D: batch #{batch_count}, MFGs filled with features"
                    )

                # mfg version do not need nfeat output
                if meta.n2r_mfgs is not None:
                    logger.debug(
                        f"H2D: batch #{batch_count}, deleting nfeat (MFG version)"
                    )
                    del meta.nfeat  # mfg版本不需要nfeat输出
            else:
                logger.debug(
                    f"H2D: batch #{batch_count}, no nfeat or efeat, skipping feature processing"
                )

            # move mem and box to gpu
            if self.shared is not None:
                logger.debug(
                    f"H2D: batch #{batch_count}, acquiring h2d_sem for mem/box transfer (2 permits)"
                )
                assert self.buf.mem_h2d is not None
                assert self.buf.box_h2d is not None
                assert self.h2d_sem is not None
                self.h2d_sem.acquire()
                self.h2d_sem.acquire()
                logger.debug(
                    f"H2D: batch #{batch_count}, h2d_sem acquired, preparing mem and box"
                )
                assert meta.plan_set is not None
                assert meta.plan_set.mem is not None
                assert meta.plan_set.box is not None
                logger.debug(
                    f"H2D: batch #{batch_count}, mem plan fetch size={meta.plan_set.mem.fetch.data.numel()}, box plan fetch size={meta.plan_set.box.fetch.data.numel()}"
                )
                meta.mem_n_box = prep_part_mem_n_box(
                    self.shared,
                    meta.plan_set.mem,
                    meta.plan_set.box,
                    pin_mem_buf=self.buf.mem_h2d,
                    pin_box_buf=self.buf.box_h2d,
                    device=self.device,
                )
                logger.debug(
                    f"H2D: batch #{batch_count}, mem and box prepared, mem size={meta.mem_n_box.mem.nid.numel()}, box size={meta.mem_n_box.box.nid.numel()}"
                )
            else:
                logger.debug(
                    f"H2D: batch #{batch_count}, no shared memory, skipping mem/box transfer"
                )

            if meta.plan_set is not None:
                if meta.plan_set.mem is not None and meta.plan_set.box is None:
                    logger.debug(
                        f"H2D: batch #{batch_count}, copying mem plan to box (final check)"
                    )
                    meta.plan_set.box = meta.plan_set.mem
                # 使用 non_blocking 即可，无需 buffer
                logger.debug(
                    f"H2D: batch #{batch_count}, transferring plan_set to cuda"
                )
                meta.plan_set = meta.plan_set.to(self.device, non_blocking=False)
                logger.debug(f"H2D: batch #{batch_count}, plan_set on GPU")

            # even there is no next batch, we still need plan_set for writeback
            if self.merge_src:
                logger.debug(
                    f"H2D: batch #{batch_count}, deduplicating first layer src nodes"
                )
                assert meta.n2r_mfgs is not None
                orig_size = (
                    meta.n2r_mfgs[0][0].num_src_nodes() if meta.n2r_mfgs[0] else 0
                )
                meta.n2r_mfgs[0] = _deduplicate_first_layer_src(meta.n2r_mfgs[0])
                new_size = (
                    meta.n2r_mfgs[0][0].num_src_nodes() if meta.n2r_mfgs[0] else 0
                )
                logger.debug(
                    f"H2D: batch #{batch_count}, deduplication complete, src nodes: {orig_size} -> {new_size}"
                )

            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            logger.debug(
                f"H2D: batch #{batch_count} complete, recorded ready_event, putting to out_q"
            )
            self.out_q.put((meta, ready_event), block=True)
            logger.debug(f"H2D: batch #{batch_count} sent to out_q")

        logger.debug(f"H2D: exiting, processed {batch_count} batches")
