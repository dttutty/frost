import logging
import threading
from queue import Queue as Q
from typing import Literal

import dgl
import numpy as np
import torch
from dgl.heterograph import DGLBlock

# In utils.py
from buffer import Buffers
from schemas.datatypes.misc import (
    DGLBlockMeta,
    EpochSplitIterInfo,
    Meta,
    PackedDGLBlockMeta,
    Shared,
)
from schemas.datatypes.plan import PackedDyPlan, PackedStPlan, PlanSet
from schemas.protocol.misc import CudaEventLike
from schemas.shm_layout import ShmKeys
from shm_manager import DoubleBufferedSHM
from thread_base_with_stream import StreamThreadBase
from utils import _deduplicate_first_layer_src
from worker.assembler.dy_asm import prep_part_mem_n_box
from worker.assembler.st_asm import StAssembler


logger = logging.getLogger(__name__)


def rebuild_meta_full_from_shm(mgr: DoubleBufferedSHM, manifest: torch.Tensor) -> Meta:
    """
    完全从 SHM 重建 Meta 对象。
    manifest: Shape [MANIFEST_SIZE], 包含 batch_idx 和各字段长度
    """
    batch_idx = int(manifest[ShmKeys.BATCH_IDX].item())

    # Helper: 根据 Key 读取 Tensor
    def read(key_enum: int) -> torch.Tensor | None:
        length = int(manifest[key_enum].item())
        if length > 0:
            # [Fix] 必须先转 int，否则 str(Enum) 会返回 "ShmKeys.ROOT_NID" 导致 Key 找不到
            # 错误写法: return mgr.read(batch_idx, str(key_enum), length)

            # 正确写法:
            return mgr.read(batch_idx, str(int(key_enum)), length).clone()

        return None

    def read_or_empty(key_enum: int) -> torch.Tensor:
        t = read(key_enum)
        if t is None:
            return torch.empty(0, dtype=torch.long)
        return t

    # 1. Basic
    root_nid = read(ShmKeys.ROOT_NID)
    root_ts = read(ShmKeys.ROOT_TS)
    pos_edge_id = read(ShmKeys.POS_EDGE_ID)
    assert pos_edge_id is not None
    n_src = int(pos_edge_id.numel())
    ms_cur_epoch = int(manifest[ShmKeys.MS_CUR_EPOCH].item())
    ms_n_epoch = int(manifest[ShmKeys.MS_N_EPOCH].item())

    m_state = None
    if ms_n_epoch > 0:
        split_map_rev = {0: "train", 1: "val", 2: "test"}
        split_val = int(manifest[ShmKeys.MS_SPLIT_TYPE].item())
        split_str: Literal["train", "val", "test"] = split_map_rev.get(
            split_val, "train"
        )

        m_state = EpochSplitIterInfo(
            cur_epoch=ms_cur_epoch,
            n_epoch=ms_n_epoch,
            cur_macro_of_split=int(manifest[ShmKeys.MS_CUR_MACRO_SPLIT].item()),
            n_macro_of_split=int(manifest[ShmKeys.MS_N_MACRO_SPLIT].item()),
            cur_macro_of_epoch=int(manifest[ShmKeys.MS_CUR_MACRO_EPOCH].item()),
            split=split_str,
        )

    # 2. Rebuild N2R
    n2r_meta = None

    # 检查是否有 N2R 数据
    if manifest[ShmKeys.N2R_NODES] > 0:
        # 读出所有数据块
        nodes_cat = read(ShmKeys.N2R_NODES)
        ts_cat = read(ShmKeys.N2R_TS)
        eid_cat = read(ShmKeys.N2R_EID)
        dts_cat = read(ShmKeys.N2R_DTS)
        col_cat = read(ShmKeys.N2R_COL)
        row_cat = read(ShmKeys.N2R_ROW)

        # 读出元数据 Tensor
        t_node_sec = read(ShmKeys.N2R_NODE_SECTIONS)
        t_edge_sec = read(ShmKeys.N2R_EDGE_SECTIONS)
        t_dims = read(ShmKeys.N2R_DIMS)

        # 静态方法解包 (需要你在 misc.py 实现 unpack_from_tensors)
        n2r_meta = PackedDGLBlockMeta.unpack_from_tensors(
            nodes_cat,
            ts_cat,
            eid_cat,
            dts_cat,
            col_cat,
            row_cat,
            t_node_sec,
            t_edge_sec,
            t_dims,
        )

    # --- 3. Rebuild R2N (Temporal Neighbors) ---
    r2n_meta = None
    r2n_mdt = None
    if manifest[ShmKeys.R2N_NODES] > 0:
        nodes_cat = read(ShmKeys.R2N_NODES)
        ts_cat = read(ShmKeys.R2N_TS)
        eid_cat = read(ShmKeys.R2N_EID)
        dts_cat = read(ShmKeys.R2N_DTS)
        col_cat = read(ShmKeys.R2N_COL)
        row_cat = read(ShmKeys.R2N_ROW)

        t_node_sec = read(ShmKeys.R2N_NODE_SECTIONS)
        t_edge_sec = read(ShmKeys.R2N_EDGE_SECTIONS)
        t_dims = read(ShmKeys.R2N_DIMS)

        r2n_mdt = read(ShmKeys.R2N_MDT)

        if nodes_cat is not None:
            unpacked = PackedDGLBlockMeta.unpack_from_tensors(
                nodes_cat,
                ts_cat,
                eid_cat,
                dts_cat,
                col_cat,
                row_cat,
                t_node_sec,
                t_edge_sec,
                t_dims,
            )
            if unpacked:
                r2n_meta = unpacked[0]  # r2n 通常只有一个 block

    # --- 4. Rebuild PlanSet ---
    mem_plan = None
    box_plan = None
    nfeat_plan = None

    # Helper: 读取 Plan 元数据
    # ... 在 read_plan_meta 函数中 ...

    def read_plan_meta(
        k_sizes: int, k_rec: int, k_send: int
    ) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:
        t_sizes = read(k_sizes)
        t_rec = read(k_rec)
        t_send = read(k_send)

        # 1. 恢复 sizes (List[int])
        # 如果 t_sizes 为空，说明没有 P2P 任务
        sizes = t_sizes.tolist() if t_sizes is not None else []

        # 2. 恢复计数器 (Numpy Array)
        # 直接转 numpy，因为 scatter 已经强制转为 int64 了，这里很安全
        rec = t_rec.numpy() if t_rec is not None else None
        send = t_send.numpy() if t_send is not None else None

        # [安全检查]
        # 如果有 sizes (意味着有任务) 但 send 为 None (意味着传输彻底丢了)，
        # 此时如果不处理，后面一定会崩。
        # 既然我们修复了 scatter，理论上不应发生。但为了防止 crash，我们可以补全。
        if len(sizes) > 0 and send is None:
            logger.warning(
                f"[H2D Fix] Batch {batch_idx}: P2P sizes={sizes} but send_cts is None. Filling with zeros."
            )
            send = np.zeros(len(sizes), dtype=np.int64)

        # 如果 send 存在但 sizes 没了
        if send is not None and len(sizes) == 0:
            sizes = [0] * len(send)

        return sizes, rec, send

    # A. Mem Plan
    if manifest[ShmKeys.MEM_REC_CTS] > 0 or manifest[ShmKeys.MEM_P2P_SIZES] > 0:
        p2p_sizes, rec_cts, send_cts = read_plan_meta(
            ShmKeys.MEM_P2P_SIZES, ShmKeys.MEM_REC_CTS, ShmKeys.MEM_SEND_CTS
        )
        packed_mem = PackedDyPlan(
            p2p_data=read_or_empty(ShmKeys.MEM_P2P),
            p2p_sizes=p2p_sizes,
            fetch_data=read_or_empty(ShmKeys.MEM_FETCH),
            writeback_data=read_or_empty(ShmKeys.MEM_WB),
            rec_cts=rec_cts,
            send_cts=send_cts,
        )
        mem_plan = packed_mem.unpack()

    # B. Box Plan
    if manifest[ShmKeys.BOX_REC_CTS] > 0 or manifest[ShmKeys.BOX_P2P_SIZES] > 0:
        p2p_sizes, rec_cts, send_cts = read_plan_meta(
            ShmKeys.BOX_P2P_SIZES, ShmKeys.BOX_REC_CTS, ShmKeys.BOX_SEND_CTS
        )
        packed_box = PackedDyPlan(
            p2p_data=read_or_empty(ShmKeys.BOX_P2P),
            p2p_sizes=p2p_sizes,
            fetch_data=read_or_empty(ShmKeys.BOX_FETCH),
            writeback_data=read_or_empty(ShmKeys.BOX_WB),
            rec_cts=rec_cts,
            send_cts=send_cts,
        )
        box_plan = packed_box.unpack()

    # C. Nfeat Plan
    if manifest[ShmKeys.NFEAT_FETCH] > 0 or manifest[ShmKeys.NFEAT_CACHE] > 0:
        packed_nfeat = PackedStPlan(
            to_cache_data=read_or_empty(ShmKeys.NFEAT_CACHE),
            fetch_data=read_or_empty(ShmKeys.NFEAT_FETCH),
        )
        nfeat_plan = packed_nfeat.unpack()

    plan_set = PlanSet(mem=mem_plan, box=box_plan, nfeat=nfeat_plan)

    # 4. 组装最终 Meta
    # --- 5. Assemble Meta ---
    # 注意：m_state 等轻量级元数据目前没有包含在 Manifest 中
    # 如果它们对 H2D 很重要，建议也加到 ShmKeys 中。
    # 这里假设 Worker 可以通过 batch_idx 推断或者暂时不需要它们。
    # 为了兼容现有 H2D 逻辑，我们可能需要创建一个 Dummy m_state 或从某处获取。
    # 暂时设为 None，下游逻辑需做判空处理。

    meta = Meta(
        n_src=n_src,
        root_nid=root_nid,
        root_ts=root_ts,
        pos_edge_id=pos_edge_id,
        n2r_meta=n2r_meta,
        r2n_meta=r2n_meta,
        r2n_mdt=r2n_mdt,
        plan_set=plan_set,
        batch_idx=batch_idx,
        state=None,  # State is not needed for H2D
        m_state=m_state,  # Warning: logic might rely on m_state.cur_macro_of_epoch for logging
        shm_lens={},  # Legacy field, empty now
    )
    return meta


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
        block.edata["dt"] = mfg_meta.dts
        block.edata["ID"] = mfg_meta.eid
        # 3. 统一上 GPU (由 Worker 决定何时上 GPU)
        blocks.append(block)

    # 4. 处理历史分组逻辑
    def chunked(iterable: list, n: int) -> list:
        return [iterable[i : i + n] for i in range(0, len(iterable), n)]

    grouped_blocks = chunked(blocks, history)

    grouped_blocks.reverse()

    return grouped_blocks


# h2d.py


class H2D(StreamThreadBase):
    def __init__(
        self,
        shm_manager: DoubleBufferedSHM,
        buf: Buffers,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
        shared: Shared | None,
        merge_src: bool,
        in_q: Q[tuple[Meta, CudaEventLike] | None | Literal["end_of_epoch"]],
        out_q: Q[tuple[Meta, CudaEventLike] | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
        h2d_sem: threading.Semaphore | None,
        device: torch.device,
        history: int = 1,
        n_neg: int = 1,
    ) -> None:
        super().__init__(name="h2d", stop_ev=stop_ev)
        self.buf = buf
        self.nfeat = nfeat
        self.efeat = efeat
        self.shared = shared
        self.device = device
        self.merge_src = merge_src
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.shm_mgr = shm_manager
        self.h2d_sem = h2d_sem
        self.history = history
        self.n_neg = n_neg  # TODO: make it configurable
        self.st_asm = StAssembler(
            shared_nfeat=nfeat, shared_efeat=efeat, device=self.device
        )
        self.seen_root_nids = torch.tensor([], dtype=torch.long, device="cpu")

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0

        while not self.stop_ev.is_set():
            meta_data = self.in_q.get(block=True)

            if meta_data is None:
                self.out_q.put(None, block=True)
                break
            elif meta_data == "end_of_epoch":
                self.out_q.put("end_of_epoch", block=True)
                continue

            # if isinstance(manifest, torch.Tensor):
            #     logger.info(
            #         f"H2D: Received Manifest[0] (BatchIdx) = {manifest[0].item()}"
            #     )
            meta, event = meta_data
            torch.cuda.current_stream().wait_event(event)

            eids: list[list[torch.Tensor]] = []
            if meta.n2r_mfgs is not None:
                for group_idx, blocks in enumerate(meta.n2r_mfgs):
                    eid = [block.edata["ID"] for block in blocks]
                    eids.append(eid)

            # move gnn_mfgs to gpu
            if meta.n2r_mfgs is not None:
                for group_idx, blocks in enumerate(meta.n2r_mfgs):
                    for block_idx, block in enumerate(blocks):
                        blocks[block_idx] = block.to(self.device, non_blocking=False)

            # can run no matter has nfeat or not
            if meta.n2r_mfgs is not None:
                meta.n2r_mfgs = self.st_asm.fill_mfg_with_efeat(
                    gnn_mfgs=meta.n2r_mfgs,
                    pin_efeat_buf=self.buf.efeat_h2d,
                    eids=eids,
                )

            # init x
            n_pos_edges = 0
            if meta.pos_edge_id is not None:
                n_pos_edges = meta.pos_edge_id.numel()
            if n_pos_edges > 0:
                meta.labels = torch.cat(
                    [
                        torch.ones(n_pos_edges, device=self.device),
                        torch.zeros(n_pos_edges * self.n_neg, device=self.device),
                    ],
                )
            else:
                # 即使为空，也要创建正确的空 Tensor 以防下游报错
                meta.labels = torch.empty(0, device=self.device)

            # move plan to gpu
            # if meta.plan_set is not None:
            #     if meta.plan_set.mem is not None and meta.plan_set.box is None:
            #         meta.plan_set.box = meta.plan_set.mem
            #     meta.plan_set = meta.plan_set

            # move nfeat and efeat to gpu
            if self.nfeat is not None or self.efeat is not None:
                # prepare if has n_feat
                if meta.plan_set is not None and meta.plan_set.nfeat is not None:
                    meta.nfeat = self.st_asm.prepare_nfeat_cache(
                        nfeat_plan=meta.plan_set.nfeat,
                        pin_nfeat_buf=self.buf.nfeat_h2d,
                    )
                    assert meta.nfeat is not None

                # can run no matter has nfeat or not
                if meta.n2r_mfgs is not None:
                    assert meta.nfeat is not None
                    meta.n2r_mfgs = self.st_asm.fill_mfg_with_nfeat(
                        gnn_mfgs=meta.n2r_mfgs,
                        nfeat=meta.nfeat,
                    )

                # mfg version do not need nfeat output
                if meta.n2r_mfgs is not None:
                    del meta.nfeat  # mfg版本不需要nfeat输出

            # move mem and box to gpu
            if self.shared is not None:
                # assert self.buf.mem_h2d is not None
                # assert self.buf.box_h2d is not None
                # assert self.h2d_sem is not None

                if meta.plan_set is not None and meta.plan_set.mem is not None:
                    # [Fix 2] 关键修复：必须在调用 prep 之前处理 box 为 None 的情况！
                    # 原来的逻辑写得太晚了，导致传进 prep 的 box 是 None
                    if meta.plan_set.box is None:
                        print(
                            f"H2D Batch {meta.batch_idx}: mem_plan exists but box_plan is None. Filling box_plan with mem_plan."
                        )
                        meta.plan_set.box = meta.plan_set.mem
                self.h2d_sem.acquire()
                self.h2d_sem.acquire()
                # assert meta.plan_set is not None
                # assert meta.plan_set.mem is not None
                # assert meta.plan_set.box is not None

                meta.mem_n_box = prep_part_mem_n_box(
                    self.shared,
                    meta.plan_set.mem,
                    meta.plan_set.box,
                    pin_mem_buf=self.buf.mem_h2d,
                    pin_box_buf=self.buf.box_h2d,
                    device=self.device,
                )
                x = meta.mem_n_box.mem.data  # [N, D]
                nid = meta.mem_n_box.mem.nid  # [N]
                # check if nid >3000
                if nid.numel() > 0:
                    max_nid = nid.max().item()
                    if max_nid > 3000:
                        raise RuntimeError(
                            f"H2D Batch {meta.batch_idx}: Unusually large nid detected in mem_n_box.mem.nid (max={max_nid}). "
                            "This may indicate a data issue or a need to adjust the threshold."
                        )

                row_has_nonzero = (x != 0).any(dim=1)  # [N] True = 这一行有非0
                zero_row_mask = ~row_has_nonzero  # True = 这一行全是0

                zero_nids = nid[zero_row_mask]

                # print(
                #     f"H2D Batch {meta.batch_idx}: {zero_nids.numel()} zero-memory rows"
                # )
                # print("some zero nids:", zero_nids[:20])

                # # print max meta.mem_n_box.mem.nid
                # print(
                #     f"H2D Batch {meta.batch_idx}: max nid in mem_n_box.mem.nid = {meta.mem_n_box.mem.nid.max().item()}"
                # )
                # # print root nids
                # self.seen_root_nids = torch.cat([self.seen_root_nids, meta.root_nid])
                # self.seen_root_nids = torch.unique(self.seen_root_nids)
                # print(
                #     f"H2D Batch {meta.batch_idx}: seen root nid = {self.seen_root_nids}"
                # )

            # even there is no next batch, we still need plan_set for writeback
            if self.merge_src:
                assert meta.n2r_mfgs is not None
                orig_size = (
                    meta.n2r_mfgs[0][0].num_src_nodes() if meta.n2r_mfgs[0] else 0
                )
                meta.n2r_mfgs[0] = _deduplicate_first_layer_src(meta.n2r_mfgs[0])
                new_size = (
                    meta.n2r_mfgs[0][0].num_src_nodes() if meta.n2r_mfgs[0] else 0
                )

            ready_event: CudaEventLike = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            mem = meta.mem_n_box.mem
            box = meta.mem_n_box.box
            mem_only = mem.nid[~torch.isin(mem.nid, box.nid)]
            box_only = box.nid[~torch.isin(box.nid, mem.nid)]
            self.out_q.put((meta, ready_event), block=True)
