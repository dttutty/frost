import logging
import threading
from queue import Queue as Q
from typing import Literal

import dgl
import numpy as np
import torch
from dgl.heterograph import DGLBlock

# In utils.py
from schemas.datatypes.misc import (
    DGLBlockMeta,
    EpochSplitIterInfo,
    Meta,
    PackedDGLBlockMeta,
)
from schemas.datatypes.plan import PackedDyPlan, PackedStPlan, PlanSet
from schemas.shm_layout import ShmKeys
from shm_manager import DoubleBufferedSHM
from thread_base_with_stream import StreamThreadBase


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


# rebuild_meta_full_from_shm.py


class Rebuilder(StreamThreadBase):
    def __init__(
        self,
        shm_manager: DoubleBufferedSHM,
        in_q: Q[Meta | None | Literal["end_of_epoch"]],
        out_q: Q[Meta | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="rebuilder", stop_ev=stop_ev)

        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.shm_mgr = shm_manager

        logger.debug("Rebuilder: shared memory enabled")

        logger.debug("Rebuilder: Initialized")

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0
        logger.debug("Rebuilder: starting logic loop")

        while not self.stop_ev.is_set():
            logger.debug(
                f"Rebuilder: waiting for meta from in_q (queue size: {self.in_q.qsize()})"
            )

            manifest = self.in_q.get(block=True)
            logger.debug(f"Rebuilder: received meta type: {type(manifest).__name__}")

            if manifest is None:
                logger.debug("Rebuilder: received None, putting to out_q and exiting")
                self.out_q.put(None, block=True)
                logger.debug(f"Rebuilder: Processed {batch_count} batches before exit")
                self.out_q.put(None, block=True)
                break
            elif manifest == "end_of_epoch":
                logger.debug("Rebuilder: received end_of_epoch, forwarding to out_q")
                self.out_q.put("end_of_epoch", block=True)
                continue

            # if isinstance(manifest, torch.Tensor):
            #     logger.info(
            #         f"Rebuilder: Received Manifest[0] (BatchIdx) = {manifest[0].item()}"
            #     )
            assert isinstance(manifest, torch.Tensor)
            meta = rebuild_meta_full_from_shm(self.shm_mgr, manifest)

            self.out_q.put(meta, block=True)
