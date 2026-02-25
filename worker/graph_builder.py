import logging
import threading
from queue import Queue as Q
from typing import Literal

import dgl
import numpy as np
import torch
import torch.distributed as dist  # [新增] 需要用到 dist.get_rank() 打印日志
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
            return mgr.read(batch_idx, str(int(key_enum)), length).clone()
        return None

    def read_or_empty(key_enum: int) -> torch.Tensor:
        t = read(key_enum)
        if t is None:
            return torch.empty(0, dtype=torch.long)
        return t

    # 1. Basic
    root_nid = read(ShmKeys.ROOT_NID)
    # check if any root_nid > 2000
    if root_nid is not None and root_nid.numel() > 0:
        max_root_nid = root_nid.max().item()
        if max_root_nid > 2000:
            raise RuntimeError(
                f"Batch {batch_idx}: Unusually large root_nid detected (max={max_root_nid}). "
                "This may indicate a data issue or a need to adjust the threshold."
            )
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

    if manifest[ShmKeys.N2R_NODES] > 0:
        nodes_cat = read(ShmKeys.N2R_NODES)
        ts_cat = read(ShmKeys.N2R_TS)
        eid_cat = read(ShmKeys.N2R_EID)
        dts_cat = read(ShmKeys.N2R_DTS)
        col_cat = read(ShmKeys.N2R_COL)
        row_cat = read(ShmKeys.N2R_ROW)

        t_node_sec = read(ShmKeys.N2R_NODE_SECTIONS)
        t_edge_sec = read(ShmKeys.N2R_EDGE_SECTIONS)
        t_dims = read(ShmKeys.N2R_DIMS)

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
                r2n_meta = unpacked[0]

    # --- 4. Rebuild PlanSet ---
    mem_plan = None
    box_plan = None
    nfeat_plan = None

    def read_plan_meta(
        k_sizes: int, k_rec: int, k_send: int
    ) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:
        t_sizes = read(k_sizes)
        t_rec = read(k_rec)
        t_send = read(k_send)

        sizes = t_sizes.tolist() if t_sizes is not None else []
        rec = t_rec.numpy() if t_rec is not None else None
        send = t_send.numpy() if t_send is not None else None

        if len(sizes) > 0 and send is None:
            logger.warning(
                f"[H2D Fix] Batch {batch_idx}: P2P sizes={sizes} but send_cts is None. Filling with zeros."
            )
            send = np.zeros(len(sizes), dtype=np.int64)

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
        state=None,
        m_state=m_state,
        shm_lens={},
    )
    return meta


@torch.no_grad()
def collate_and_build_graph(
    mfg_meta_list: list[DGLBlockMeta], history: int
) -> list[list[DGLBlock]]:
    blocks = []

    for mfg_meta in mfg_meta_list:
        block = dgl.create_block(
            (mfg_meta.col, mfg_meta.row),
            num_src_nodes=mfg_meta.dim_in,
            num_dst_nodes=mfg_meta.dim_out,
        )

        num_roots = mfg_meta.dim_out
        all_nodes = mfg_meta.nodes
        block.srcdata["ID"] = all_nodes
        block.dstdata["ID"] = all_nodes[:num_roots]
        block.srcdata["ts"] = mfg_meta.ts
        block.dstdata["ts"] = mfg_meta.ts[:num_roots]
        block.edata["dt"] = mfg_meta.dts
        block.edata["ID"] = mfg_meta.eid
        blocks.append(block)

    def chunked(iterable: list, n: int) -> list:
        return [iterable[i : i + n] for i in range(0, len(iterable), n)]

    grouped_blocks = chunked(blocks, history)
    grouped_blocks.reverse()

    return grouped_blocks


class GraphBuilder(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers | None,
        efeat: torch.Tensor | None,
        shared: Shared | None,
        in_q: Q[Meta | None | Literal["end_of_epoch"]],
        out_q: Q[tuple[Meta, CudaEventLike] | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
        device: torch.device,
        history: int = 1,
    ) -> None:
        super().__init__(name="graph_builder", stop_ev=stop_ev)

        self.buf = buf
        self.efeat = efeat
        self.device = device
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.history = history

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            meta = self.in_q.get(block=True)

            if meta is None:
                self.out_q.put(None, block=True)
                break
            elif meta == "end_of_epoch":
                self.out_q.put("end_of_epoch", block=True)
                continue

            if meta.root_nid is None or meta.root_nid.numel() == 0:
                logger.error(
                    "root_nid is None/Empty. "
                    "Possible causes: 1. Host sent empty batch. 2. Received Padding Tensor (Rank mismatch)."
                )
                raise RuntimeError("H2D received empty Meta!")

            if meta.plan_set is not None:
                if meta.plan_set.mem is not None and meta.plan_set.box is None:
                    meta.plan_set.box = meta.plan_set.mem

                meta.plan_set = meta.plan_set.to(self.device, non_blocking=False)

            # move pos_efeat to gpu
            if self.efeat is not None:
                # 只有当 pos_edge_id 真的有数据时才传输
                if meta.pos_edge_id is not None and meta.pos_edge_id.numel() > 0:
                    # --- [New Debug Code Start] ---
                    # 在做 index_select 之前，强制检查索引范围
                    # 这样如果越界，会报 Python 错误，而不是 CUDA Assert Error
                    max_eid = meta.pos_edge_id.max().item()
                    min_eid = meta.pos_edge_id.min().item()
                    num_edges = self.efeat.size(0)

                    if max_eid >= num_edges or min_eid < 0:
                        err_msg = (
                            f"\n[CRITICAL ERROR Rank {dist.get_rank()}] Edge Feature Index Out of Bounds!\n"
                            f"  > Efeat Shape: {self.efeat.shape}\n"
                            f"  > Max Requested Index: {max_eid}\n"
                            f"  > Min Requested Index: {min_eid}\n"
                            f"  > Batch Idx: {meta.batch_idx}\n"
                        )
                        logger.error(err_msg)
                        import sys

                        sys.stdout.flush()
                        raise RuntimeError(
                            f"Edge index out of bounds: {max_eid} >= {num_edges}"
                        )
                    # --- [New Debug Code End] ---

                    if (
                        self.buf is not None
                        and self.buf.pos_efeat_h2d is not None
                        and meta.pos_edge_id.numel() <= self.buf.pos_efeat_h2d.shape[0]
                    ):
                        n = meta.pos_edge_id.numel()
                        buf_view = self.buf.pos_efeat_h2d[:n]

                        torch.index_select(
                            self.efeat, 0, meta.pos_edge_id, out=buf_view
                        )
                        meta.pos_efeat = buf_view.to(self.device, non_blocking=False)
                    else:
                        meta.pos_efeat = self.efeat[meta.pos_edge_id].to(
                            self.device, non_blocking=False
                        )
                else:
                    logger.warning(
                        "GraphBuilder: [WARNING] batch has NO pos_edge_id! "
                        f"Manifest[{ShmKeys.POS_EDGE_ID}] was 0. "
                        "Please check Scatter logs."
                    )

            # make mfg from raw data
            if meta.n2r_meta is not None:
                meta.n2r_mfgs = collate_and_build_graph(meta.n2r_meta, self.history)

            ready_event: CudaEventLike = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            self.out_q.put((meta, ready_event), block=True)
