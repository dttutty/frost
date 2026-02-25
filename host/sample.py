from __future__ import annotations

import logging

import torch
from graph_sampler import SimpleSampler, WindowSampler

from schemas.datatypes.misc import DGLBlockMeta
from schemas.datatypes.plan import BatchNodeStateIO
from tensor_set import TensorSet


logger = logging.getLogger(__name__)


def extract_raw_data(ret: list) -> list[DGLBlockMeta]:
    mfg_meta_list = []
    for i, tgb in enumerate(ret):
        # 1. 强制转为 int64 (long)，避免 int32/int64 混用带来的隐患
        # 注意：copy=False 尽可能不复制内存，但 .long() 可能会触发复制
        nodes = torch.from_numpy(tgb.nodes()).long()
        ts = torch.from_numpy(tgb.ts()).double()  # 时间戳建议转 double 或 float
        eid = torch.from_numpy(tgb.eid()).long()
        raw_dts = torch.from_numpy(tgb.dts()).double()
        col = torch.from_numpy(tgb.col()).long()
        row = torch.from_numpy(tgb.row()).long()

        dim_in = tgb.dim_in()
        dim_out = tgb.dim_out()

        dts = raw_dts[dim_out:]

        # 调试断言：确保切片后的长度等于边数
        if __debug__:
            assert dts.shape[0] == eid.shape[0], (
                f"Mismatch! dts sliced len {dts.shape[0]} != eid len {eid.shape[0]}. "
                f"Total dts: {raw_dts.shape[0]}, dim_out: {dim_out}"
            )

        # === [关键调试检查] ===
        # 检查1: 节点数量必须至少包含所有的种子节点 (dim_out)
        # 如果这个断言触发，说明 C++ Sampler 丢了孤立点！
        assert nodes.numel() >= dim_out, (
            f"Batch {i} Error: tgb.nodes() count ({nodes.numel()}) < dim_out ({dim_out}). "
            "Sampler seems to have dropped isolated seed nodes!"
        )

        # 检查2: 确保所有 row/col 的索引都在 nodes 范围内
        # 这是一个耗时检查，确定没问题后可以注释掉
        if __debug__:
            max_idx = nodes.numel()
            if row.numel() > 0:
                assert row.max() < max_idx, (
                    f"Row index {row.max()} OOB for nodes size {max_idx}"
                )
                assert col.max() < max_idx, (
                    f"Col index {col.max()} OOB for nodes size {max_idx}"
                )

        mfg_meta: DGLBlockMeta = DGLBlockMeta(
            nodes=nodes,
            ts=ts,
            eid=eid,
            dts=dts,
            col=col,
            row=row,
            dim_in=dim_in,
            dim_out=dim_out,
        )
        mfg_meta_list.append(mfg_meta)
    return mfg_meta_list


def analyze_blocks(
    mfg_meta_list: list[DGLBlockMeta], n_history: int
) -> tuple[TensorSet, TensorSet]:
    all_nodes: list[torch.Tensor] = []
    root_nodes: list[torch.Tensor] = []
    for index, mfg_meta in enumerate(mfg_meta_list):
        nodes = mfg_meta.nodes
        num_roots = mfg_meta.dim_out
        all_nodes.append(nodes)
        if index < n_history:
            root_nodes.append(nodes[:num_roots])
    return TensorSet(torch.cat(all_nodes)), TensorSet(torch.cat(root_nodes))


def analyze_need(mfg_meta_list: list[DGLBlockMeta], n_history: int) -> BatchNodeStateIO:
    """
    Analyzes the provided list of DGLBlockMeta to determine the maximum
    number of source and destination nodes across the last n_history blocks.

    Args:
        mfg_meta_list (List[DGLBlockMeta]): List of DGLBlockMeta objects.

    Returns:
        BatchNodeStateIO: An object containing the sets of all source nodes
                        and the set of all root nodes.
    """
    all_nodes, root_nodes = analyze_blocks(mfg_meta_list, n_history)
    batch_node_state_io = BatchNodeStateIO(
        mem_needed=all_nodes,
        mem_updated=root_nodes,
        mailbox_needed=all_nodes,
        mailbox_updated=root_nodes,
        nfeat_needed=all_nodes,
    )
    return batch_node_state_io


def sample(
    ngh_sampler: SimpleSampler | WindowSampler,
    src: torch.Tensor,
    dst: torch.Tensor,
    neg_dst: torch.Tensor,
    ts: torch.Tensor,
) -> list[DGLBlockMeta]:
    roots = torch.cat([src, dst, neg_dst])
    ts_np = ts.repeat(3).numpy()
    roots_np = roots.numpy()
    ngh_sampler.sample(roots_np, ts_np)
    ret = ngh_sampler.get_ret()
    gnn_mfgs_raws = extract_raw_data(ret)
    return gnn_mfgs_raws
