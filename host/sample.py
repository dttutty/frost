from __future__ import annotations

import logging

import torch

from sampler_core import ParallelSampler
from schemas.datatypes.misc import DGLBlockMeta
from schemas.datatypes.plan import BatchNodeStateIO
from tensor_set import TensorSet


logger = logging.getLogger(__name__)


def extract_raw_data(ret: list) -> list[DGLBlockMeta]:
    # 用简单的字典或者 NamedTuple 存储纯数据
    mfg_meta_list = []
    for tgb in ret:
        # 1. 提取 Numpy 数据并转为 CPU Tensor
        # 此时不要去 GPU，也不要创建 DGLBlock
        mfg_meta: DGLBlockMeta = DGLBlockMeta(
            nodes=torch.from_numpy(tgb.nodes()),
            ts=torch.from_numpy(tgb.ts()),
            eid=torch.from_numpy(tgb.eid()),
            dts=torch.from_numpy(tgb.dts()),
            col=torch.from_numpy(tgb.col()),
            row=torch.from_numpy(tgb.row()),
            dim_in=tgb.dim_in(),
            dim_out=tgb.dim_out(),
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
    ngh_sampler: ParallelSampler,
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
