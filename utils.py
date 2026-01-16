import os
from typing import List

import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from dgl.heterograph import DGLBlock

# In utils.py
from schemas.datatypes import GNNCFG, Config, MemoryCFG, SampleCFG, TrainCFG


def dgl_blocks_to_tensor_dicts(blocks):
    """Converts a list of DGL blocks to a list of serializable tensor dictionaries."""
    tensor_dicts = []
    for block in blocks:
        # Get the graph structure in CSC format, which is efficient
        csc_mat = block.adj_tensors("csc")
        tensor_dicts.append(
            {
                "num_src_nodes": block.number_of_src_nodes(),
                "num_dst_nodes": block.number_of_dst_nodes(),
                "csc_indptr": csc_mat[0],
                "csc_indices": csc_mat[1],
                "edge_ids": csc_mat[2],
            }
        )
    return tensor_dicts


def tensor_dicts_to_dgl_blocks(tensor_dicts, device="cpu"):
    """Reconstructs a list of DGL blocks from a list of tensor dictionaries."""
    blocks = []
    for d in tensor_dicts:
        # Create a graph from the CSC representation
        g = dgl.create_block(
            (d["csc_indptr"], d["csc_indices"]),
            num_src_nodes=d["num_src_nodes"],
            num_dst_nodes=d["num_dst_nodes"],
            device=device,
        )
        # The edge IDs are stored in the 'eid' feature of the new block
        g.edata[dgl.EID] = d["edge_ids"]
        blocks.append(g)
    return blocks


def get_ids(blocks, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in blocks[0]:
            nids.append(b.srcdata["ID"].long())
    if "ID" in blocks[0][0].edata:
        if edge_feats is not None:
            for block in blocks:
                for b in block:
                    eids.append(b.edata["ID"].long())
    else:
        eids = None
    return nids, eids


from typing import Optional, Tuple


def load_feat(
    dataset: str, feat_type: str, n_count: int, rnd_dim: int = 128
) -> Tuple[Optional[torch.Tensor], int, Optional[torch.dtype]]:
    """
    Load features for either nodes or edges based on feat_type.
    Returns (feat, d_feat, dtype).
    """
    assert feat_type in ["node", "edge"], "feat_type must be 'node' or 'edge'"

    # 1. Initialize all variables with "empty" default values
    # This handles the case where file doesn't exist AND rnd_dim == 0
    feat = None
    d_feat = 0
    dtype = None

    # Construct path dynamically based on type
    file_name = f"{feat_type}_features.pt"
    path = f"DATA/{dataset}/{file_name}"

    if os.path.exists(path):
        # Case A: Load from file
        feat = torch.load(path)

        # Ensure float32 dtype if it was bool
        if feat.dtype == torch.bool:
            feat = feat.type(torch.float32)

        assert len(feat.shape) == 2
        assert feat.shape[0] == n_count

        d_feat = feat.shape[1]
        dtype = feat.dtype

    elif rnd_dim > 0:
        # Case B: Generate random features (only if rnd_dim > 0)

        # Hardcoded checks for specific datasets
        if dataset == "LASTFM":
            expected = 1980 if feat_type == "node" else 1293103
            assert n_count == expected
        if dataset == "MOOC":
            expected = 7144 if feat_type == "node" else 411749
            assert n_count == expected

        feat = torch.randn(n_count, rnd_dim)
        d_feat = rnd_dim
        dtype = feat.dtype

    return feat, d_feat, dtype


def load_graph(d) -> tuple[dict, pd.DataFrame, int, int]:
    g = np.load("DATA/{}/ext_full.npz".format(d))
    df = pd.read_csv("DATA/{}/edges.csv".format(d))
    src = df["src"].to_numpy()
    dst = df["dst"].to_numpy()
    n_nodes_infer_from_df = max(src.max(), dst.max()) + 1
    n_nodes_infer_from_g = g["indptr"].shape[0] - 1
    assert n_nodes_infer_from_g == n_nodes_infer_from_df
    n_edges = df.shape[0]
    return g, df, n_nodes_infer_from_g, n_edges


def parse_config(f) -> Config:
    conf = yaml.safe_load(open(f, "r"))

    cfg_s = conf.get("sampling")
    if cfg_s is None:
        sample_param = None
    else:
        sample_param = SampleCFG(**cfg_s[0])

    cfg_m = conf.get("memory")
    if cfg_m is None:
        memory_param = None
    else:
        memory_param = MemoryCFG(**cfg_m[0])

    gnn_conf = GNNCFG(**conf["gnn"][0])
    train_conf = TrainCFG(**conf["train"][0])

    print(f"Parsed config: {sample_param}, {memory_param}, {gnn_conf}, {train_conf}")

    cfg = Config(s=sample_param, m=memory_param, g=gnn_conf, t=train_conf)
    cfg.t.batch_size = int(cfg.t.batch_size)
    return cfg


def to_dgl_blocks(ret, history, save_on_gpu=False) -> List[List[DGLBlock]]:
    """
    用于正常的 GNN 前向计算，支持多层聚合和多快照历史。
    """
    blocks = []
    for tgb in ret:
        # 提取数据并转为 Tensor
        all_nodes = torch.from_numpy(tgb.nodes())
        tgb_ts = torch.from_numpy(tgb.ts())
        tgb_eid = torch.from_numpy(tgb.eid())
        tgb_dts = torch.from_numpy(tgb.dts())

        # 正向逻辑：col -> row (邻居指向中心)
        edges = (tgb.col(), tgb.row())

        # 创建 DGL Block
        block = dgl.create_block(
            edges, num_src_nodes=tgb.dim_in(), num_dst_nodes=tgb.dim_out()
        )

        # num_dst_nodes =tgb.dim_out()少
        # num_src_nodes = tgb.dim_in()多

        num_roots = tgb.dim_out()

        block.srcdata["ID"] = all_nodes
        block.dstdata["ID"] = all_nodes[:num_roots]

        block.srcdata["ts"] = tgb_ts
        block.dstdata["ts"] = tgb_ts[:num_roots]  # 只有目标节点需要 ts

        block.edata["dt"] = tgb_dts[num_roots:]  # 只有边需要 dt
        block.edata["ID"] = tgb_eid

        blocks.append(block.to("cuda:0") if save_on_gpu else block)

    # 按照历史步长分组
    grouped_blocks = list(map(list, zip(*[iter(blocks)] * history)))
    # 反转层级顺序以匹配从外向里的聚合顺序
    grouped_blocks.reverse()
    return grouped_blocks


def to_apan_mailbox_block(ret, save_on_gpu=False) -> DGLBlock:
    """
    专门用于 APAN 的异步邮件传播。
    只取 ret 中的第一个 TGB (1 layer, 1 history)。
    """
    # this is not standard DGL block
    # 按照 train_dist.py 的逻辑，reverse 时只使用第一个元素
    tgb = ret[0]

    all_nodes = torch.from_numpy(tgb.nodes())
    tgb_ts = torch.from_numpy(tgb.ts())
    tgb_eid = torch.from_numpy(tgb.eid())
    tgb_dts = torch.from_numpy(tgb.dts())

    # 传播逻辑：row -> col (中心指向邻居)
    edges = (tgb.row(), tgb.col())

    block = dgl.create_block(
        edges, num_src_nodes=tgb.dim_out(), num_dst_nodes=tgb.dim_in()
    )

    # num_src_nodes = tgb.dim_out()少
    # num_dst_nodes = tgb.dim_in()多

    num_roots = tgb.dim_out()
    # 在传播过程中，目标节点（邻居）需要获取 ID 和时间戳
    block.srcdata["ID"] = all_nodes[:num_roots]
    block.dstdata["ID"] = all_nodes

    block.srcdata["ts"] = tgb_ts[:num_roots]
    block.dstdata["ts"] = tgb_ts  # 只有目标节点需要 ts

    # edge data
    block.edata["dt"] = tgb_dts[num_roots:]
    block.edata["ID"] = tgb_eid

    return block.to("cuda:0") if save_on_gpu else block


def node_to_dgl_blocks(root_nodes, ts, cuda: bool) -> List[List[DGLBlock]]:
    blocks = list()
    block = dgl.create_block(
        ([], []), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0]
    )
    block.srcdata["ID"] = root_nodes
    block.srcdata["ts"] = ts
    if cuda:
        blocks.insert(0, [block.to("cuda:0")])
    else:
        blocks.insert(0, [block])
    return blocks


def blocks_to_cuda(blockss: List[List[DGLBlock]]) -> List[List[DGLBlock]]:
    for blocks in blockss:
        for block in blocks:
            block = block.to("cuda:0")
    return blockss


def _deduplicate_first_layer_src(blockss_0):
    """
    Deduplicates source nodes in the first block layer based on (timestamp, ID).
    This function modifies the blocks in blocks_on_gpu_0 in place.
    """
    for i, block in enumerate(blockss_0):
        if block.num_src_nodes() > block.num_dst_nodes():
            num_dst = block.num_dst_nodes()

            # Combine timestamp and ID for unique identification
            ts = block.srcdata["ts"][num_dst:]
            nid = block.srcdata["ID"][num_dst:].float()
            nts = torch.stack([ts, nid], dim=1)

            unts, inv_map = torch.unique(nts, dim=0, return_inverse=True)

            # New source nodes include original destinations + unique non-destinations
            new_src_count = unts.shape[0] + num_dst

            # Create new block with updated graph structure
            # Edges now point from original destinations/new unique sources to destinations
            src_nodes = inv_map + num_dst
            dst_nodes = block.edges()[1]

            # A node can emit multiple edges or it may not emit any edges at all
            new_block = dgl.create_block(
                (src_nodes, dst_nodes),
                num_src_nodes=new_src_count,
                num_dst_nodes=num_dst,
                device=block.device,
            )

            # Copy over data to the new block
            new_block.srcdata["ts"] = torch.cat(
                [block.srcdata["ts"][:num_dst], unts[:, 0]], dim=0
            )
            new_block.srcdata["ID"] = torch.cat(
                [block.srcdata["ID"][:num_dst], unts[:, 1]], dim=0
            )
            new_block.edata["dt"] = block.edata["dt"]
            new_block.edata["ID"] = block.edata["ID"]

            blockss_0[i] = new_block
    return blockss_0
