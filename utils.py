import os
from typing import List

import dgl
import pandas as pd
import torch
import yaml
from dgl.heterograph import DGLBlock

# In utils.py
from schemas.datatypes import (
    GNNCFG,
    Config,
    DGLBlockMeta,
    ExtFullGraph,
    MemoryCFG,
    SampleCFG,
    TrainCFG,
)


def _mail_delivery_tensor(
    unique_root_nodes: torch.Tensor, mail_delivery_block: DGLBlock
) -> torch.Tensor:
    src_local_id, dst_local_id = mail_delivery_block.edges(
        order="eid"
    )  # find out all edges
    global_src = mail_delivery_block.srcdata["ID"][src_local_id]
    global_dst = mail_delivery_block.dstdata["ID"][dst_local_id]

    uniq_global_src = torch.unique(global_src)
    uniq_global_dst = torch.unique(global_dst)
    is_in_root = torch.isin(global_src, unique_root_nodes)

    roots_sorted = torch.sort(unique_root_nodes).values
    src_sorted = torch.sort(global_src).values
    assert torch.equal(roots_sorted, src_sorted), "roots != unique(global_src)"

    final_src = global_src[is_in_root]
    final_dst = global_dst[is_in_root]
    tensor = torch.stack([final_src, final_dst], dim=0)
    return tensor


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


def load_graph(d) -> tuple[ExtFullGraph, pd.DataFrame, int, int]:
    g: ExtFullGraph = ExtFullGraph.load(f"DATA/{d}/ext_full.npz")
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


def extract_raw_data(ret: List) -> List[DGLBlockMeta]:
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
    mfg_meta_list: List[DGLBlockMeta], n_history: int
) -> Tuple[set, set]:
    """
    Analyzes the provided list of DGLBlockMeta to determine the maximum
    number of source and destination nodes across the last n_history blocks.

    Args:
        mfg_meta_list (List[DGLBlockMeta]): List of DGLBlockMeta objects.

    Returns:
        Tuple[set, set]: A tuple containing the set of all   source nodes
                         and the set of all root nodes.
    """
    all_nodes = set()
    root_nodes = set()
    for index, mfg_meta in enumerate(mfg_meta_list):
        nodes = mfg_meta.nodes.numpy()
        num_roots = mfg_meta.dim_out
        all_nodes.update(nodes.tolist())
        if index < n_history:
            root_nodes.update(nodes[:num_roots].tolist())

    return all_nodes, root_nodes


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


def to_apan_mailbox_block(
    mfg_meta_list: List[DGLBlockMeta], save_on_gpu=False
) -> DGLBlock:
    """
    专门用于 APAN 的异步邮件传播。
    只取 ret 中的第一个 TGB (1 layer, 1 history)。
    """
    # this is not standard DGL block
    # 按照 train_dist.py 的逻辑，reverse 时只使用第一个元素
    mfg_meta: DGLBlockMeta = mfg_meta_list[0]
    # 传播逻辑：row -> col (中心指向邻居)
    block = dgl.create_block(
        (mfg_meta.row, mfg_meta.col),
        num_src_nodes=mfg_meta.dim_out,
        num_dst_nodes=mfg_meta.dim_in,
    )
    all_nodes = mfg_meta.nodes
    num_roots = mfg_meta.dim_out

    # num_src_nodes = tgb.dim_out()少
    # num_dst_nodes = tgb.dim_in()多

    # 在传播过程中，目标节点（邻居）需要获取 ID 和时间戳
    block.srcdata["ID"] = all_nodes[:num_roots]
    block.dstdata["ID"] = all_nodes

    block.srcdata["ts"] = mfg_meta.ts[:num_roots]
    block.dstdata["ts"] = mfg_meta.ts  # 只有目标节点需要 ts

    # edge data
    block.edata["dt"] = mfg_meta.dts[num_roots:]
    block.edata["ID"] = mfg_meta.eid

    return block.to("cuda:0") if save_on_gpu else block


def node_to_dgl_blocks(root_nodes, ts) -> List[List[DGLBlock]]:
    blocks = list()
    block = dgl.create_block(
        ([], []), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0]
    )
    block.srcdata["ID"] = root_nodes
    block.srcdata["ts"] = ts
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
