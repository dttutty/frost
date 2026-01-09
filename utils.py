import os
from typing import List

import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from dgl.heterograph import DGLBlock
from sampler_core import ParallelSampler

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


def sampler_wrapper(no_neg, root_nodes, sampler: ParallelSampler | None, ts):
    if no_neg:
        pos_root_end = root_nodes.shape[0] * 2 // 3
        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
    else:
        sampler.sample(root_nodes, ts)
    ret = sampler.get_ret()
    sample_time = ret[0].sample_time()
    return ret, sample_time


def to_dgl_blocks(
    ret, history, reverse=False, save_on_pgu=False
) -> List[List[DGLBlock]]:
    # eg. if sample_param.history = 3, sample_param.layer = 2, ret = [Ly1His1, Ly1His2, Ly1His3, Ly2His1, Ly2His2, Ly2His3]
    # then grouped_blocks = [[Ly1His1, Ly1His2], [Ly1His3, Ly2His1, [Ly2His2, Ly2His3]]
    blocks = list()
    # 6 tgbs -> blocks length 6 [1,2,3,4,5,6]
    # if history = 2, grouped_blocks = [[1,2],[3,4],[5,6]]
    for tgb in ret:
        tgb_nodes = torch.from_numpy(tgb.nodes())
        tgb_dts = torch.from_numpy(tgb.dts())
        tgb_ts = torch.from_numpy(tgb.ts())
        tgb_eid = torch.from_numpy(tgb.eid())
        if not reverse:
            edges = (tgb.col(), tgb.row())
            num_src, num_dst = tgb.dim_in(), tgb.dim_out()
            dt_slice_start = num_dst
            id_ts_target = "srcdata"
        else:
            edges = (tgb.row(), tgb.col())
            num_src, num_dst = tgb.dim_out(), tgb.dim_in()
            dt_slice_start = num_src
            id_ts_target = "dstdata"
        block = dgl.create_block(edges, num_src_nodes=num_src, num_dst_nodes=num_dst)
        getattr(block, id_ts_target)["ID"] = tgb_nodes
        getattr(block, id_ts_target)["ts"] = tgb_ts

        # Assign edge data
        block.edata["dt"] = tgb_dts[dt_slice_start:]
        block.edata["ID"] = tgb_eid

        blocks.append(block.to("cuda:0") if save_on_pgu else block)
    # [[Ly1His1, Ly1His2], [Ly1His3, Ly2His1, [Ly2His2, Ly2His3]]
    grouped_blocks = list(map(list, zip(*[iter(blocks)] * history)))
    grouped_blocks.reverse()
    return grouped_blocks


def node_to_dgl_blocks(root_nodes, ts, cuda: bool) -> List[List[DGLBlock]]:
    blocks = list()
    block = dgl.create_block(
        ([], []), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0]
    )
    block.srcdata["ID"] = torch.from_numpy(root_nodes)
    block.srcdata["ts"] = torch.from_numpy(ts)
    if cuda:
        blocks.insert(0, [block.to("cuda:0")])
    else:
        blocks.insert(0, [block])
    return blocks


def blocks_to_cuda(blocks):
    new_blocks = []
    for block in blocks:
        new_block = [block.to("cuda:0") for block in block]
        new_blocks.append(new_block)
    return new_blocks


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
