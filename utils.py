import os
import time
from typing import List

import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from dgl.heterograph import DGLBlock

from profiler import Timer

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


def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata["ID"].long())
    if "ID" in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata["ID"].long())
    else:
        eids = None
    return nids, eids


def load_feat(
    d, rand_de=128, rand_dn=128
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    node_feats = None
    if os.path.exists("DATA/{}/node_features.pt".format(d)):
        node_feats = torch.load("DATA/{}/node_features.pt".format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists("DATA/{}/edge_features.pt".format(d)):
        edge_feats = torch.load("DATA/{}/edge_features.pt".format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == "LASTFM":
            edge_feats = torch.randn(1293103, rand_de)
        elif d == "MOOC":
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == "LASTFM":
            node_feats = torch.randn(1980, rand_dn)
        elif d == "MOOC":  # a bug fixed by zhaolei @sqp17
            node_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats


def load_graph(d) -> tuple[dict, pd.DataFrame]:
    g = np.load("DATA/{}/ext_full.npz".format(d))
    df = pd.read_csv("DATA/{}/edges.csv".format(d))
    return g, df


def parse_config(f, fashion="new"):
    conf = yaml.safe_load(open(f, "r"))
    if fashion == "new":
        sample_param = SampleCFG(**conf["sampling"][0])
        memory_param = MemoryCFG(**conf["memory"][0])
        gnn_param = GNNCFG(**conf["gnn"][0])
        train_param = TrainCFG(**conf["train"][0])
        print(
            f"Parsed config: {sample_param}, {memory_param}, {gnn_param}, {train_param}"
        )
    else:
        sample_param = conf["sampling"][0]
        memory_param = conf["memory"][0]
        gnn_param = conf["gnn"][0]
        train_param = conf["train"][0]

    # print(f"Parsed config: {sample_param}, {memory_param}, {gnn_param}, {train_param}")
    cfg = Config(s=sample_param, m=memory_param, g=gnn_param, t=train_param)
    cfg.t.batch_size = int(cfg.t.batch_size)
    return cfg


def get_shape_and_dtype(meta_payload):
    final_shape, final_dtype = meta_payload["shape"], meta_payload["dtype"]
    return final_shape, final_dtype


def get_node_edge_dimensions(node_np, edge_np):
    def _get_numpy_array_info(arr):
        if arr is None:
            return 0, 0, None
        shape = arr.shape
        dim0 = shape[0] if len(shape) > 0 else 0
        dim1 = shape[1] if len(shape) > 1 else 0
        return dim0, dim1, arr.dtype

    num_nodes_w_feat, node_feat_dim, node_dtype = _get_numpy_array_info(node_np)
    num_edges_w_feat, edge_feat_dim, edge_dtype = _get_numpy_array_info(edge_np)
    return (
        num_nodes_w_feat,
        node_feat_dim,
        node_dtype,
        num_edges_w_feat,
        edge_feat_dim,
        edge_dtype,
    )


def get_model_save_path(data_name, config_base_name):
    if not os.path.isdir("models"):
        os.mkdir("models")
    return "models/{}_{}_{}.pkl".format(
        data_name, config_base_name, time.time()
    )  # time.time() 在 rank 0 执行


def get_node_count_from_graph(g):
    return g["indptr"].shape[0] - 1


from sampler_core import ParallelSampler


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
    ret, history, reverse=False, save_on_pgu=True
) -> List[List[DGLBlock]]:
    # eg. if sample_param.history = 3, sample_param.layer = 2, ret = [Ly1His1, Ly1His2, Ly1His3, Ly2His1, Ly2His2, Ly2His3]
    # then grouped_mfgs = [[Ly1His1, Ly1His2], [Ly1His3, Ly2His1, [Ly2His2, Ly2His3]]
    mfgs = list()
    # 6 tgbs -> mfgs length 6 [1,2,3,4,5,6]
    # if history = 2, grouped_mfgs = [[1,2],[3,4],[5,6]]
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

        mfgs.append(block.to("cuda:0") if save_on_pgu == True else block)
    # [[Ly1His1, Ly1His2], [Ly1His3, Ly2His1, [Ly2His2, Ly2His3]]
    grouped_mfgs = list(map(list, zip(*[iter(mfgs)] * history)))
    grouped_mfgs.reverse()
    return grouped_mfgs


def node_to_dgl_blocks(root_nodes, ts, cuda) -> List[List[DGLBlock]]:
    mfgs = list()
    block = dgl.create_block(
        ([], []), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0]
    )
    block.srcdata["ID"] = torch.from_numpy(root_nodes)
    block.srcdata["ts"] = torch.from_numpy(ts)
    if cuda == True:
        mfgs.insert(0, [block.to("cuda:0")])
    else:
        mfgs.insert(0, [block])
    return mfgs


def mfgs_to_cuda(mfgs):
    new_mfgs = []
    for mfg in mfgs:
        new_mfg = [block.to("cuda:0") for block in mfg]
        new_mfgs.append(new_mfg)
    return new_mfgs


from typing import List

import torch


# @contextmanager
# def Timer():
#     class _Timer:
#         def __init__(self):
#             self.duration = 0
#         @property
#         def last_duration(self):
#             return self.duration
#     t = _Timer()
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     yield t
#     end.record()
#     torch.cuda.synchronize()
#     t.duration = start.elapsed_time(end) / 1000.0


def _deduplicate_first_layer_src(mfgs_0_on_gpu):
    """
    Deduplicates source nodes in the first MFG layer based on (timestamp, ID).
    This function modifies the blocks in mfgs_on_gpu_0 in place.
    """
    for i, block in enumerate(mfgs_0_on_gpu):
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

            mfgs_0_on_gpu[i] = new_block


def _fetch_features_deduplicated(
    mfgss_on_gpu, mfgss_on_cpu, all_features, entity_type, pinned, buffer, profiler=None
):
    """
    Deduplicates IDs globally and fetches features for nodes or edges.
    entity_type: 'node' or 'edge'
    """
    time_stats = {}
    count_stats = {}

    is_node = entity_type == "node"

    with Timer() as t:
        # 1. Collect all IDs from the relevant parts of mfgs
        if is_node:
            # For nodes, only IDs from the first layer's source nodes are needed
            ids_to_collect = torch.cat(
                [b.srcdata["ID"] for b in mfgss_on_cpu[0]], dim=0
            )
        else:  # is_edge
            edges = [
                b.edata["ID"] for mfg in mfgss_on_cpu for b in mfg if "ID" in b.edata
            ]
            ids_to_collect = (
                torch.cat(edges, dim=0) if edges else torch.tensor([], dtype=torch.long)
            )

        # 2. Find unique IDs and their inverse mapping
        unique_ids, inv_map = torch.unique(
            ids_to_collect, sorted=False, return_inverse=True
        )
        inv_map = inv_map.to("cuda", non_blocking=True)

    time_stats[f"0_compute_uniq_{entity_type}_id"] = t.last_duration

    # 3. Fetch features for unique IDs
    with Timer() as t:
        if pinned:
            torch.index_select(
                all_features, 0, unique_ids, out=buffer[: unique_ids.shape[0]]
            )
            features_on_gpu = buffer[: unique_ids.shape[0]].cuda(non_blocking=True)
        else:
            features_on_gpu = all_features[unique_ids].to("cuda", non_blocking=True)
    time_stats[f"1_transfer_{entity_type}_feat"] = t.last_duration

    # 4. Assign position ('pos') mapping back to the blocks
    with Timer() as t:
        offset = 0
        mfgs_to_update = (
            mfgss_on_gpu[0] if is_node else (b for mfg in mfgss_on_gpu for b in mfg)
        )
        for b in mfgs_to_update:
            if is_node:
                num_items = b.num_src_nodes()
                b.srcdata["pos"] = inv_map[offset : offset + num_items]
            elif "ID" in b.edata:  # is_edge
                num_items = b.num_edges()
                b.edata["pos"] = inv_map[offset : offset + num_items]
            else:
                num_items = 0
            offset += num_items
    time_stats[f"2_transfer_{entity_type}_mapping"] = t.last_duration

    # Update stats
    count_stats[f"unique_{entity_type}_id_count"] = unique_ids.shape[0]
    count_stats[f"total_{entity_type}_id_count"] = ids_to_collect.shape[0]
    if profiler:
        profiler.increase_time(time_stats)
        profiler.increase_count(count_stats)

    return features_on_gpu, unique_ids


def _fetch_features_no_dedup(
    mfgss_on_gpu, all_features, entity_type, pinned, buffers, profiler=None
):
    """
    Fetches features for each block individually without global dedup.
    """
    time_stats = {}
    is_node = entity_type == "node"

    mfgs_to_iterate = (
        mfgss_on_gpu[0] if is_node else (b for mfg in mfgss_on_gpu for b in mfg)
    )

    i = 0
    for block in mfgs_to_iterate:
        # For edges, skip blocks that don't need feature fetching
        if not is_node and not (block.num_src_nodes() > block.num_dst_nodes()):
            continue

        with Timer() as t:
            id_key, data_key, out_key = (
                ("ID", "srcdata", "h") if is_node else ("ID", "edata", "f")
            )

            ids = getattr(block, data_key)[id_key]

            if pinned:
                cpu_ids = ids.cpu().long()
                torch.index_select(
                    all_features, 0, cpu_ids, out=buffers[i][: cpu_ids.shape[0]]
                )
                getattr(block, data_key)[out_key] = buffers[i][: cpu_ids.shape[0]].cuda(
                    non_blocking=True
                )
            else:
                gpu_ids = ids.long()
                # Ensure features are on the same device as the model expects
                feats = all_features[gpu_ids.to(all_features.device)].float()
                getattr(block, data_key)[out_key] = feats.cuda()

            i += 1
        time_stats[f"1_transfer_{entity_type}_feat"] = t.last_duration
        if profiler:
            profiler.increase_time(time_stats)


from schemas.datatypes import PrevCache


def prepare_mfgs(
    dedup,
    mfgss_on_cpu,
    mfgss_on_gpu,
    node_feats,
    edge_feats,
    deduplicate_mfg_src=False,
    pinned=False,
    nfeat_buffs=None,
    efeat_buffs=None,
    profiler=None,
    prev_cache: PrevCache | None = None,
    # Incremental update arguments
):
    (prev_node_uniq, prev_edge_uniq, prev_node_feats_on_gpu, prev_edge_feats_on_gpu) = (
        prev_cache.uniqn,
        prev_cache.uniqe,
        prev_cache.gpunfeat,
        prev_cache.gpuefeat,
    )

    # Path 1: Incremental update logic (if previous batch data is provided)
    if prev_node_uniq is not None:
        return _prepare_mfgs_incremental(
            mfgss_on_gpu,
            mfgss_on_cpu,
            node_feats,
            edge_feats,
            prev_node_uniq,
            prev_edge_uniq,
            prev_node_feats_on_gpu,
            prev_edge_feats_on_gpu,
        )

    # Path 2: Initial processing for a new batch

    # Step 1: Optional - Deduplicate source nodes of the first layer
    if deduplicate_mfg_src:
        _deduplicate_first_layer_src(mfgss_on_gpu[0])

    node_feats_on_gpu, edge_feats_on_gpu = None, None
    node_uniq, edge_uniq = None, None

    # Step 2: Fetch features based on the dedup strategy
    if dedup:
        if node_feats is not None:
            node_feats_on_gpu, node_uniq = _fetch_features_deduplicated(
                mfgss_on_gpu,
                mfgss_on_cpu,
                node_feats,
                "node",
                pinned,
                nfeat_buffs,
                profiler,
            )
        if edge_feats is not None:
            edge_feats_on_gpu, edge_uniq = _fetch_features_deduplicated(
                mfgss_on_gpu,
                mfgss_on_cpu,
                edge_feats,
                "edge",
                pinned,
                efeat_buffs,
                profiler,
            )
    else:  # No global dedup
        if node_feats is not None:
            _fetch_features_no_dedup(
                mfgss_on_gpu, node_feats, "node", pinned, nfeat_buffs, profiler
            )
        if edge_feats is not None:
            _fetch_features_no_dedup(
                mfgss_on_gpu, edge_feats, "edge", pinned, efeat_buffs, profiler
            )

    return mfgss_on_gpu, node_feats_on_gpu, edge_feats_on_gpu, node_uniq, edge_uniq


def _get_current_ids(mfgss_on_cpu):
    """Step 1: Collect all node and edge IDs from the current batch's blocks."""
    current_node_ids = torch.cat([b.srcdata["ID"] for b in mfgss_on_cpu[0]])

    current_edge_ids_list = [
        b.edata["ID"] for mfg in mfgss_on_cpu for b in mfg if "ID" in b.edata
    ]
    current_edge_ids = (
        torch.cat(current_edge_ids_list)
        if current_edge_ids_list
        else torch.tensor([], dtype=torch.long)
    )

    return current_node_ids, current_edge_ids


def _find_new_unique_ids(current_ids, prev_uniq_ids):
    """Step 2: Given current IDs and previous unique IDs, find the new unique IDs."""
    if len(current_ids) == 0:
        return torch.tensor([], dtype=prev_uniq_ids.dtype, device=prev_uniq_ids.device)

    current_uniq_ids = torch.unique(current_ids)
    is_old_mask = torch.isin(current_uniq_ids, prev_uniq_ids, assume_unique=True)
    new_uniq_ids = current_uniq_ids[~is_old_mask]
    return new_uniq_ids


def _apply_unified_mapping(mfgss_on_gpu, final_node_uniq, final_edge_uniq):
    """Step 5: Apply the final ID-to-position mapping to all blocks in a vectorized way."""
    # --- Node Mapping ---
    max_node_id = final_node_uniq.max()
    node_map = torch.full((max_node_id + 1,), -1, dtype=torch.long)
    node_map[final_node_uniq] = torch.arange(len(final_node_uniq), dtype=torch.long)

    gpu_node_map = node_map.to(mfgss_on_gpu[0][0].device)  # Move map to GPU once
    for b in mfgss_on_gpu[0]:
        b.srcdata["pos"] = gpu_node_map[b.srcdata["ID"].long()]

    # --- Edge Mapping ---
    if len(final_edge_uniq) > 0:
        max_edge_id = final_edge_uniq.max()
        edge_map = torch.full((max_edge_id + 1,), -1, dtype=torch.long)
        edge_map[final_edge_uniq] = torch.arange(len(final_edge_uniq), dtype=torch.long)

        gpu_edge_map = edge_map.to(mfgss_on_gpu[0][0].device)  # Move map to GPU once
        for mfg in mfgss_on_gpu:
            for b in mfg:
                if "ID" in b.edata:
                    b.edata["pos"] = gpu_edge_map[b.edata["ID"].long()]


def _prepare_mfgs_incremental(
    mfgss_on_gpu,
    mfgss_on_cpu,
    node_feats,
    edge_feats,
    prev_node_uniq,
    prev_edge_uniq,
    prev_node_feats_on_gpu,
    prev_edge_feats_on_gpu,
):
    """
    Incrementally fetches features, only transferring data for new nodes and edges.
    (Refactored for clarity and readability)
    """
    # Step 1: Get all IDs from the current batch of blocks.
    current_node_ids, current_edge_ids = _get_current_ids(mfgss_on_cpu)

    # Step 2: Find which of those IDs are new compared to the previous batch.
    new_node_uniq = _find_new_unique_ids(current_node_ids, prev_node_uniq)
    new_edge_uniq = _find_new_unique_ids(current_edge_ids, prev_edge_uniq)

    # Step 3: Fetch features from main memory ONLY for the new IDs.
    new_nodes_feat_on_gpu = node_feats[new_node_uniq].cuda(non_blocking=True)
    new_edges_feat_on_gpu = edge_feats[new_edge_uniq].cuda(non_blocking=True)
    # self.mfgss_on_gpu, self.node_feats_on_gpu, self.edge_feats_on_gpu, self.node_uniq, self.edge_uniq
    # Step 4: Combine the already-on-GPU features with the newly fetched features.
    final_node_feats_on_gpu = torch.cat(
        [prev_node_feats_on_gpu, new_nodes_feat_on_gpu], dim=0
    )
    final_edge_feats_on_gpu = torch.cat(
        [prev_edge_feats_on_gpu, new_edges_feat_on_gpu], dim=0
    )

    final_node_uniq = torch.cat([prev_node_uniq, new_node_uniq])
    final_edge_uniq = torch.cat([prev_edge_uniq, new_edge_uniq])

    # Step 5: Update all blocks with the new position mapping.
    _apply_unified_mapping(mfgss_on_gpu, final_node_uniq, final_edge_uniq)

    return (
        mfgss_on_gpu,
        final_node_feats_on_gpu,
        final_edge_feats_on_gpu,
        final_node_uniq,
        final_edge_uniq,
    )
