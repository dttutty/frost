import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from numba import njit, prange
from sampler_core import ParallelSampler
from tqdm import tqdm


class NegLinkSampler:
    def __init__(self, g):
        self.num_nodes = g["indptr"].shape[0] - 1

    def sample(self, src_ids, src_tss, num_neg=1):
        return torch.randint(0, self.num_nodes, (len(src_ids) * num_neg,))


class NegLinkInductiveSampler:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def sample(self, n):
        return np.random.choice(self.nodes, size=n)


# class RealNegLinkSampler:
#     def __init__(self, g):
#         self.g = g
#         self.max_node_id = g['indptr'].shape[0] - 2  # max node id is len(indptr) - 2

#     def get_all_neighbors(self, node_id, node_ts):
#         # Extract the CSR arrays from the NpzFile object
#         indptr = self.g['indptr']
#         indices = self.g['indices']

#         # Check for valid node_id
#         if node_id < 0 or node_id >= len(indptr) - 1:
#             print(f"Error: node_id {node_id} is out of bounds.")
#             return np.array([], dtype=indices.dtype)

#         # Get the start and end pointers from the indptr array
#         start = indptr[node_id]
#         end = indptr[node_id + 1]

#         # Slice the indices array to get the neighbors
#         return indices[start:end]

#     def sample_real_neg_dst(self, src_id, real_pos_dst, num_neg_sample):
#         real_pos_set = set(real_pos_dst.tolist())
#         neg_samples = []
#         batch_size = max(4, num_neg_sample * 2)
#         while len(neg_samples) < num_neg_sample:
#             candidates = np.random.randint(
#                 0, self.max_node_id + 1, size=batch_size
#             )
#             for c in candidates:
#                 if c != src_id and c not in real_pos_set:
#                     neg_samples.append(c)
#                     if len(neg_samples) == num_neg_sample:
#                         break
#         return np.array(neg_samples, dtype=np.int32)

#     def sample(self, src_ids, num_neg_sample=1):
#         neg_matrix = np.empty((len(src_ids), num_neg_sample), dtype=np.int32)
#         for i, src in enumerate(src_ids):
#             src = int(src)
#             pos = self.get_all_neighbors(src)
#             neg = self.sample_real_neg_dst(src, pos, num_neg_sample)
#             neg_matrix[i] = neg
#         src_cross = np.tile(src_ids, num_neg_sample)
#         neg_cross = neg_matrix.T.reshape(-1)
#         return neg_cross


# Numba-accelerated helper to fetch CSR neighbors\
@njit
def _get_neighbors(indptr, indices, node_id):
    if node_id < 0 or node_id >= indptr.shape[0] - 1:
        return np.empty(0, dtype=indices.dtype)
    start = indptr[node_id]
    end = indptr[node_id + 1]
    out = np.empty(end - start, dtype=indices.dtype)
    for i in range(start, end):
        out[i - start] = indices[i]
    return out


# Numba-accelerated single-node negative sampling\
@njit
def _sample_neg_for_src(indptr, indices, max_node_id, src_id, num_neg):
    neigh = _get_neighbors(indptr, indices, src_id)
    negs = np.empty(num_neg, dtype=np.int32)
    count = 0
    while count < num_neg:
        c = np.random.randint(0, max_node_id + 1)
        if c == src_id:
            continue
        # linear scan in compiled code
        ok = True
        for j in range(neigh.shape[0]):
            if neigh[j] == c:
                ok = False
                break
        if ok:
            negs[count] = c
            count += 1
    return negs


# Numba-accelerated batch sampling and cross-pairing\
@njit(parallel=True)
def sample(src_ids, indptr, indices, num_neg, max_node_id):
    n = src_ids.shape[0]
    # matrix to hold per-source neg samples
    neg_mat = np.empty((n, num_neg), dtype=np.int32)
    for i in prange(n):
        neg_mat[i] = _sample_neg_for_src(
            indptr, indices, max_node_id, src_ids[i], num_neg
        )
    # build cross-paired output
    total = n * num_neg
    neg_cross = np.empty(total, dtype=np.int32)
    for idx in prange(total):
        # row = idx // n, col = idx % n if transpose order
        r = idx // n
        c = idx % n
        neg_cross[idx] = neg_mat[c, r]
    return neg_cross


# Wrapper class
def create_sampler(g):
    indptr = g["indptr"]
    indices = g["indices"]
    max_node_id = indptr.shape[0] - 2

    class NumbaNegSampler:
        def __init__(self):
            self.indptr = indptr
            self.indices = indices
            self.max_node_id = max_node_id

        def sample(self, src_ids, src_tss, num_neg=1):
            src_arr = np.asarray(src_ids, dtype=np.int32)
            return sample(src_arr, self.indptr, self.indices, num_neg, self.max_node_id)

    return NumbaNegSampler()


class AvoidInfoLeakNegLinkSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def sample(self, src_ids, src_tss, num_neg=1):
        return sampler.sample_negatives(
            root_nodes=src_ids.astype(np.int32),
            root_ts=src_tss.astype(np.float32),
            num_neg_samples=num_neg,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="dataset name")
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument(
        "--batch_size", type=int, default=600, help="path to config file"
    )
    parser.add_argument("--num_thread", type=int, default=64, help="number of thread")
    args = parser.parse_args()

    df = pd.read_csv("DATA/{}/edges.csv".format(args.data))
    g = np.load("DATA/{}/ext_full.npz".format(args.data))
    sample_config = yaml.safe_load(open(args.config, "r"))["sampling"][0]

    sampler = ParallelSampler(
        g["indptr"],
        g["indices"],
        g["eid"],
        g["ts"].astype(np.float32),
        args.num_thread,
        1,
        sample_config["layer"],
        sample_config["neighbor"],
        sample_config["strategy"] == "recent",
        sample_config["prop_time"],
        sample_config["history"],
        float(sample_config["duration"]),
    )

    num_nodes = max(int(df["src"].max()), int(df["dst"].max()))
    neg_link_sampler = NegLinkSampler(num_nodes)

    tot_time = 0
    ptr_time = 0
    coo_time = 0
    sea_time = 0
    sam_time = 0
    uni_time = 0
    total_nodes = 0
    unique_nodes = 0
    for _, rows in tqdm(
        df.groupby(df.index // args.batch_size), total=len(df) // args.batch_size
    ):
        root_nodes = np.concatenate(
            [rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]
        ).astype(np.int32)
        ts = np.concatenate(
            [rows.time.values, rows.time.values, rows.time.values]
        ).astype(np.float32)
        sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        tot_time += ret[0].tot_time()
        ptr_time += ret[0].ptr_time()
        coo_time += ret[0].coo_time()
        sea_time += ret[0].search_time()
        sam_time += ret[0].sample_time()
        # for i in range(sample_config['history']):
        #     total_nodes += ret[i].dim_in() - ret[i].d_hid()
        #     unique_nodes += ret[i].dim_in() - ret[i].d_hid()
        #     if ret[i].dim_in() > ret[i].d_hid():
        #         ts = torch.from_numpy(ret[i].ts()[ret[i].d_hid():])
        #         nid = torch.from_numpy(ret[i].nodes()[ret[i].d_hid():]).float()
        #         nts = torch.stack([ts,nid],dim=1).cuda()
        #         uni_t_s = time.time()
        #         unts, idx = torch.unique(nts, dim=0, return_inverse=True)
        #         uni_time += time.time() - uni_t_s
        #         total_nodes += idx.shape[0]
        #         unique_nodes += unts.shape[0]

    print("total time  : {:.4f}".format(tot_time))
    print("pointer time: {:.4f}".format(ptr_time))
    print("coo time    : {:.4f}".format(coo_time))
    print("search time : {:.4f}".format(sea_time))
    print("sample time : {:.4f}".format(sam_time))
    # print('unique time : {:.4f}'.format(uni_time))
    # print('unique per  : {:.4f}'.format(unique_nodes / total_nodes))
