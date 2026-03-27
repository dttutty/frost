import argparse
import itertools
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to edges.csv")
parser.add_argument("--add_reverse", default=True, action="store_true")
parser.add_argument(
    "--tqdm", action="store_true", default=True, help="enable tqdm progress bars"
)
args = parser.parse_args()

df = pd.read_csv(args.data, header=0)

if {"u", "i", "idx"}.issubset(df.columns):
    src_col = "u"
    dst_col = "i"
    eid_col = "idx"
elif {"src", "dst", "eid"}.issubset(df.columns):
    src_col = "src"
    dst_col = "dst"
    eid_col = "eid"
else:
    raise ValueError(
        "edges.csv must contain either {u, i, idx} or {src, dst, eid} columns. "
        f"Got: {list(df.columns)}"
    )

if "ts" not in df.columns:
    raise ValueError(f"edges.csv must contain a ts column. Got: {list(df.columns)}")

num_nodes = max(int(df[src_col].max()), int(df[dst_col].max())) + 1
print("num_nodes: ", num_nodes)

full_graph_with_reverse_edges_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
full_graph_with_reverse_edges_indices = [[] for _ in range(num_nodes)]
full_graph_with_reverse_edges_ts = [[] for _ in range(num_nodes)]
full_graph_with_reverse_edges_eid = [[] for _ in range(num_nodes)]

edge_iter = tqdm(df.iterrows(), total=len(df)) if args.tqdm else df.iterrows()
for idx, row in edge_iter:
    src = int(row[src_col])
    dst = int(row[dst_col])
    ts = int(row["ts"])
    eid_val = int(row[eid_col])
    full_graph_with_reverse_edges_indices[src].append(dst)
    full_graph_with_reverse_edges_ts[src].append(ts)
    full_graph_with_reverse_edges_eid[src].append(eid_val)
    if args.add_reverse:
        full_graph_with_reverse_edges_indices[dst].append(src)
        full_graph_with_reverse_edges_ts[dst].append(ts)
        full_graph_with_reverse_edges_eid[dst].append(eid_val)

node_iter = tqdm(range(num_nodes)) if args.tqdm else range(num_nodes)
for i in node_iter:
    full_graph_with_reverse_edges_indptr[i + 1] = full_graph_with_reverse_edges_indptr[
        i
    ] + len(full_graph_with_reverse_edges_indices[i])

full_graph_with_reverse_edges_indices = np.array(
    list(itertools.chain(*full_graph_with_reverse_edges_indices)), dtype=np.int64
)
full_graph_with_reverse_edges_ts = np.array(
    list(itertools.chain(*full_graph_with_reverse_edges_ts)), dtype=np.int64
)
full_graph_with_reverse_edges_eid = np.array(
    list(itertools.chain(*full_graph_with_reverse_edges_eid)), dtype=np.int64
)

print("Sorting...")


def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    local_indices = indices[beg:end]
    local_t = t[beg:end]
    local_eid = eid[beg:end]
    # Impose a total order so ties on timestamp are deterministic across runs.
    sidx = np.lexsort((local_indices, local_eid, local_t))
    indices[beg:end] = local_indices[sidx]
    t[beg:end] = local_t[sidx]
    eid[beg:end] = local_eid[sidx]


sort_iter = (
    tqdm(range(full_graph_with_reverse_edges_indptr.shape[0] - 1))
    if args.tqdm
    else range(full_graph_with_reverse_edges_indptr.shape[0] - 1)
)
for i in sort_iter:
    tsort(
        i,
        full_graph_with_reverse_edges_indptr,
        full_graph_with_reverse_edges_indices,
        full_graph_with_reverse_edges_ts,
        full_graph_with_reverse_edges_eid,
    )

# import pdb; pdb.set_trace()
print("saving...")
output_dir = os.path.dirname(args.data)
np.savez(
    os.path.join(
        output_dir,
        "full_graph_with_reverse_edge.npz"
        if args.add_reverse
        else "full_graph_no_reverse_edge.npz",
    ),
    indptr=full_graph_with_reverse_edges_indptr,
    indices=full_graph_with_reverse_edges_indices,
    ts=full_graph_with_reverse_edges_ts,
    eid=full_graph_with_reverse_edges_eid,
)
