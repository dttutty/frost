import argparse
import itertools
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="dataset name")
parser.add_argument("--add_reverse", default=False, action="store_true")
args = parser.parse_args()

df = pd.read_csv(f"DATA/{args.data}/edges.csv")
num_nodes = max(int(df["src"].max()), int(df["dst"].max())) + 1
logger.debug("num_nodes: ", num_nodes)


class Dummy:
    def __init__(self, num_nodes: int, add_reverse: bool, dtype: type = int) -> None:
        self.num_nodes = num_nodes
        self.add_reverse = add_reverse
        self.indptr: np.ndarray = np.zeros(num_nodes + 1, dtype=dtype)
        self.indices: list[list[int]] = [[] for _ in range(num_nodes)]
        self.ts: list[list[float]] = [[] for _ in range(num_nodes)]
        self.eid: list[list[int]] = [[] for _ in range(num_nodes)]

    def append(self, src: int, dst: int, time: float, idx: int) -> None:
        self.indices[src].append(dst)
        self.ts[src].append(time)
        self.eid[src].append(idx)
        if self.add_reverse:
            self.indices[dst].append(src)
            self.ts[dst].append(time)
            self.eid[dst].append(idx)
        # self.indptr[src + 1:] += 1

    def save(self, filename: str) -> None:
        for i in tqdm(range(num_nodes)):
            self.indptr[i + 1] = self.indptr[i] + len(self.indices[i])
            self.indptr[i + 1] = self.indptr[i] + len(self.indices[i])
            self.indptr[i + 1] = self.indptr[i] + len(self.indices[i])

        def flatten_to_array(list_of_lists: list[list[int | float]]) -> np.ndarray:
            # chain.from_iterable is a bit leaner than chain(*…)
            return np.array(list(itertools.chain.from_iterable(list_of_lists)))

        indices = flatten_to_array(self.indices)
        ts = flatten_to_array(self.ts)
        eid = flatten_to_array(self.eid)

        def tsort(
            i: int,
            indptr: np.ndarray,
            indices: np.ndarray,
            t: np.ndarray,
            eid: np.ndarray,
        ) -> None:
            beg = indptr[i]
            end = indptr[i + 1]
            sidx = np.argsort(t[beg:end])
            indices[beg:end] = indices[beg:end][sidx]
            t[beg:end] = t[beg:end][sidx]
            eid[beg:end] = eid[beg:end][sidx]

        logger.debug("Sorting...")
        for i in tqdm(range(self.indptr.shape[0] - 1)):
            tsort(i, self.indptr, indices, ts, eid)
        logger.debug("Saving...")
        # import pdb; pdb.set_trace()
        np.savez(
            f"DATA/{args.data}/{filename}.npz",
            indptr=self.indptr,
            indices=indices,
            ts=ts,
            eid=eid,
        )


int_train_dummay = Dummy(num_nodes, args.add_reverse)
int_full_dummay = Dummy(num_nodes, args.add_reverse)
ext_full_dummay = Dummy(num_nodes, args.add_reverse)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row["src"])
    dst = int(row["dst"])
    if row["int_roll"] == 0:
        int_train_dummay.append(src, dst, row["time"], idx)
    if row["int_roll"] != 3:
        int_full_dummay.append(src, dst, row["time"], idx)
    ext_full_dummay.append(src, dst, row["time"], idx)

int_train_dummay.save("int_train")
int_full_dummay.save("int_full")
ext_full_dummay.save("ext_full")
