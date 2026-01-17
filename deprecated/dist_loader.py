# dist_loader.py
import math
import numpy as np
import torch
import torch.distributed as dist
from typing import Iterator, Optional
import pandas as pd

def plan_gpu_batches(num_samples: int, target_batch_size: int, ngpus: int):
    approx_iterations = num_samples // target_batch_size
    nbatch_per_gpu = max(approx_iterations // ngpus, 1)
    iteration_total = nbatch_per_gpu * ngpus
    adjusted_batch_size = math.ceil(num_samples / iteration_total)
    return iteration_total, adjusted_batch_size

def batches_assign(index: np.ndarray, batch_size: int):
    return index // batch_size

def generate_reordered_indexes_1(reorder: int, batch_size: int, num_samples: int, base_assign: np.ndarray):
    group_idx = []
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    chunk = batch_size // reorder
    group_idx = np.repeat(np.array(group_idx), chunk)
    tiles = num_samples // batch_size + 1
    group_idx = np.tile(group_idx, tiles)[:num_samples]
    return base_assign + group_idx

def generate_reordered_indexes_2(reorder: int, batch_size: int, base_assign: np.ndarray):
    out = []
    chunk = batch_size // reorder
    for i in range(1, reorder):
        pad = np.zeros(chunk * i) - 1
        out.append(np.concatenate([pad, base_assign])[: base_assign.shape[0]])
    return out

class TemporalEdgesDataset:
    def __init__(self, df: pd.DataFrame, batch_size: int, reorder: Optional[int] = None, seed: int = 0):
        self.df = df
        self.batch_size = batch_size
        self.reorder = reorder
        self.seed = seed

    def _build_assignment(self, world_size: int):
        idx = self.df.index.values
        base = batches_assign(idx, self.batch_size)
        groups = [base]
        if self.reorder is not None and self.reorder > 1:
            groups.append(generate_reordered_indexes_1(self.reorder, self.batch_size, len(self.df), base))
            groups.extend(generate_reordered_indexes_2(self.reorder, self.batch_size, base))
        return groups

    def iter_rank(self, rank: int, world_size: int) -> Iterator[pd.DataFrame]:
        groups = self._build_assignment(world_size)
        g = groups[0]  # deterministic; or randomly pick with np.random.RandomState(self.seed)
        # split batches across ranks round-robin
        uniq = np.unique(g[g >= 0])
        for bi in uniq:
            if (bi % world_size) != rank:
                continue
            mask = (g == bi)
            batch_df = self.df[mask]
            if not batch_df.empty:
                yield batch_df
