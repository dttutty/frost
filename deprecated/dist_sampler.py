# dist_sampler.py

import numpy as np
import torch
import torch.distributed as dist

from deprecated.sampler_core import ParallelSampler


class DistSampler:
    """
    Wraps sampler_core.ParallelSampler and maintains a shared ts_ptr table.
    Each rank proposes sparse pointer updates; we all-gather and merge via max.
    """

    def __init__(
        self,
        core: ParallelSampler,
        indptr: np.ndarray,
        num_history: int,
        world_size: int,
        rank: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.core = core
        self.world_size = world_size
        self.rank = rank
        self.num_history = num_history
        num_nodes = indptr.size - 1
        base = torch.from_numpy(indptr[:-1].astype(np.int64))  # shape [N]
        ts_ptr = base.unsqueeze(0).repeat(num_history + 1, 1).contiguous()  # [H+1, N]
        # share memory on a single node; still fine if multi-node since we reduce explicitly
        self.ts_ptr = ts_ptr
        self.device = device

    @torch.no_grad()
    def sample_with_ptr(self, root_nodes: np.ndarray, root_ts: np.ndarray):
        """
        Call C++ sampler with external ts_ptr and get sparse pointer updates.
        Returns:
            blocks: sampler_core get_ret()
            sparse_updates: Dict[slice_idx -> (nodes_np, vals_np)]
        """
        # pass flattened view to C++
        flat_ptr = self.ts_ptr.view(-1).numpy()
        upd_nodes_list, upd_vals_list = self.core.sample_with_ptr(
            root_nodes.astype(np.int32), root_ts.astype(np.float32), flat_ptr
        )
        blocks = self.core.get_ret()
        # Build dict of updates per slice
        sparse_updates = {}
        for slc, (n_arr, v_arr) in enumerate(zip(upd_nodes_list, upd_vals_list)):
            if len(n_arr) == 0:
                continue
            sparse_updates[slc] = (
                np.asarray(n_arr, dtype=np.int64),
                np.asarray(v_arr, dtype=np.int64),
            )
        return blocks, sparse_updates

    @torch.no_grad()
    def sync_and_apply_updates(
        self, local_updates: dict[int, tuple[np.ndarray, np.ndarray]]
    ):
        """
        Sparse all-gather and merge: for each slice, concat all pairs and take max per node.
        """
        # Convert dict to plain Python object for all_gather_object
        serializable = {
            int(k): (local_updates[k][0].tolist(), local_updates[k][1].tolist())
            for k in local_updates
        }
        gathered: list[dict] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, serializable)

        # Merge updates per slice
        for slc in range(self.num_history + 1):
            nodes_cat = []
            vals_cat = []
            for d in gathered:
                if d is None:
                    continue
                if slc in d:
                    n_list, v_list = d[slc]
                    if len(n_list) == 0:
                        continue
                    nodes_cat.append(np.asarray(n_list, dtype=np.int64))
                    vals_cat.append(np.asarray(v_list, dtype=np.int64))
            if not nodes_cat:
                continue
            nodes_cat = np.concatenate(nodes_cat)
            vals_cat = np.concatenate(vals_cat)
            # deduplicate by node: take max value per node
            order = np.argsort(nodes_cat, kind="mergesort")
            nodes_cat = nodes_cat[order]
            vals_cat = vals_cat[order]
            unique_nodes, first_idx = np.unique(nodes_cat, return_index=True)
            # max per node segment
            max_vals = []
            for i in range(len(unique_nodes)):
                start = first_idx[i]
                end = first_idx[i + 1] if i + 1 < len(unique_nodes) else len(nodes_cat)
                max_vals.append(np.max(vals_cat[start:end]))
            max_vals = np.asarray(max_vals, dtype=np.int64)
            # apply to shared table
            self.ts_ptr[slc, torch.from_numpy(unique_nodes)] = torch.maximum(
                self.ts_ptr[slc, torch.from_numpy(unique_nodes)],
                torch.from_numpy(max_vals),
            )
