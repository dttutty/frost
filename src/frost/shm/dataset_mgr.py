import logging
import os
import socket
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from frost.datatypes.env import Env
from frost.seq_sampler import TemporalGraph, TemporalGraphType
from frost.shm._core import create_shared_mem_array as csm
from frost.shm._core import get_shared_mem_array as gsm
from frost.utils.utils import broadcast_data

logger = logging.getLogger(__name__)


def load_graph(d: str) -> tuple[TemporalGraphType, pd.DataFrame, int, int]:
    npz = np.load(f"DATA/{d}/ext_full.npz")

    # Force arrays to be C-contiguous with correct dtypes
    # int32 is enough for MAG and GDELT, you can adjust if your dataset has more nodes/edges (number > 2^31)
    indptr_arr = np.ascontiguousarray(npz["indptr"], dtype=np.int32)
    indices_arr = np.ascontiguousarray(npz["indices"], dtype=np.int32)
    eid_arr = np.ascontiguousarray(npz["eid"], dtype=np.int32)
    ts_arr = np.ascontiguousarray(npz["ts"], dtype=np.float32)

    # Initialize TemporalGraph directly
    g = TemporalGraph(indptr=indptr_arr, indices=indices_arr, eid=eid_arr, ts=ts_arr)

    df = pd.read_csv(f"DATA/{d}/edges.csv")
    src = df["src"].to_numpy()
    dst = df["dst"].to_numpy()
    n_nodes_infer_from_df = max(src.max(), dst.max()) + 1

    n_nodes_infer_from_g = indptr_arr.shape[0] - 1
    assert n_nodes_infer_from_g == n_nodes_infer_from_df

    n_edges = df.shape[0]

    return g, df, n_nodes_infer_from_g, n_edges


def get_free_port() -> int:
    """在 master 节点上获取一个空闲端口"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class DatasetManager:
    def __init__(
        self,
        env: Env,
        run_id: str,
        dataset_name: str,  # from hydra config
        target_ndim: int | None,  # from hydra config
        target_edim: int | None,  # from hydra config
    ):
        self.env = env
        self.run_id = run_id
        self.dataset = dataset_name
        self.target_ndim = target_ndim
        self.target_edim = target_edim

    def prepare(self):
        """
        Prepare graph data and features, sync across distributed ranks.
        Returns a tuple of all prepared data.
        """
        # Local variable initialization
        graph = df = nfeat = efeat = None
        n_nodes = n_edges = ndim = edim = 0
        ndtype = edtype = save_to = port = None
        metadata = None

        if self.env.is_host:
            # Load topology on host
            graph, df, n_nodes, n_edges = load_graph(self.dataset)

            # Process features on host
            node_np, ndim, ndtype = self._process_feature(
                self.dataset, "node", n_nodes, self.target_ndim
            )
            edge_np, edim, edtype = self._process_feature(
                self.dataset, "edge", n_edges, self.target_edim
            )

            # Create shared memory on host
            nfeat = self._create_shm("nfeat", node_np, [n_nodes, ndim])
            efeat = self._create_shm("efeat", edge_np, [n_edges, edim])

            # Prepare metadata for broadcasting
            save_to = self._get_save_path(self.dataset)
            port = get_free_port()
            metadata = (n_nodes, n_edges, ndim, edim, ndtype, edtype, save_to, port)

        # Synchronize across all ranks
        dist.barrier()

        # Broadcast metadata from host to all workers
        metadata = broadcast_data(metadata, src=self.env.host_rank)

        # Unpack metadata on all ranks (including workers)
        (n_nodes, n_edges, ndim, edim, ndtype, edtype, save_to, port) = metadata

        if not self.env.is_host:
            # Attach shared memory on workers using broadcasted metadata
            nfeat = self._attach_shm("nfeat", [n_nodes, ndim], ndtype)
            efeat = self._attach_shm("efeat", [n_edges, edim], edtype)

        dist.barrier()  # Ensure all ranks have attached SHM before proceeding

        return (
            graph,  # Only non-None on host
            df,  # Only non-None on host
            n_nodes,
            n_edges,
            ndim,
            edim,
            ndtype,
            edtype,
            nfeat,  # Accessible on all ranks (via SHM)
            efeat,  # Accessible on all ranks (via SHM)
            save_to,
            port,
        )

    def _process_feature(self, dataset, feat_type, count, target_dim):
        """
        Unified feature loading logic.
        target_dim: 0 -> Skip | >0 -> Random | None -> Disk
        """
        if target_dim == 0:
            return None, 0, None

        if target_dim is not None and target_dim > 0:
            # Generate random features
            feat = torch.randn(count, target_dim, dtype=torch.float32)
            return feat, target_dim, feat.dtype

        # Load from disk
        path = f"DATA/{dataset}/{feat_type}_features.pt"
        if not os.path.exists(path):
            logger.warning(f"{feat_type} feature not found at {path}. Returning empty.")
            return None, 0, None

        try:
            feat = torch.load(path)
            if feat.dtype == torch.bool:
                feat = feat.type(torch.float32)
            return feat, feat.shape[1], feat.dtype
        except Exception as e:
            logger.error(f"Failed to load {feat_type} feature: {e}")
            return None, 0, None

    def _create_shm(self, name, data_np, shape):
        """Host: Allocate and copy data to SHM."""
        if data_np is None or shape[1] <= 0:
            return None
        feat_shm = csm(f"{self.run_id}_{name}", shape, dtype=data_np.dtype)
        feat_shm.copy_(data_np)
        return feat_shm

    def _attach_shm(self, name, shape, dtype):
        """Worker: Attach to the SHM created by Host."""
        if shape[1] <= 0 or dtype is None:
            return None
        return gsm(f"{self.run_id}_{name}", shape, dtype=dtype)

    def _get_save_path(self, dataset):
        """Generate formatted path for model saving."""
        if not os.path.isdir("models"):
            os.makedirs("models", exist_ok=True)
        timestamp = time.strftime("%m%d_%H%M")
        return f"models/{dataset}_{timestamp}.pkl"
