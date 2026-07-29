"""Dataset loading and shared-memory feature distribution.

The host process loads graph topology and node/edge features, places the
features into POSIX shared-memory segments, and broadcasts lightweight
metadata (shapes, dtypes, paths) to all worker ranks so they can attach to
the same segments without re-loading from disk.
"""

import logging
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from frost.data.graph_policy import (
    check_graph_direction,
    resolve_graph_artifact,
)
from frost.datatypes.env import Env
from frost.seq_sampler import TemporalGraph, TemporalGraphType
from frost.shm._core import create_shared_mem_array as csm
from frost.shm._core import get_shared_mem_array as gsm
from frost.utils.utils import broadcast_data

logger = logging.getLogger(__name__)
DATA_DIR = Path("DATA")
_NP_TO_TORCH = {
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
}


def _resolve_dataset_dir(dataset: str) -> Path:
    direct = DATA_DIR / dataset
    if direct.is_dir():
        return direct

    lower = DATA_DIR / dataset.lower()
    if lower.is_dir():
        return lower

    if DATA_DIR.is_dir():
        for child in DATA_DIR.iterdir():
            if child.is_dir() and child.name.lower() == dataset.lower():
                return child

    available = []
    if DATA_DIR.is_dir():
        available = sorted(child.name for child in DATA_DIR.iterdir() if child.is_dir())
    raise FileNotFoundError(
        f"Could not resolve dataset directory for {dataset} under {DATA_DIR}. "
        f"Available datasets: {available}"
    )


def _resolve_required_dataset_path(dataset: str, relative_paths: list[str]) -> Path:
    base = _resolve_dataset_dir(dataset)
    for relative_path in relative_paths:
        path = base / relative_path
        if path.exists():
            return path

    checked = [str(base / relative_path) for relative_path in relative_paths]
    raise FileNotFoundError(
        f"Could not resolve dataset file for {dataset}. Checked: {checked}"
    )


def _torch_dtype_from_numpy(np_dtype: np.dtype) -> torch.dtype:
    try:
        return _NP_TO_TORCH[np.dtype(np_dtype)]
    except KeyError as exc:
        raise TypeError(
            f"Unsupported NumPy dtype for shared memory: {np_dtype}"
        ) from exc


def _resolve_npy_alias_path(path: Path) -> Path:
    """Resolve text-alias `.npy` files to the real NumPy payload.

    Some dataset snapshots materialize a symlink-like indirection as a tiny
    UTF-8 text file whose contents are the relative target path, for example
    `node_features.npy -> node_role.npy`. Follow that chain until we land on an
    actual NumPy file.
    """

    seen_paths: set[Path] = set()
    current = path
    while True:
        if current in seen_paths:
            raise ValueError(
                f"Detected cyclic .npy alias indirection involving {current}"
            )
        seen_paths.add(current)

        with current.open("rb") as f:
            prefix = f.read(6)
        if prefix == b"\x93NUMPY":
            return current

        try:
            alias_text = current.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{current} is neither a valid NumPy file nor a UTF-8 alias."
            ) from exc

        if not alias_text:
            raise ValueError(f"Alias file {current} is empty.")

        target = (current.parent / alias_text).resolve()
        if not target.exists():
            raise FileNotFoundError(
                f"Alias file {current} points to missing target {target}."
            )
        current = target


def _expected_feature_rows(feat_type: str, n_items: int) -> int:
    if feat_type == "edge":
        return n_items + 1
    return n_items


def _validate_feature_rows(
    *,
    feat_type: str,
    path: Path,
    actual_rows: int,
    n_items: int,
) -> None:
    expected_rows = _expected_feature_rows(feat_type, n_items)
    if actual_rows == expected_rows:
        return

    if feat_type == "edge":
        raise ValueError(
            f"Feature row mismatch for {path}: expected {expected_rows} "
            f"(n_edges + 1, with row 0 reserved), got {actual_rows}"
        )

    raise ValueError(
        f"Feature row mismatch for {path}: expected {expected_rows}, got {actual_rows}"
    )


def _rows_per_chunk(feature_dim: int, target_dtype: torch.dtype) -> int:
    target_itemsize = torch.tensor([], dtype=target_dtype).element_size()
    bytes_per_row = max(feature_dim * target_itemsize, 1)
    return max(1, (128 * 1024 * 1024) // bytes_per_row)


def load_graph(
    d: str,
    index_dtype: np.dtype,
    time_dtype: np.dtype,
    graph_direction: str = "auto",
) -> tuple[TemporalGraphType, pd.DataFrame, int, int]:
    """Load a temporal graph from pre-processed NPZ and CSV files.

    Reads the single supported temporal-graph artifact and ``edges.csv``. Graph
    direction is checked against an explicit setting or TGL's official policy.

    Args:
        d: Dataset name; used as a directory name under ``DATA/``.

    Returns:
        A four-tuple of:
        - ``g``: :class:`~frost.seq_sampler.TemporalGraph` instance.
        - ``df``: Edge-list :class:`~pandas.DataFrame`.
        - ``n_nodes``: Total number of nodes.
        - ``n_edges``: Total number of edges.
    """
    dataset_dir = _resolve_dataset_dir(d)
    npz_path = resolve_graph_artifact(dataset_dir)
    csv_path = _resolve_required_dataset_path(d, ["edges.csv"])
    with np.load(npz_path, allow_pickle=False) as npz:
        indptr_arr = np.ascontiguousarray(npz["indptr"], dtype=index_dtype)
        indices_arr = np.ascontiguousarray(npz["indices"], dtype=index_dtype)
        eid_arr = np.ascontiguousarray(npz["eid"], dtype=index_dtype)
        ts_arr = np.ascontiguousarray(npz["ts"], dtype=time_dtype)

    df = pd.read_csv(csv_path)
    required_columns = {"eid", "src", "dst", "time", "default_split"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {d} is missing required edge columns: "
            f"{sorted(missing_columns)}. Found columns: {list(df.columns)}"
        )
    src = df["src"].to_numpy()
    dst = df["dst"].to_numpy()
    n_nodes_infer_from_df = max(src.max(), dst.max()) + 1

    n_nodes_infer_from_g = indptr_arr.shape[0] - 1
    assert n_nodes_infer_from_g == n_nodes_infer_from_df

    n_edges = df.shape[0]
    actual_direction = check_graph_direction(
        dataset=d,
        requested=graph_direction,
        n_edges=n_edges,
        n_adjacencies=len(indices_arr),
    )
    logger.info(
        "Loaded %s temporal graph for %s from %s",
        actual_direction,
        d,
        npz_path,
    )

    g = TemporalGraph(indptr=indptr_arr, indices=indices_arr, eid=eid_arr, ts=ts_arr)

    return g, df, n_nodes_infer_from_g, n_edges


def get_free_port() -> int:
    """Find a free port on the master node."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class DatasetManager:
    """Orchestrates dataset loading and feature sharing across distributed ranks.

    The host process loads the graph and features then writes them into shared
    memory; worker processes attach to the pre-populated segments after a
    barrier.  A single call to :meth:`prepare` handles the full protocol for
    whichever role the current rank plays.

    Args:
        env: Distributed environment descriptor.
        run_id: Prefix string used to name shared-memory segments.
        dataset: Name of the dataset directory under ``DATA/``.
    """

    def __init__(
        self,
        env: Env,
        run_id: str,
        dataset: str,  # from hydra config
        index_dtype: np.dtype,
        time_dtype: np.dtype,
        feature_dtype: torch.dtype,
        graph_direction: str = "auto",
        ignore_nfeat_on_disk: bool = False,
        ignore_efeat_on_disk: bool = False,
    ):
        self.env = env
        self.run_id = run_id
        self.dataset = dataset
        self.index_dtype = np.dtype(index_dtype)
        self.time_dtype = np.dtype(time_dtype)
        self.feature_dtype = feature_dtype
        self.graph_direction = graph_direction
        self.ignore_nfeat_on_disk = ignore_nfeat_on_disk
        self.ignore_efeat_on_disk = ignore_efeat_on_disk

    def prepare(
        self,
    ) -> tuple[
        TemporalGraphType | None,
        pd.DataFrame | None,
        int,
        int,
        int,
        int,
        torch.dtype | None,
        torch.dtype | None,
        torch.Tensor | None,
        torch.Tensor | None,
        int | None,
    ]:
        """
        Prepare graph data and features, sync across distributed ranks.
        Returns a tuple of all prepared data.
        """
        # Local variable initialization
        graph: TemporalGraphType | None = None
        df: pd.DataFrame | None = None
        nfeat: torch.Tensor | None = None
        efeat: torch.Tensor | None = None
        n_nodes = n_edges = ndim = edim = 0
        nfeat_rows = efeat_rows = 0
        ndtype: torch.dtype | None = None
        edtype: torch.dtype | None = None
        port: int | None = None
        metadata = None

        if self.env.is_host:
            # Load topology on host
            graph, df, n_nodes, n_edges = load_graph(
                self.dataset,
                index_dtype=self.index_dtype,
                time_dtype=self.time_dtype,
                graph_direction=self.graph_direction,
            )

            # Process features on host
            nfeat, ndim, ndtype = self._load_feature_to_shm(
                dataset=self.dataset,
                feat_type="node",
                shm_name="nfeat",
                expected_rows=n_nodes,
            )
            efeat, edim, edtype = self._load_feature_to_shm(
                dataset=self.dataset,
                feat_type="edge",
                shm_name="efeat",
                expected_rows=n_edges,
            )
            nfeat_rows = 0 if nfeat is None else int(nfeat.shape[0])
            efeat_rows = 0 if efeat is None else int(efeat.shape[0])

            # Prepare metadata for broadcasting
            port = get_free_port()
            metadata = (
                n_nodes,
                n_edges,
                ndim,
                edim,
                ndtype,
                edtype,
                nfeat_rows,
                efeat_rows,
                port,
            )

        # Synchronize across all ranks
        dist.barrier()

        # Broadcast metadata from host to all workers
        metadata = broadcast_data(metadata, src=self.env.host_rank)

        # Unpack metadata on all ranks (including workers)
        (
            n_nodes,
            n_edges,
            ndim,
            edim,
            ndtype,
            edtype,
            nfeat_rows,
            efeat_rows,
            port,
        ) = metadata

        if not self.env.is_host:
            # Attach shared memory on workers using broadcasted metadata
            nfeat = self._attach_shm("nfeat", [nfeat_rows, ndim], ndtype)
            efeat = self._attach_shm("efeat", [efeat_rows, edim], edtype)

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
            port,
        )

    def _resolve_feature_source(
        self, dataset: str, feat_type: str
    ) -> tuple[str, Path] | None:
        stem = f"{feat_type}_features"
        candidates = [
            (f"{stem}.npy", "npy"),
            (f"{stem}.pt", "pt"),
        ]
        base = _resolve_dataset_dir(dataset)
        for relative_path, source_kind in candidates:
            path = base / relative_path
            if path.exists():
                return source_kind, path
        return None

    def _load_feature_to_shm(
        self,
        dataset: str,
        feat_type: str,
        shm_name: str,
        expected_rows: int,
    ) -> tuple[torch.Tensor | None, int, torch.dtype | None]:
        """Load a feature matrix and materialize it directly into shared memory."""
        if feat_type == "node" and self.ignore_nfeat_on_disk:
            logger.info(
                "ignore_nfeat_on_disk=True; skipping on-disk node features for dataset %s.",
                dataset,
            )
            return None, 0, None
        if feat_type == "edge" and self.ignore_efeat_on_disk:
            logger.info(
                "ignore_efeat_on_disk=True; skipping on-disk edge features for dataset %s.",
                dataset,
            )
            return None, 0, None

        source = self._resolve_feature_source(dataset, feat_type)
        if source is None:
            logger.warning(
                "No %s feature found for dataset %s. Returning empty.",
                feat_type,
                dataset,
            )
            return None, 0, None

        source_kind, path = source
        logger.info(
            "Loading %s features for %s from %s (%s)",
            feat_type,
            dataset,
            path,
            source_kind,
        )
        try:
            if source_kind == "npy":
                return self._load_npy_feature_to_shm(
                    path, feat_type, shm_name, expected_rows
                )
            if source_kind == "pt":
                return self._load_pt_feature_to_shm(
                    path, feat_type, shm_name, expected_rows
                )
        except Exception as exc:
            logger.error("Failed to load %s feature from %s: %s", feat_type, path, exc)
            return None, 0, None

        raise ValueError(f"Unsupported feature source kind: {source_kind}")

    def _load_npy_feature_to_shm(
        self,
        path: Path,
        feat_type: str,
        shm_name: str,
        expected_rows: int,
    ) -> tuple[torch.Tensor, int, torch.dtype]:
        resolved_path = _resolve_npy_alias_path(path)
        if resolved_path != path:
            logger.info("Resolved %s alias %s -> %s", feat_type, path, resolved_path)
        arr = np.load(resolved_path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D feature array at {resolved_path}, got shape {arr.shape}"
            )
        _validate_feature_rows(
            feat_type=feat_type,
            path=resolved_path,
            actual_rows=arr.shape[0],
            n_items=expected_rows,
        )

        src_dtype = _torch_dtype_from_numpy(arr.dtype)
        final_dtype = self.feature_dtype
        if src_dtype == torch.bool:
            final_dtype = self.feature_dtype
        feat_shm = csm(f"{self.run_id}_{shm_name}", arr.shape, dtype=final_dtype)
        rows_per_chunk = _rows_per_chunk(arr.shape[1], final_dtype)
        for start in range(0, arr.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, arr.shape[0])
            chunk_np = np.array(arr[start:end], copy=True)
            chunk = torch.from_numpy(chunk_np).to(dtype=final_dtype)
            feat_shm[start:end].copy_(chunk)
        return feat_shm, arr.shape[1], final_dtype

    def _load_pt_feature_to_shm(
        self,
        path: Path,
        feat_type: str,
        shm_name: str,
        expected_rows: int,
    ) -> tuple[torch.Tensor, int, torch.dtype]:
        feat = torch.load(path, map_location="cpu")
        if not isinstance(feat, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {path}, got {type(feat)!r}")
        if feat.ndim != 2:
            raise ValueError(
                f"Expected 2D feature tensor at {path}, got shape {tuple(feat.shape)}"
            )
        _validate_feature_rows(
            feat_type=feat_type,
            path=path,
            actual_rows=int(feat.shape[0]),
            n_items=expected_rows,
        )

        final_dtype = self.feature_dtype
        if feat.dtype != final_dtype:
            feat = feat.to(final_dtype)
        feat_shm = csm(f"{self.run_id}_{shm_name}", feat.shape, dtype=final_dtype)
        feat_shm.copy_(feat)
        return feat_shm, feat.shape[1], final_dtype

    def _create_shm(self, name, data_np, shape):
        """Host: Allocate and copy data to SHM."""
        if data_np is None or shape[1] <= 0:
            return None
        feat_shm = csm(f"{self.run_id}_{name}", shape, dtype=data_np.dtype)
        feat_shm.copy_(data_np)
        return feat_shm

    def _attach_shm(self, name, shape, dtype):
        """Worker: Attach to the SHM created by Host."""
        if shape[0] <= 0 or shape[1] <= 0 or dtype is None:
            return None
        return gsm(f"{self.run_id}_{name}", shape, dtype=dtype)
