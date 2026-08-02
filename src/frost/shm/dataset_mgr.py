"""Dataset loading and shared-memory feature distribution.

The host process loads graph topology and node/edge features, places ordinary
features into POSIX shared-memory segments, and keeps large PyTorch node
features file-backed. It broadcasts lightweight metadata so worker ranks can
attach to the shared storage without materializing duplicate feature tables.
"""

import logging
import mmap
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
from frost.data.paths import (
    is_raw_tgb_dataset,
    resolve_artifact_dataset_dir,
    resolve_tgb_raw_dataset_dir,
)
from frost.data.tgb import (
    load_tgb_features,
    load_tgb_runtime_graph,
)
from frost.datatypes.env import Env
from frost.seq_sampler import TemporalGraph, TemporalGraphType
from frost.shm._core import create_shared_mem_array as csm
from frost.shm._core import get_shared_mem_array as gsm
from frost.utils.utils import broadcast_data

logger = logging.getLogger(__name__)
_COMPACT_FEATURE_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.float16,
    torch.bfloat16,
}
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


def _resolve_required_dataset_path(dataset: str, relative_paths: list[str]) -> Path:
    base = resolve_artifact_dataset_dir(dataset)
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


def _feature_storage_dtype(
    source_dtype: torch.dtype, compute_dtype: torch.dtype
) -> torch.dtype:
    """Keep naturally compact features compact until selected rows reach a worker."""
    if source_dtype in _COMPACT_FEATURE_DTYPES:
        return source_dtype
    return compute_dtype


def _normalize_edge_frame(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize upstream TGL edge tables to FROST's canonical runtime schema."""
    if "time" not in frame.columns and "ts" in frame.columns:
        frame = frame.rename(columns={"ts": "time"})
    if "default_split" not in frame.columns and "ext_roll" in frame.columns:
        frame["default_split"] = frame["ext_roll"]
    if "eid" not in frame.columns:
        frame["eid"] = np.arange(1, len(frame) + 1, dtype=np.int64)

    required_columns = {"eid", "src", "dst", "time", "default_split"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {dataset} is missing required edge columns: "
            f"{sorted(missing_columns)}. Found columns: {list(frame.columns)}"
        )
    return frame.loc[:, ["eid", "src", "dst", "time", "default_split"]]


def _resolve_npy_alias_path(path: Path) -> Path:
    """Resolve text-alias `.npy` files to the real NumPy payload.

    Some dataset snapshots materialize a symlink-like indirection as a tiny
    UTF-8 text file whose contents are the relative target path. Follow that
    chain until we land on an actual NumPy file.
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
    if (
        actual_rows == expected_rows
        or (feat_type == "edge" and actual_rows == n_items)
        or (feat_type == "node" and actual_rows > n_items)
    ):
        return

    if feat_type == "edge":
        raise ValueError(
            f"Feature row mismatch for {path}: expected {expected_rows} "
            f"(n_edges + 1, with row 0 reserved), got {actual_rows}"
        )

    raise ValueError(
        f"Feature row mismatch for {path}: expected {expected_rows}, got {actual_rows}"
    )


def _rows_per_chunk(
    feature_dim: int,
    target_dtype: torch.dtype,
    *,
    source_itemsize: int | None = None,
) -> int:
    target_itemsize = torch.tensor([], dtype=target_dtype).element_size()
    itemsize = max(target_itemsize, source_itemsize or 0)
    bytes_per_row = max(feature_dim * itemsize, 1)
    return max(1, (128 * 1024 * 1024) // bytes_per_row)


def _load_mmap_pt_tensor(path: Path) -> torch.Tensor:
    """Map a tensor checkpoint without reserving private copy-on-write memory."""
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        value = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(value)!r}")
    return value


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
        d: Runtime dataset name resolved through its registered source.

    Returns:
        A four-tuple of:
        - ``g``: :class:`~frost.seq_sampler.TemporalGraph` instance.
        - ``df``: Edge-list :class:`~pandas.DataFrame`.
        - ``n_nodes``: Total number of nodes.
        - ``n_edges``: Total number of edges.
    """
    if is_raw_tgb_dataset(d):
        raw_dir = resolve_tgb_raw_dataset_dir(d)
        runtime_graph = load_tgb_runtime_graph(raw_dir, d)
        arrays = runtime_graph.arrays
        indptr_arr = np.ascontiguousarray(arrays["indptr"], dtype=index_dtype)
        indices_arr = np.ascontiguousarray(arrays["indices"], dtype=index_dtype)
        eid_arr = np.ascontiguousarray(arrays["eid"], dtype=index_dtype)
        ts_arr = np.ascontiguousarray(arrays["ts"], dtype=time_dtype)
        df = runtime_graph.frame
        source_description = raw_dir / "ml_tgbl-wiki.pkl"
    else:
        dataset_dir = resolve_artifact_dataset_dir(d)
        npz_path = resolve_graph_artifact(dataset_dir)
        csv_path = _resolve_required_dataset_path(d, ["edges.csv", "edges.csv.zst"])
        with np.load(npz_path, allow_pickle=False) as npz:
            indptr_arr = np.ascontiguousarray(npz["indptr"], dtype=index_dtype)
            indices_arr = np.ascontiguousarray(npz["indices"], dtype=index_dtype)
            eid_arr = np.ascontiguousarray(npz["eid"], dtype=index_dtype)
            ts_arr = np.ascontiguousarray(npz["ts"], dtype=time_dtype)
        df = _normalize_edge_frame(d, pd.read_csv(csv_path))
        source_description = npz_path
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
        source_description,
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
        dataset: Runtime dataset name resolved through its registered source.
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
        feature_numa_node: int | None = None,
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
        self.feature_numa_node = feature_numa_node
        self._file_backed_feature_paths: dict[str, str] = {}

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
        nfeat_backing_path: str | None = None
        efeat_backing_path: str | None = None
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
            if nfeat is not None:
                # Feature tables may include isolated nodes absent from the
                # topology CSR. They still define valid global node IDs and
                # therefore participate in model-state capacity.
                n_nodes = max(n_nodes, int(nfeat.shape[0]))
            efeat, edim, edtype = self._load_feature_to_shm(
                dataset=self.dataset,
                feat_type="edge",
                shm_name="efeat",
                expected_rows=n_edges,
            )
            nfeat_rows = 0 if nfeat is None else int(nfeat.shape[0])
            efeat_rows = 0 if efeat is None else int(efeat.shape[0])
            nfeat_backing_path = self._file_backed_feature_paths.get("node")
            efeat_backing_path = self._file_backed_feature_paths.get("edge")

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
                nfeat_backing_path,
                efeat_backing_path,
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
            nfeat_backing_path,
            efeat_backing_path,
            port,
        ) = metadata

        if not self.env.is_host:
            # Attach shared memory on workers using broadcasted metadata
            nfeat = self._attach_feature(
                "nfeat",
                [nfeat_rows, ndim],
                ndtype,
                backing_path=nfeat_backing_path,
            )
            efeat = self._attach_feature(
                "efeat",
                [efeat_rows, edim],
                edtype,
                backing_path=efeat_backing_path,
            )

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
        base = resolve_artifact_dataset_dir(dataset)
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

        if is_raw_tgb_dataset(dataset):
            raw_features = load_tgb_features(
                resolve_tgb_raw_dataset_dir(dataset), dataset, feat_type
            )
            if raw_features is None:
                logger.info(
                    "Official TGB provides no %s features for dataset %s.",
                    feat_type,
                    dataset,
                )
                return None, 0, None
            return self._load_tgb_feature_to_shm(
                dataset=dataset,
                feat_type=feat_type,
                shm_name=shm_name,
                expected_rows=expected_rows,
                raw_features=raw_features,
            )

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

    def _load_tgb_feature_to_shm(
        self,
        *,
        dataset: str,
        feat_type: str,
        shm_name: str,
        expected_rows: int,
        raw_features: np.ndarray,
    ) -> tuple[torch.Tensor, int, torch.dtype]:
        raw_dir = resolve_tgb_raw_dataset_dir(dataset)
        if raw_features.shape[0] != expected_rows:
            raise ValueError(
                f"Official TGB {feat_type}-feature rows do not match the "
                f"table: {raw_features.shape[0]} != {expected_rows}"
            )
        output_offset = int(feat_type == "edge")
        output_shape = (expected_rows + output_offset, raw_features.shape[1])

        final_dtype = self.feature_dtype
        feat_shm = csm(
            f"{self.run_id}_{shm_name}",
            output_shape,
            dtype=final_dtype,
            numa_node=self.feature_numa_node,
        )
        if output_offset:
            feat_shm[:1].zero_()
        rows_per_chunk = _rows_per_chunk(
            raw_features.shape[1],
            final_dtype,
            source_itemsize=raw_features.dtype.itemsize,
        )
        for start in range(0, raw_features.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, raw_features.shape[0])
            chunk = torch.from_numpy(np.array(raw_features[start:end], copy=True)).to(
                dtype=final_dtype
            )
            feat_shm[start + output_offset : end + output_offset].copy_(chunk)
        logger.info(
            "Built %s features for %s in shared memory from %s",
            feat_type,
            dataset,
            raw_dir,
        )
        return feat_shm, raw_features.shape[1], final_dtype

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

        source_dtype = _torch_dtype_from_numpy(arr.dtype)
        final_dtype = _feature_storage_dtype(source_dtype, self.feature_dtype)
        output_offset = int(feat_type == "edge" and arr.shape[0] == expected_rows)
        output_shape = (arr.shape[0] + output_offset, arr.shape[1])
        feat_shm = csm(
            f"{self.run_id}_{shm_name}",
            output_shape,
            dtype=final_dtype,
            numa_node=self.feature_numa_node,
        )
        if output_offset:
            feat_shm[0].zero_()
        rows_per_chunk = _rows_per_chunk(
            arr.shape[1], final_dtype, source_itemsize=arr.dtype.itemsize
        )
        for start in range(0, arr.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, arr.shape[0])
            chunk_np = np.array(arr[start:end], copy=True)
            chunk = torch.from_numpy(chunk_np).to(dtype=final_dtype)
            feat_shm[start + output_offset : end + output_offset].copy_(chunk)
        return feat_shm, arr.shape[1], final_dtype

    def _load_pt_feature_to_shm(
        self,
        path: Path,
        feat_type: str,
        shm_name: str,
        expected_rows: int,
    ) -> tuple[torch.Tensor, int, torch.dtype]:
        feat = _load_mmap_pt_tensor(path)
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

        final_dtype = _feature_storage_dtype(feat.dtype, self.feature_dtype)
        if feat_type == "node" and feat.dtype == final_dtype:
            self._file_backed_feature_paths[feat_type] = str(path.resolve())
            logger.info(
                "Using file-backed mmap for %s features from %s",
                feat_type,
                path,
            )
            return feat, feat.shape[1], final_dtype

        if feat.dtype != final_dtype:
            feat = feat.to(final_dtype)
        output_offset = int(feat_type == "edge" and feat.shape[0] == expected_rows)
        output_shape = (feat.shape[0] + output_offset, feat.shape[1])
        feat_shm = csm(
            f"{self.run_id}_{shm_name}",
            output_shape,
            dtype=final_dtype,
            numa_node=self.feature_numa_node,
        )
        if output_offset:
            feat_shm[0].zero_()
        feat_shm[output_offset:].copy_(feat)
        return feat_shm, feat.shape[1], final_dtype

    def _attach_feature(
        self,
        name: str,
        shape: list[int],
        dtype: torch.dtype | None,
        *,
        backing_path: str | None,
    ) -> torch.Tensor | None:
        if backing_path is None:
            return self._attach_shm(name, shape, dtype)
        if dtype is None:
            raise ValueError(f"File-backed feature {name} is missing its dtype")

        feature = _load_mmap_pt_tensor(Path(backing_path))
        if list(feature.shape) != shape:
            raise ValueError(
                f"File-backed feature shape changed for {backing_path}: "
                f"expected {shape}, got {list(feature.shape)}"
            )
        if feature.dtype != dtype:
            raise ValueError(
                f"File-backed feature dtype changed for {backing_path}: "
                f"expected {dtype}, got {feature.dtype}"
            )
        return feature

    def _attach_shm(self, name, shape, dtype):
        """Worker: Attach to the SHM created by Host."""
        if shape[0] <= 0 or shape[1] <= 0 or dtype is None:
            return None
        return gsm(f"{self.run_id}_{name}", shape, dtype=dtype)
