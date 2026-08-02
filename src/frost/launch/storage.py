from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from omegaconf import DictConfig

from frost.datatypes.env import Env
from frost.models.contracts import RootExpansion
from frost.seq_sampler import TemporalGraphType
from frost.shm.dataset_mgr import DatasetManager
from frost.shm.control_plane import SharedControlPlane, control_payload_capacity
from frost.shm.mem_mgr import MemBoxStore, MemBoxStoreManager
from frost.shm.ring_buf import RingBuf, cal_max_buf_size_for_layouts
from frost.utils.dtypes import RuntimeDTypes


@dataclass(frozen=True)
class BatchLayout:
    train_per_gpu: int
    eval_macro: int
    eval_per_gpu: int


@dataclass(frozen=True)
class LaunchDataset:
    graph: TemporalGraphType | None
    frame: pd.DataFrame | None
    n_nodes: int
    ndim: int
    edim: int
    nfeat: torch.Tensor | None
    efeat: torch.Tensor | None


def resolve_batch_layout(
    *, macro_batch_size: int, eval_macro_batch_size: int | None, n_gpu: int
) -> BatchLayout:
    if macro_batch_size % n_gpu != 0:
        raise ValueError(
            f"macro_batch_size ({macro_batch_size}) must be divisible by "
            f"n_gpu ({n_gpu})."
        )
    resolved_eval_macro = (
        macro_batch_size
        if eval_macro_batch_size is None
        else int(eval_macro_batch_size)
    )
    if resolved_eval_macro < n_gpu:
        raise ValueError(
            f"eval_macro_batch_size ({resolved_eval_macro}) must be at least "
            f"n_gpu ({n_gpu})."
        )
    if resolved_eval_macro % n_gpu != 0:
        raise ValueError(
            f"eval_macro_batch_size ({resolved_eval_macro}) must be divisible by "
            f"n_gpu ({n_gpu})."
        )
    return BatchLayout(
        train_per_gpu=macro_batch_size // n_gpu,
        eval_macro=resolved_eval_macro,
        eval_per_gpu=resolved_eval_macro // n_gpu,
    )


def prepare_launch_dataset(
    *,
    cfg: DictConfig,
    env: Env,
    run_id: str,
    dtypes: RuntimeDTypes,
    feature_numa_node: int | None = None,
) -> LaunchDataset:
    prepared = DatasetManager(
        env=env,
        run_id=run_id,
        dataset=cfg.dataset,
        index_dtype=dtypes.index_np,
        time_dtype=dtypes.timestamp_np,
        feature_dtype=dtypes.feature_torch,
        graph_direction=str(cfg.get("graph_direction", "auto")),
        ignore_nfeat_on_disk=bool(cfg.get("ignore_nfeat_on_disk", False)),
        ignore_efeat_on_disk=bool(cfg.get("ignore_efeat_on_disk", False)),
        feature_numa_node=feature_numa_node,
    ).prepare()
    graph, frame, n_nodes, _, ndim, edim, _, _, nfeat, efeat, _ = prepared
    return LaunchDataset(
        graph=graph,
        frame=frame,
        n_nodes=n_nodes,
        ndim=ndim,
        edim=edim,
        nfeat=nfeat,
        efeat=efeat,
    )


def prepare_ring_buffer(
    *,
    env: Env,
    run_id: str,
    dtypes: RuntimeDTypes,
    fanout: list[int],
    root_expansion: RootExpansion,
    train_batch_size: int,
    eval_batch_size: int,
    train_n_neg: int,
    eval_n_neg: int,
    ring_buf_size: int,
    gpu_numa_nodes: list[int | None] | None = None,
) -> RingBuf:
    max_n_neg = max(train_n_neg, eval_n_neg)
    slot_capacity = cal_max_buf_size_for_layouts(
        [(train_batch_size, train_n_neg), (eval_batch_size, eval_n_neg)],
        fanout=fanout,
        root_expansion=root_expansion,
    )
    return RingBuf(
        run_id=run_id,
        env=env,
        ndtype=dtypes.index_torch,
        edtype=dtypes.index_torch,
        tsdtype=dtypes.timestamp_torch,
        fanout=fanout,
        max_batch_size=train_batch_size,
        n_neg=max_n_neg,
        root_expansion=root_expansion,
        max_size=ring_buf_size,
        slot_capacity=slot_capacity,
        gpu_numa_nodes=gpu_numa_nodes,
    )


def prepare_control_plane(
    *, env: Env, run_id: str, ring_buffer: RingBuf, n_nodes: int
) -> SharedControlPlane:
    """Create the tensor-payload control ring shared by host and workers."""
    node_capacity = min(ring_buffer.slot_capacity, n_nodes)
    payload_capacity = control_payload_capacity(
        n_gpus=env.n_gpu,
        node_capacity=node_capacity,
        mask_capacity=ring_buffer.slot_capacity,
    )
    return SharedControlPlane(
        env=env,
        run_id=run_id,
        payload_capacity=payload_capacity,
        max_size=2,
    )


def prepare_dynamic_store(
    *,
    enabled: bool,
    env: Env,
    run_id: str,
    dtypes: RuntimeDTypes,
    n_nodes: int,
    boxsize: int,
    d_mem: int,
    edim: int,
    numa_node: int | None = None,
    gpu_resident: bool = False,
) -> MemBoxStore | None:
    if not enabled:
        return None
    if gpu_resident:
        if env.is_host:
            return None
        device = torch.device(f"cuda:{env.local_rank}")
        d_mail = 2 * d_mem + edim
        return MemBoxStore(
            mem=torch.zeros((n_nodes, d_mem), dtype=dtypes.state_torch, device=device),
            mem_ts=torch.zeros(
                n_nodes, dtype=dtypes.timestamp_torch, device=device
            ),
            box=torch.zeros(
                (n_nodes, boxsize, d_mail),
                dtype=dtypes.state_torch,
                device=device,
            ),
            box_ts=torch.zeros(
                (n_nodes, boxsize),
                dtype=dtypes.timestamp_torch,
                device=device,
            ),
            box_pos=torch.zeros(n_nodes, dtype=torch.int64, device=device),
        )
    return MemBoxStoreManager(
        env=env,
        mdtype=dtypes.state_torch,
        tdtype=dtypes.timestamp_torch,
        run_id=run_id,
        n_nodes=n_nodes,
        boxsize=boxsize,
        d_mem=d_mem,
        edim=edim,
        numa_node=numa_node,
    ).prepare()
