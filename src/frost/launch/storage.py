from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from omegaconf import DictConfig

from frost.datatypes.env import Env
from frost.seq_sampler import TemporalGraphType
from frost.shm.dataset_mgr import DatasetManager
from frost.shm.mem_mgr import MemBoxStore, MemBoxStoreManager
from frost.shm.ring_buf import RingBuf, cal_max_buf_size_for_layouts
from frost.utils.precision import PrecisionSpec


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
    *, cfg: DictConfig, env: Env, run_id: str, precision: PrecisionSpec
) -> LaunchDataset:
    prepared = DatasetManager(
        env=env,
        run_id=run_id,
        dataset=cfg.dataset,
        index_dtype=precision.index_np,
        time_dtype=precision.time_preprocessing_np,
        feature_dtype=precision.feature_torch,
        ignore_nfeat_on_disk=bool(cfg.get("ignore_nfeat_on_disk", False)),
        ignore_efeat_on_disk=bool(cfg.get("ignore_efeat_on_disk", False)),
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
    precision: PrecisionSpec,
    fanout: list[int],
    sample_4_neg: bool,
    train_batch_size: int,
    eval_batch_size: int,
    train_n_neg: int,
    eval_n_neg: int,
    ring_buf_size: int,
) -> RingBuf:
    max_n_neg = max(train_n_neg, eval_n_neg)
    slot_capacity = cal_max_buf_size_for_layouts(
        [(train_batch_size, train_n_neg), (eval_batch_size, eval_n_neg)],
        fanout=fanout,
        sample_4_neg=sample_4_neg,
    )
    return RingBuf(
        run_id=run_id,
        env=env,
        ndtype=precision.index_torch,
        edtype=precision.index_torch,
        tsdtype=precision.time_preprocessing_torch,
        fanout=fanout,
        max_batch_size=train_batch_size,
        n_neg=max_n_neg,
        sample_4_neg=sample_4_neg,
        max_size=ring_buf_size,
        slot_capacity=slot_capacity,
    )


def prepare_dynamic_store(
    *,
    enabled: bool,
    env: Env,
    run_id: str,
    precision: PrecisionSpec,
    n_nodes: int,
    boxsize: int,
    d_mem: int,
    edim: int,
) -> MemBoxStore | None:
    if not enabled:
        return None
    return MemBoxStoreManager(
        env=env,
        mdtype=precision.memory_torch,
        tdtype=precision.time_preprocessing_torch,
        run_id=run_id,
        n_nodes=n_nodes,
        boxsize=boxsize,
        d_mem=d_mem,
        edim=edim,
    ).prepare()
