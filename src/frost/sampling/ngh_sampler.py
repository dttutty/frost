from __future__ import annotations

import logging
import os
import sys
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import torch

from frost.seq_sampler import SeqSampler, TemporalGraphType
from frost.shm.ring_buf import RingBuf
from frost.utils.tensor_set import TensorSet

Int64Array = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]
logger = logging.getLogger(__name__)


def _ensure_seq_sampler_stable_mode() -> None:
    """Prefer the single-threaded sampler path in debugger sessions.

    We've observed sporadic native crashes when the OpenMP path is used under
    debug adapters (debugpy / pydevd). Falling back to the single-threaded C++
    implementation avoids that class of failures while preserving correctness.
    Users can still opt back in explicitly by setting
    ``FROST_SEQSAMPLER_NO_OMP=0``.
    """
    no_omp = os.getenv("FROST_SEQSAMPLER_NO_OMP")
    if no_omp is not None:
        return

    debugger_attached = sys.gettrace() is not None
    if debugger_attached:
        os.environ["FROST_SEQSAMPLER_NO_OMP"] = "1"
        logger.warning(
            "Detected active Python debugger; defaulting SeqSampler to "
            "single-thread mode (FROST_SEQSAMPLER_NO_OMP=1) for stability."
        )


class NoSampler:
    def __init__(
        self,
        graph: TemporalGraphType,
        ring_buf: RingBuf,
        n_neg: int = 1,  # from hydra config
        fanout: list[int] | None = None,
        sample_4_neg: bool = False,
    ):
        self.ring_buf = ring_buf
        self.n_neg = n_neg
        self.fanout = None  # NoSampler is always 0-hop; ignore any passed fanout
        self.sample_4_neg = sample_4_neg

    def sample(
        self,
        nid: torch.Tensor,
        ts: torch.Tensor,
        pos_eid: torch.Tensor,
        gpu_id: int,
    ) -> tuple[TensorSet, int, torch.Tensor]:
        n_src = len(nid) // (2 + self.n_neg)
        num_seeds = n_src * (2 + self.n_neg)

        # Wait for an empty slot
        slot = self.ring_buf.acquire_slot(gpu_id)
        if slot is None:
            raise RuntimeError("Failed to acquire slot from ring buffer")

        # Write seed nodes safely
        slot["NID"][:num_seeds] = nid.cpu()
        ts_cpu = ts.cpu()
        slot["TS"][:n_src] = ts_cpu
        slot["TS"][n_src : 2 * n_src] = ts_cpu
        slot["TS"][2 * n_src : num_seeds] = ts_cpu.repeat(self.n_neg)
        slot["EID"][:n_src] = pos_eid.cpu()
        slot["EID"][n_src:num_seeds] = -1

        length = num_seeds

        # Slice and filter before committing to avoid race conditions
        valid_nid_raw = slot["NID"][:length].clone()
        valid_nid = valid_nid_raw[valid_nid_raw != -1]

        # Commit slot only after host is done
        self.ring_buf.commit_slot(gpu_id)

        return TensorSet(valid_nid), length, valid_nid_raw


def dist_safe_tensor(arr, dtype):
    """
    通过强行副本转换，彻底移除 nanobind 的 PyCapsule 残留
    """
    if arr is None:
        return torch.empty(0, dtype=dtype)

    # nanobind may return an empty ndarray proxy for zero-sized outputs.
    # Materialize to numpy first and short-circuit to avoid touching invalid
    # capsule-backed memory in edge cases.
    np_arr = np.asarray(arr)
    if np_arr.size == 0:
        return torch.empty(0, dtype=dtype)

    # 这一步非常关键：np.array(..., copy=True) 会在 Python 堆上创建全新内存
    np_safe = np.array(np_arr, copy=True)

    # 转回 torch 后再 clone 并 contiguous 确保存储结构最简
    return torch.from_numpy(np_safe).to(dtype)


class NghSampler:
    def __init__(
        self,
        graph: TemporalGraphType,
        fanout: list[int],  # from hydra config
        ring_buf: RingBuf,
        sample_4_neg: bool,  # from hydra config
        n_neg: int,  # from hydra config
        strategy: Literal["uniform", "recent"],  # from hydra config
        seed: int | None,  # from hydra config
    ):
        _ensure_seq_sampler_stable_mode()
        assert len(fanout) == 1, (
            "Only single-hop sampling is supported in NghSampler. Use MultiHopNghSampler for multi-hop sampling."
        )
        self.graph = graph
        self.fanout = fanout
        self.seq_sampler = SeqSampler(graph=graph, strategy=strategy, seed=seed)
        self.n_ngh = fanout[0]
        self.ring_buf = ring_buf
        self.sample_4_neg = sample_4_neg
        self.n_neg = n_neg
        # Keep sampler input dtype aligned with temporal-graph indices to avoid
        # unsafe casts in the nanobind/C++ boundary.
        graph_indices = getattr(graph, "indices", None)
        self._node_np_dtype = (
            graph_indices.dtype if graph_indices is not None else np.int64
        )
        omp_disabled = os.getenv("FROST_SEQSAMPLER_NO_OMP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        logger.info(
            "NghSampler initialized with SeqSampler backend path: %s (FROST_SEQSAMPLER_NO_OMP=%r)",
            "single-thread" if omp_disabled else "OpenMP",
            os.getenv("FROST_SEQSAMPLER_NO_OMP"),
        )

    def sample(
        self,
        nid: torch.Tensor,
        ts: torch.Tensor,
        pos_eid: torch.Tensor,
        gpu_id: int,
    ) -> tuple[TensorSet, int, torch.Tensor]:
        n_src = nid.shape[0] // (2 + self.n_neg)
        num_seeds = n_src * (2 + self.n_neg)

        if self.sample_4_neg:
            # roots = [src, dst, neg...], while ts only has one timestamp per source edge.
            # Repeat timestamps so C++ sampler receives arrays with identical lengths.
            nid_4_sample = nid
            ts_4_sample = ts.repeat(2 + self.n_neg)
        else:
            # Only sample neighborhoods for [src, dst].
            nid_4_sample = nid[0 : n_src * 2]
            ts_4_sample = ts.repeat(2)

        if nid_4_sample.shape[0] != ts_4_sample.shape[0]:
            raise ValueError(
                "NghSampler requires nid/ts to have equal length before calling C++ sampler, "
                f"got nid={nid_4_sample.shape[0]}, ts={ts_4_sample.shape[0]}"
            )

        # Materialize contiguous numpy buffers before crossing into nanobind.
        # This avoids passing non-contiguous views / unexpected dtypes that can
        # trigger undefined behavior in C++ sampler code.
        nid_np_safe = np.ascontiguousarray(
            nid_4_sample.detach().cpu().numpy(), dtype=self._node_np_dtype
        )
        ts_np_safe = np.ascontiguousarray(
            ts_4_sample.detach().cpu().numpy(), dtype=np.float32
        )

        # Wait for an empty slot
        slot = self.ring_buf.acquire_slot(gpu_id)
        if slot is None:
            raise RuntimeError("Failed to acquire slot from ring buffer")

        # Write seed nodes safely
        slot["NID"][:num_seeds] = nid.cpu()
        ts_cpu = ts.cpu()
        slot["TS"][:n_src] = ts_cpu
        slot["TS"][n_src : 2 * n_src] = ts_cpu
        slot["TS"][2 * n_src : num_seeds] = ts_cpu.repeat(self.n_neg)
        slot["EID"][:n_src] = pos_eid.cpu()
        slot["EID"][n_src:num_seeds] = -1

        if nid_np_safe.size == 0 or self.n_ngh <= 0:
            n_tensor = torch.empty(0, dtype=torch.int32)
            e_tensor = torch.empty(0, dtype=torch.int32)
            t_tensor = torch.empty(0, dtype=torch.float32)
        else:
            # Get neighbors from C++ nanobind backend
            ngh_nid, ngh_eid, ngh_ts = self.seq_sampler.get_historical_neighbors(
                nid_np_safe, ts_np_safe, self.n_ngh
            )

            # 在 NghSampler.sample 内部使用：
            n_tensor = dist_safe_tensor(ngh_nid, torch.int32).reshape(-1)
            e_tensor = dist_safe_tensor(ngh_eid, torch.int32).reshape(-1)
            t_tensor = dist_safe_tensor(ngh_ts, torch.float32).reshape(-1)

        num_out = n_tensor.numel()

        # Write sampled features to shared memory
        slot["NID"][num_seeds : num_seeds + num_out] = n_tensor
        slot["TS"][num_seeds : num_seeds + num_out] = t_tensor
        slot["EID"][num_seeds : num_seeds + num_out] = e_tensor

        length = num_seeds + num_out

        # Slice and filter before committing
        valid_nid_raw = slot["NID"][:length].clone()
        valid_nid = valid_nid_raw[valid_nid_raw != -1]

        # Commit slot only after host is done
        self.ring_buf.commit_slot(gpu_id)

        return TensorSet(valid_nid), length, valid_nid_raw


class MultiHopNghSampler:
    """
    Not support memory based sampling
    """

    padding_node_id: int

    def __init__(
        self,
        graph: TemporalGraphType,
        fanout: list[int],  # from hydra config
        ring_buf: RingBuf,
        n_neg: int,  # from hydra config
        strategy: Literal["uniform", "recent"],  # from hydra config
        seed: int | None,  # from hydra config
    ):
        _ensure_seq_sampler_stable_mode()
        self.sampler = SeqSampler(graph=graph, strategy=strategy, seed=seed)
        self.fanout = fanout
        self.ring_buf = ring_buf
        self.n_neg = n_neg
        self.padding_node_id = -1

    def sample(
        self,
        nid: torch.Tensor,
        ts: torch.Tensor,
        pos_eid: torch.Tensor,
        gpu_id: int,
    ) -> tuple[TensorSet, int, torch.Tensor]:
        """Sample multi-hop historical neighbors by repeatedly calling ``SeqSampler``."""
        padding_node_id = self.padding_node_id
        fanout = self.fanout
        n_src = len(nid) // (2 + self.n_neg)
        num_seeds = n_src * (2 + self.n_neg)

        nid_np = cast(np.ndarray, nid.cpu().numpy())
        ts_np = cast(np.ndarray, ts.cpu().numpy())
        assert nid_np.shape[0] == ts_np.shape[0], "nid and ts must have the same length"

        if len(fanout) == 0:
            raise ValueError("n_ngh_per_hop must contain at least one hop")

        for hop_idx, n_ngh in enumerate(fanout):
            if n_ngh <= 0:
                raise ValueError(
                    f"n_ngh_per_hop[{hop_idx}] must be positive, got {n_ngh}"
                )

        # Wait for an empty slot
        slot = self.ring_buf.acquire_slot(gpu_id)
        if slot is None:
            raise RuntimeError("Failed to acquire slot from ring buffer")

        # Write seed nodes safely
        slot["NID"][:num_seeds] = nid.cpu()
        ts_cpu = ts.cpu()
        slot["TS"][:n_src] = ts_cpu
        slot["TS"][n_src : 2 * n_src] = ts_cpu
        slot["TS"][2 * n_src : num_seeds] = ts_cpu.repeat(self.n_neg)
        slot["EID"][:n_src] = pos_eid.cpu()
        slot["EID"][n_src:num_seeds] = -1

        cur_nid = nid_np
        cur_ts = ts_np

        start = num_seeds
        end = start
        for n_ngh in fanout:
            hop_size = len(cur_nid)
            valid_mask = cur_nid != padding_node_id
            end += len(cur_nid) * n_ngh

            sampled_nodes = slot["NID"][start:end].reshape(hop_size, n_ngh)
            sampled_edges = slot["EID"][start:end].reshape(hop_size, n_ngh)
            sampled_times = slot["TS"][start:end].reshape(hop_size, n_ngh)

            if np.any(valid_mask):
                valid_nodes = cur_nid[valid_mask]
                valid_times = cur_ts[valid_mask]
                valid_nodes_safe = valid_nodes.copy()
                valid_times_safe = valid_times.copy()

                out_nodes, out_edges, out_times = self.sampler.get_historical_neighbors(
                    valid_nodes_safe,
                    valid_times_safe,
                    n_ngh=n_ngh,
                )

                # Use defensive conversion to avoid DLPack / capsule lifetime issues
                # from nanobind-managed arrays.
                out_nodes_tensor = dist_safe_tensor(out_nodes, torch.int32).reshape(
                    -1, n_ngh
                )
                out_edges_tensor = dist_safe_tensor(out_edges, torch.int32).reshape(
                    -1, n_ngh
                )
                out_times_tensor = dist_safe_tensor(out_times, torch.float32).reshape(
                    -1, n_ngh
                )

                sampled_nodes[valid_mask] = out_nodes_tensor
                sampled_edges[valid_mask] = out_edges_tensor
                sampled_times[valid_mask] = out_times_tensor

            cur_nid = slot["NID"][start:end].numpy().reshape(-1)
            cur_ts = slot["TS"][start:end].numpy().reshape(-1)
            start = end

        length = end

        # Slice and filter before committing
        valid_nid_raw = slot["NID"][:length].clone()
        valid_nid = valid_nid_raw[valid_nid_raw != -1]

        # Commit slot only after host is done
        self.ring_buf.commit_slot(gpu_id)

        return TensorSet(valid_nid), length, valid_nid_raw
