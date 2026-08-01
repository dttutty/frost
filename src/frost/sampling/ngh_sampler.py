"""Temporal neighborhood sampling into the host-to-worker ring buffer."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Literal, cast

import numpy as np
import torch

from frost.models.contracts import RootExpansion
from frost.seq_sampler import SeqSampler, TemporalGraphType
from frost.shm.ring_buf import RingBuf

logger = logging.getLogger(__name__)


def _resolve_root_layout(
    root_nid: torch.Tensor, root_ts: torch.Tensor
) -> tuple[int, int, int]:
    """Return ``(n_src, n_neg, n_root)`` for a root layout."""
    n_src = int(root_ts.numel())
    if n_src <= 0:
        raise ValueError("root_ts must contain at least one positive edge timestamp")
    if root_nid.numel() % n_src != 0:
        raise ValueError(
            f"root_nid length ({root_nid.numel()}) must be divisible by "
            f"root_ts length ({n_src})."
        )
    n_neg = root_nid.numel() // n_src - 2
    if n_neg <= 0:
        raise ValueError(
            "Root layout must contain src, positive dst, and at least one negative; "
            f"got width {root_nid.numel() // n_src}."
        )
    return n_src, n_neg, int(root_nid.numel())


def _acquire_and_write_roots(
    ring_buf: RingBuf,
    gpu_id: int,
    root_nid: torch.Tensor,
    root_ts: torch.Tensor,
    pos_eid: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], int]:
    """Acquire a slot and write the shared root prefix once for every sampler."""
    n_src, n_neg, n_root = _resolve_root_layout(root_nid, root_ts)
    slot = ring_buf.acquire_slot(gpu_id)
    if slot is None:
        raise RuntimeError("Failed to acquire slot from ring buffer")

    slot["NID"][:n_root] = root_nid.cpu()
    ts_cpu = root_ts.cpu().to(dtype=slot["TS"].dtype)
    slot["TS"][:n_src] = ts_cpu
    slot["TS"][n_src : 2 * n_src] = ts_cpu
    slot["TS"][2 * n_src : n_root] = ts_cpu.repeat_interleave(n_neg)
    slot["EID"][:n_src] = pos_eid.cpu()
    slot["EID"][n_src:n_root] = -1
    return slot, n_root


def _commit(
    ring_buf: RingBuf,
    gpu_id: int,
    slot: dict[str, torch.Tensor],
    length: int,
) -> tuple[torch.Tensor, int]:
    # The host materializes compact access sets synchronously before publishing
    # the matching control message.  Until that publication a worker cannot
    # consume or recycle this slot, so a view is sufficient and avoids cloning
    # the complete sampled layout solely for planner input extraction.
    sampled_nid = slot["NID"][:length]
    ring_buf.commit_slot(gpu_id)
    return sampled_nid, length


def _ensure_seq_sampler_stable_mode() -> None:
    """Prefer the single-threaded native sampler while a debugger is attached."""
    if os.getenv("FROST_SEQSAMPLER_NO_OMP") is None and sys.gettrace() is not None:
        os.environ["FROST_SEQSAMPLER_NO_OMP"] = "1"
        logger.warning(
            "Detected active Python debugger; using the single-thread SeqSampler."
        )


def _enable_tgl_sampler_parity_if_requested(enabled: bool) -> None:
    if enabled:
        os.environ["FROST_DEBUG_TGL_BATCH_SHARED_RECENT"] = "1"
        os.environ["FROST_DEBUG_TGL_UNIFORM_LOW_DEGREE_RECENT"] = "1"


def dist_safe_tensor(value, dtype: torch.dtype) -> torch.Tensor:
    """Detach a nanobind-owned array into ordinary torch storage."""
    if value is None:
        return torch.empty(0, dtype=dtype)
    array = np.asarray(value)
    if array.size == 0:
        return torch.empty(0, dtype=dtype)
    return torch.from_numpy(np.array(array, copy=True)).to(dtype)


def _copy_native_array_into(destination: torch.Tensor, value: object) -> None:
    """Copy a live native sampler array directly into a CPU destination."""
    array = np.asarray(value)
    if array.shape != tuple(destination.shape):
        raise ValueError(
            f"Native sampler output shape {array.shape} does not match "
            f"destination shape {tuple(destination.shape)}."
        )
    np.copyto(destination.numpy(), array, casting="same_kind")


class NoSampler:
    """Write roots only for a zero-hop model."""

    fanout: tuple[()] = ()

    def __init__(
        self,
        ring_buf: RingBuf,
        root_expansion: RootExpansion = RootExpansion.ALL_ROOTS,
    ) -> None:
        self.ring_buf = ring_buf
        self.root_expansion = root_expansion

    def sample(
        self,
        root_nid: torch.Tensor,
        root_ts: torch.Tensor,
        sample_nid: torch.Tensor,
        sample_ts: torch.Tensor,
        pos_eid: torch.Tensor,
        gpu_id: int,
    ) -> tuple[torch.Tensor, int]:
        del sample_nid, sample_ts
        slot, n_root = _acquire_and_write_roots(
            self.ring_buf, gpu_id, root_nid, root_ts, pos_eid
        )
        return _commit(self.ring_buf, gpu_id, slot, n_root)


class NeighborhoodSampler:
    """Sample any positive number of historical-neighborhood hops."""

    padding_node_id = -1

    def __init__(
        self,
        graph: TemporalGraphType,
        fanout: list[int],
        ring_buf: RingBuf,
        root_expansion: RootExpansion,
        strategy: Literal["uniform", "recent", "time_interval_aware"],
        seed: int | None,
        time_scaling_factor: float = 0.0,
        tgl_sampler_parity: bool = False,
    ) -> None:
        if not fanout or any(value <= 0 for value in fanout):
            raise ValueError(f"fanout must contain positive hop sizes, got {fanout}.")
        if root_expansion is not RootExpansion.ALL_ROOTS and len(fanout) > 1:
            raise ValueError("Positive-root-only expansion currently supports one hop.")

        _ensure_seq_sampler_stable_mode()
        _enable_tgl_sampler_parity_if_requested(tgl_sampler_parity)
        self.seq_sampler = SeqSampler(
            graph=graph,
            strategy=strategy,
            seed=seed,
            time_scaling_factor=time_scaling_factor,
        )
        self.fanout = tuple(fanout)
        self.ring_buf = ring_buf
        self.root_expansion = root_expansion
        graph_indices = getattr(graph, "indices", None)
        self._node_np_dtype = (
            graph_indices.dtype if graph_indices is not None else np.int64
        )
        graph_ts = getattr(graph, "ts", None)
        self._time_np_dtype = graph_ts.dtype if graph_ts is not None else np.int64
        self._time_torch_dtype = torch.from_numpy(
            np.empty(0, dtype=self._time_np_dtype)
        ).dtype

        omp_disabled = os.getenv("FROST_SEQSAMPLER_NO_OMP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        logger.info(
            "NeighborhoodSampler initialized with %d hop(s), backend=%s, "
            "tgl_sampler_parity=%r",
            len(self.fanout),
            "single-thread" if omp_disabled else "OpenMP",
            tgl_sampler_parity,
        )

    def sample(
        self,
        root_nid: torch.Tensor,
        root_ts: torch.Tensor,
        sample_nid: torch.Tensor,
        sample_ts: torch.Tensor,
        pos_eid: torch.Tensor,
        gpu_id: int,
    ) -> tuple[torch.Tensor, int]:
        if sample_nid.shape[0] != sample_ts.shape[0]:
            raise ValueError(
                "NeighborhoodSampler requires nid/ts to have equal length, "
                f"got nid={sample_nid.shape[0]}, ts={sample_ts.shape[0]}."
            )

        current_nid = np.ascontiguousarray(
            sample_nid.detach().cpu().numpy(), dtype=self._node_np_dtype
        )
        current_ts = np.ascontiguousarray(
            sample_ts.detach().cpu().numpy(), dtype=self._time_np_dtype
        )
        slot, n_root = _acquire_and_write_roots(
            self.ring_buf, gpu_id, root_nid, root_ts, pos_eid
        )

        start = n_root
        end = start
        for n_neighbors in self.fanout:
            parent_count = len(current_nid)
            valid_parent = current_nid != self.padding_node_id
            all_parents_valid = bool(np.all(valid_parent))
            end += parent_count * n_neighbors
            sampled_nid = slot["NID"][start:end].reshape(parent_count, n_neighbors)
            sampled_eid = slot["EID"][start:end].reshape(parent_count, n_neighbors)
            sampled_ts = slot["TS"][start:end].reshape(parent_count, n_neighbors)

            sampled_nid.fill_(self.padding_node_id)
            sampled_eid.fill_(-1)
            sampled_ts.zero_()
            if all_parents_valid or bool(np.any(valid_parent)):
                direct_sampler = getattr(
                    self.seq_sampler, "get_historical_neighbors_into", None
                )
                if all_parents_valid and callable(direct_sampler):
                    direct_sampler(
                        current_nid,
                        current_ts,
                        sampled_nid.numpy(),
                        sampled_eid.numpy(),
                        sampled_ts.numpy(),
                    )
                else:
                    # Filtering preserves the historical per-query RNG indexing
                    # when padded parents appear in a multi-hop layout.
                    query_nid = current_nid
                    query_ts = current_ts
                    if not all_parents_valid:
                        query_nid = current_nid[valid_parent].copy()
                        query_ts = current_ts[valid_parent].copy()
                    out_nid, out_eid, out_ts = (
                        self.seq_sampler.get_historical_neighbors(
                            cast(Any, query_nid),
                            cast(Any, query_ts),
                            n_neighbors,
                        )
                    )
                    if all_parents_valid:
                        _copy_native_array_into(sampled_nid, out_nid)
                        _copy_native_array_into(sampled_eid, out_eid)
                        _copy_native_array_into(sampled_ts, out_ts)
                    else:
                        sampled_nid[valid_parent] = dist_safe_tensor(
                            out_nid, root_nid.dtype
                        ).reshape(-1, n_neighbors)
                        sampled_eid[valid_parent] = dist_safe_tensor(
                            out_eid, pos_eid.dtype
                        ).reshape(-1, n_neighbors)
                        sampled_ts[valid_parent] = dist_safe_tensor(
                            out_ts, self._time_torch_dtype
                        ).reshape(-1, n_neighbors)

            current_nid = slot["NID"][start:end].numpy().reshape(-1)
            current_ts = slot["TS"][start:end].numpy().reshape(-1)
            start = end

        return _commit(self.ring_buf, gpu_id, slot, end)
