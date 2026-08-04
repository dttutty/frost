import json
import logging
import os
import threading
import time
from queue import Queue
from typing import Literal

import numpy as np
import torch

from frost.runtime.host.types import MetasStNeed
from frost.datatypes.plan import StaticPlan
from frost.runtime._thread_base import ThreadBase, put_shutdown_sentinel
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


def _timing_enabled() -> bool:
    return os.environ.get("FROST_HOST_TIMING_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class StaticPlanner(ThreadBase):
    """
    Planner for static node features (nfeat), where each batch's access set
    is determined solely by which nodes appear in that batch (no P2P needed).
    """

    def __init__(
        self,
        gpus_distance: np.ndarray,
        in_q: Queue[MetasStNeed | None | Literal["end_of_epoch"]],
        out_q: Queue[list[StaticPlan] | None],
        stop_ev: threading.Event,
        cache_horizon: int = 8,
        large_batch_need_rows: int = 40_000,
        very_large_batch_need_rows: int = 120_000,
    ) -> None:
        super().__init__(name="static_planner", stop_ev=stop_ev)
        if cache_horizon < 1:
            raise ValueError("static feature cache horizon must be >= 1")
        if large_batch_need_rows < 1:
            raise ValueError("large-batch feature need threshold must be >= 1")
        if very_large_batch_need_rows <= large_batch_need_rows:
            raise ValueError(
                "very-large feature need threshold must exceed large threshold"
            )
        # Store configuration
        self.gpus_distance = gpus_distance
        self.n_gpus = len(self.gpus_distance)
        self.cache_horizon = cache_horizon
        self.large_batch_need_rows = large_batch_need_rows
        self.very_large_batch_need_rows = very_large_batch_need_rows
        self._active_cache_horizon = cache_horizon

        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.pending: list[MetasStNeed] = []
        self.cached: list[TensorSet] | None = None
        self.fetchs: list[TensorSet] | None = None
        self._timing_enabled = _timing_enabled()
        self._plan_s_curve: list[float] = []
        self._out_q_put_s_curve: list[float] = []
        self._need_counts_curve: list[list[int]] = []
        self._fetch_counts_curve: list[list[int]] = []
        self._cache_hit_counts_curve: list[list[int]] = []
        self._next_cache_counts_curve: list[list[int]] = []

    def _put_plan(self, plan: list[StaticPlan]) -> None:
        start = time.perf_counter()
        self.out_q.put(plan)
        if self._timing_enabled:
            self._out_q_put_s_curve.append(time.perf_counter() - start)

    def logic(self) -> None:
        """Main planning loop for static node features.

        Uses a bounded lookahead: incoming per-GPU need sets are held until the
        planner can retain features that will be reused within ``cache_horizon``
        future batches. This preserves hot rows across a one-batch gap without
        turning the worker cache into an unbounded feature store. Sentinel handling:

        - ``None``: forward ``None`` to the output queue and exit.
        - ``"end_of_epoch"``: flush every pending plan and reset the cache.
        """
        while not self.stop_ev.is_set():
            need = self.in_q.get(block=True)

            if need is None:
                put_shutdown_sentinel(self.out_q, None)
                break

            if need == "end_of_epoch":
                self._flush_pending()
                self.cached = None
                self.fetchs = None
                continue

            if not self.pending:
                self.cached = [TensorSet.empty_like(item.data) for item in need]
                self.fetchs = need
                max_need_rows = max(len(item) for item in need)
                self._active_cache_horizon = (
                    self.cache_horizon
                    if self.large_batch_need_rows
                    <= max_need_rows
                    < self.very_large_batch_need_rows
                    else min(2, self.cache_horizon)
                )
            self.pending.append(need)
            if len(self.pending) > self._active_cache_horizon:
                self._emit_oldest_pending()
        if self._timing_enabled and self._plan_s_curve:
            payload = {
                "system": "frost",
                "plan_kind": "nfeat",
                "cache_horizon": self.cache_horizon,
                "active_cache_horizon": self._active_cache_horizon,
                "batch_count": len(self._plan_s_curve),
                "plan_s_curve": self._plan_s_curve,
                "out_q_put_s_curve": self._out_q_put_s_curve,
                "need_counts_curve": self._need_counts_curve,
                "fetch_counts_curve": self._fetch_counts_curve,
                "cache_hit_counts_curve": self._cache_hit_counts_curve,
                "next_cache_counts_curve": self._next_cache_counts_curve,
            }
            print(
                "APPENDIX_STATIC_PLANNER_TIMING_CURVE "
                f"{json.dumps(payload, sort_keys=True)}"
            )

    def _flush_pending(self) -> None:
        while self.pending:
            self._emit_oldest_pending()

    def _emit_oldest_pending(self) -> None:
        current = self.pending[0]
        future = self.pending[1 : self._active_cache_horizon + 1]
        start = time.perf_counter()
        plan = self._plan_horizon(current, future)
        if self._timing_enabled:
            self._plan_s_curve.append(time.perf_counter() - start)
        self._put_plan(plan)
        self.pending.pop(0)

    def _plan(
        self,
        avails_of_1: list[TensorSet],
        needs_of_2: list[TensorSet],
    ) -> list[StaticPlan]:
        """Compatibility helper for a one-batch-lookahead plan."""
        if self.cached is None:
            self.cached = [TensorSet.empty_like(item.data) for item in avails_of_1]
        if self.fetchs is None:
            self.fetchs = avails_of_1
        return self._plan_horizon(avails_of_1, [needs_of_2])

    def _plan_horizon(
        self,
        current_needs: MetasStNeed,
        future_needs: list[MetasStNeed],
    ) -> list[StaticPlan]:
        """Plan current fetches and retain rows reused within the horizon."""
        assert self.cached is not None
        assert self.fetchs is not None

        n_gpus = self.n_gpus
        next_fetchs: list[TensorSet] = []
        next_cached: list[TensorSet] = []
        plans: list[StaticPlan] = []

        total_to_cache = 0
        total_to_fetch = 0
        need_counts: list[int] = []
        fetch_counts: list[int] = []
        cache_hit_counts: list[int] = []
        next_cache_counts: list[int] = []

        for n in range(n_gpus):
            future_union = (
                TensorSet(
                    torch.cat([future[n].data for future in future_needs]),
                    sort_now=True,
                    ensure_unique=True,
                )
                if future_needs
                else TensorSet.empty_like(current_needs[n].data)
            )

            available = self.cached[n] | current_needs[n]
            to_cache = available & future_union
            plan = StaticPlan(
                to_cache=to_cache,
                fetch=self.fetchs[n],
            )
            if future_needs:
                next_fetch = future_needs[0][n] - to_cache
            else:
                next_fetch = TensorSet.empty_like(current_needs[n].data)
            next_fetchs.append(next_fetch)
            next_cached.append(to_cache)
            plans.append(plan)

            total_to_cache += len(to_cache)
            total_to_fetch += len(self.fetchs[n])
            if self._timing_enabled:
                need_count = len(current_needs[n])
                fetch_count = len(self.fetchs[n])
                need_counts.append(need_count)
                fetch_counts.append(fetch_count)
                cache_hit_counts.append(need_count - fetch_count)
                next_cache_counts.append(len(to_cache))

        if self._timing_enabled:
            self._need_counts_curve.append(need_counts)
            self._fetch_counts_curve.append(fetch_counts)
            self._cache_hit_counts_curve.append(cache_hit_counts)
            self._next_cache_counts_curve.append(next_cache_counts)

        # if __debug__:
        #     # print self.fetchs[0].data.shape[0] and self.fetchs[1].data.shape[0]
        #     logger.debug(f"Fetchs shapes: {[f.fetch.data.shape[0] for f in plans_of_1]}")
        #     if any(f.fetch.data.shape[0] == 0 for f in plans_of_1):
        #         raise ValueError("Warning: Some fetchs are empty!")
        #     logger.debug(f"Cache shapes: {[f.to_cache.data.shape[0] for f in plans_of_1]}")

        self.cached = next_cached
        self.fetchs = next_fetchs
        assert len(plans) == n_gpus
        return plans
