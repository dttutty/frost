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
    ) -> None:
        super().__init__(name="static_planner", stop_ev=stop_ev)
        # Store configuration
        self.gpus_distance = gpus_distance
        self.n_gpus = len(self.gpus_distance)

        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.pending: list[TensorSet] | None = None
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

        Uses a one-batch lookahead: each incoming per-GPU nfeat need set is held as
        ``pending`` until the next batch arrives so that the planner can decide which
        features from the current batch to keep in cache (``to_cache``) versus which
        new features the next batch must fetch from CPU (``fetch``).  Sentinel handling:

        - ``None``: forward ``None`` to the output queue and exit.
        - ``"end_of_epoch"``: plan the pending batch with empty next-batch need sets.
        """
        batch_count = 0

        while not self.stop_ev.is_set():
            need = self.in_q.get(block=True)

            if need is None:
                put_shutdown_sentinel(self.out_q, None)
                break

            if need == "end_of_epoch":  # None
                assert self.pending is not None
                avails = self.pending
                # Last batch of the epoch: next_needs is empty
                next_needs: list[TensorSet] = [
                    TensorSet(
                        torch.tensor([], dtype=torch.long, device=torch.device("cpu"))
                    )
                    for _ in range(self.n_gpus)
                ]

                start = time.perf_counter()
                plan = self._plan(avails, next_needs)
                if self._timing_enabled:
                    self._plan_s_curve.append(time.perf_counter() - start)
                self._put_plan(plan)
                self.pending = None
                continue

            batch_count += 1

            if self.pending is None:  # first batch
                self.pending = need
                self.fetchs = need
                # we still need to wait because we need to know what to cache
            else:  # not first batch
                avails = self.pending

                start = time.perf_counter()
                plan = self._plan(avails, need)
                if self._timing_enabled:
                    self._plan_s_curve.append(time.perf_counter() - start)

                self._put_plan(plan)
                self.pending = need
        if self._timing_enabled and self._plan_s_curve:
            payload = {
                "system": "frost",
                "plan_kind": "nfeat",
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

    def _plan(
        self,
        avails_of_1: list[TensorSet],
        needs_of_2: list[TensorSet],
    ) -> list[StaticPlan]:
        """Compute per-GPU StaticPlan: what to cache from batch 1 and what to fetch for batch 2."""

        n_gpus = self.n_gpus
        device = torch.device("cpu")

        fetchs_of_2: list[TensorSet] = [
            TensorSet(torch.tensor([], dtype=torch.long, device=device))
            for _ in range(n_gpus)
        ]
        plans_of_1 = []

        total_to_cache = 0
        total_to_fetch = 0
        need_counts: list[int] = []
        fetch_counts: list[int] = []
        cache_hit_counts: list[int] = []
        next_cache_counts: list[int] = []

        for n in range(n_gpus):
            to_cache_of_1 = avails_of_1[n] & needs_of_2[n]
            plan_of_1 = StaticPlan(
                to_cache=to_cache_of_1,
                fetch=self.fetchs[n],
            )
            fetchs_of_2[n] = needs_of_2[n] - avails_of_1[n]
            plans_of_1.append(plan_of_1)

            total_to_cache += len(to_cache_of_1)
            total_to_fetch += len(self.fetchs[n])
            if self._timing_enabled:
                need_count = len(avails_of_1[n])
                fetch_count = len(self.fetchs[n])
                need_counts.append(need_count)
                fetch_counts.append(fetch_count)
                cache_hit_counts.append(need_count - fetch_count)
                next_cache_counts.append(len(to_cache_of_1))

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

        self.fetchs = fetchs_of_2
        assert len(plans_of_1) == n_gpus
        return plans_of_1
