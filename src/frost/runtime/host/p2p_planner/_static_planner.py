import logging
import os
import threading
import traceback
from queue import Queue
from typing import Literal

import numpy as np
import torch

from frost.aliases.host import MetasStNeed
from frost.datatypes.plan import StaticPlan
from frost.runtime._thread_base import ThreadBase
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


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

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        self.fetch: list[TensorSet] = [
            TensorSet(torch.tensor([], dtype=torch.long, device=device))
            for _ in range(self.n_gpus)
        ]

    def run(self) -> None:
        try:
            self.logic()
        except Exception:
            logger.error("!!! StaticPlanner Thread Crashed !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)

    def logic(self) -> None:
        batch_count = 0

        while not self.stop_ev.is_set():
            need = self.in_q.get(block=True)

            if need is None:
                self.out_q.put(None)
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

                plan = self._plan(avails, next_needs)
                self.out_q.put(plan)
                self.pending = None
                continue

            batch_count += 1

            if self.pending is None:  # first batch
                self.pending = need
                self.fetchs = need
                # we still need to wait because we need to know what to cache
            else:  # not first batch
                avails = self.pending

                plan = self._plan(avails, need)

                self.out_q.put(plan)
                self.pending = need

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

        # if __debug__:
        #     # print self.fetchs[0].data.shape[0] and self.fetchs[1].data.shape[0]
        #     logger.debug(f"Fetchs shapes: {[f.fetch.data.shape[0] for f in plans_of_1]}")
        #     if any(f.fetch.data.shape[0] == 0 for f in plans_of_1):
        #         raise ValueError("Warning: Some fetchs are empty!")
        #     logger.debug(f"Cache shapes: {[f.to_cache.data.shape[0] for f in plans_of_1]}")

        self.fetchs = fetchs_of_2
        assert plans_of_1[0] is not None and plans_of_1[1] is not None
        return plans_of_1
