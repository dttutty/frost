import logging
import os
import threading
import traceback
from queue import Queue
from typing import Literal

import numpy as np

from schemas.alias import MetasStNeed
from schemas.datatypes.plan import StPlan
from thread_base import ThreadBase


logger = logging.getLogger(__name__)


class StaticPlanner(ThreadBase):
    """
    这个是对那种mailbox_update=mem_update的planner
    """

    def __init__(
        self,
        gpus_distance: np.ndarray,
        in_q: Queue[MetasStNeed | None | Literal["end_of_epoch"]],
        out_q: Queue[list[StPlan] | None],
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="static_planner", stop_ev=stop_ev)
        """
        初始化规划器。

        Args:
            gpus_distance: GPU 距离矩阵，用于计算距离。
        """
        # 1. 存储静态配置 (解决痛点 1 和 4)
        self.gpus_distance = gpus_distance
        self.n_gpus = len(self.gpus_distance)

        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.pending: list[set[int]] | None = None
        self.fetchs: list[set[int]] = [set() for _ in range(self.n_gpus)]

    def run(self) -> None:
        try:
            self.logic()
        except Exception:
            logger.debug("!!! Planner Thread Crashed !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            need = self.in_q.get(block=True)
            if isinstance(need, list):
                if self.pending is None:  # first batch
                    self.pending = need
                    self.fetchs = need
                    # we still need to wait because we need to know what to cache
                else:  # not first batch
                    avails = self.pending
                    plan = self._plan(avails, need)

                    self.out_q.put(plan)
                    self.pending = need

            elif need == "end_of_epoch":  # None
                assert self.pending is not None
                avails = self.pending
                # 没有下一个 batch 了，next_needs 为空
                next_needs: list[set[int]] = [set() for _ in range(self.n_gpus)]

                plan = self._plan(avails, next_needs)
                self.out_q.put(plan)
                self.pending = None
            else:
                self.out_q.put(None)
                break

    def _plan(
        self,
        avails_of_1: list[set[int]],
        needs_of_2: list[set[int]],
    ) -> list[StPlan]:
        """规划内存（Memory）的P2P、CPU预取和CPU写回。"""

        n_gpus = self.n_gpus
        # fetchs_of_2 need to be filled
        fetchs_of_2: list[set[int]] = [set() for _ in range(n_gpus)]
        plans_of_1 = []

        for n in range(n_gpus):
            to_cache_of_1 = avails_of_1[n] & needs_of_2[n]
            plan_of_1 = StPlan(
                to_cache=to_cache_of_1,
                fetch=self.fetchs[n],
            )
            fetchs_of_2[n] = needs_of_2[n] - avails_of_1[n]
            plans_of_1.append(plan_of_1)
        self.fetchs = fetchs_of_2
        return plans_of_1
