import logging
import os
import threading
import traceback
from queue import Queue
from typing import List, Set

import numpy as np

from schemas.datatypes.plan import StPlan
from thread_base import ThreadBase


logger = logging.getLogger(__name__)


class StaticPlanner(ThreadBase):
    """
    这个是对那种mailbox_update=mem_update的planner
    """

    n_gpus: int
    pending_needs: List[Set[int] | None] | None

    def __init__(
        self,
        gpus_distance: np.ndarray,
        in_q: Queue[List[Set[int] | None] | None],
        out_q: Queue[StPlan | None] | None,
        stop_ev: threading.Event,
    ):
        super().__init__(name="static_planner")
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
        self.pending_needs = None

    def run(self):
        try:
            self.logic()
        except Exception:
            print("!!! Planner Thread Crashed !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)

    def logic(self):
        while not self.stop_ev.is_set():
            needs = self.in_q.get(block=True)
            if needs is not None:
                if self.pending_needs is None:
                    self.pending_needs = needs
                    self.fetchs = needs
                else:
                    avails = self.pending_needs
                    plan = self._plan(avails, needs)

                    self.out_q.put(plan)
                    self.pending_needs = needs

            elif needs is None:
                assert self.pending_needs is not None
                avails = self.pending_needs
                # 没有下一个 batch 了，next_needs 为空
                next_needs = [set() for _ in range(self.n_gpus)]

                plan = self._plan(avails, next_needs)
                self.out_q.put(plan)
                self.pending_needs = None

    def _plan(
        self,
        avails: List[Set[int]],
        needs: List[Set[int]],
    ) -> List[StPlan]:
        """规划内存（Memory）的P2P、CPU预取和CPU写回。"""

        n_gpus = self.n_gpus
        next_fetchs = [set() for _ in range(n_gpus)]
        plans = []

        for n in range(n_gpus):
            to_cache = avails[n] & needs[n]
            plan = StPlan(
                to_cache=to_cache,
                fetch=self.fetchs[n],
            )
            next_fetchs[n] = needs[n] - avails[n]
            plans.append(plan)
        self.fetchs = next_fetchs
        return plans
