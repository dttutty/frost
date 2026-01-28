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
        logger.debug(f"StaticPlanner: initializing with n_gpus={len(gpus_distance)}")
        # 1. 存储静态配置 (解决痛点 1 和 4)
        self.gpus_distance = gpus_distance
        self.n_gpus = len(self.gpus_distance)

        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.pending: list[set[int]] | None = None
        self.fetchs: list[set[int]] = [set() for _ in range(self.n_gpus)]
        logger.debug("StaticPlanner: initialization complete")

    def run(self) -> None:
        logger.debug("StaticPlanner: starting run")
        try:
            self.logic()
        except Exception:
            logger.error("!!! StaticPlanner Thread Crashed !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)

    def logic(self) -> None:
        batch_count = 0
        logger.debug("StaticPlanner: starting logic loop")

        while not self.stop_ev.is_set():
            logger.debug("StaticPlanner: waiting for need from in_q")
            need = self.in_q.get(block=True)
            logger.debug(f"StaticPlanner: received data type={type(need).__name__}")

            if isinstance(need, list):
                batch_count += 1
                logger.debug(
                    f"StaticPlanner: processing batch #{batch_count}, needs count={len(need)}"
                )

                if self.pending is None:  # first batch
                    logger.debug(
                        f"StaticPlanner: batch #{batch_count} is first batch, initializing pending and fetchs"
                    )
                    self.pending = need
                    self.fetchs = need
                    # Log sizes for each GPU
                    for gpu_id, needs in enumerate(need):
                        logger.debug(
                            f"StaticPlanner: GPU {gpu_id} initial needs size={len(needs)}"
                        )
                    logger.debug(
                        f"StaticPlanner: batch #{batch_count} first batch initialized (no plan output)"
                    )
                    # we still need to wait because we need to know what to cache
                else:  # not first batch
                    logger.debug(
                        f"StaticPlanner: batch #{batch_count} planning for avails->needs transition"
                    )
                    avails = self.pending

                    # Log avails and needs sizes
                    for gpu_id in range(len(avails)):
                        logger.debug(
                            f"StaticPlanner: GPU {gpu_id} avails={len(avails[gpu_id])}, needs={len(need[gpu_id])}"
                        )

                    plan = self._plan(avails, need)

                    logger.debug(
                        f"StaticPlanner: batch #{batch_count} plan created, putting to out_q"
                    )
                    self.out_q.put(plan)
                    self.pending = need
                    logger.debug(f"StaticPlanner: batch #{batch_count} complete")

            elif need == "end_of_epoch":  # None
                logger.debug("StaticPlanner: received end_of_epoch")
                assert self.pending is not None
                avails = self.pending
                # 没有下一个 batch 了，next_needs 为空
                next_needs: list[set[int]] = [set() for _ in range(self.n_gpus)]
                logger.debug(
                    "StaticPlanner: planning final batch with empty next_needs"
                )

                plan = self._plan(avails, next_needs)
                logger.debug("StaticPlanner: putting final plan to out_q")
                self.out_q.put(plan)
                self.pending = None
                logger.debug(
                    "StaticPlanner: end_of_epoch complete, pending reset to None"
                )
            else:
                logger.debug(
                    "StaticPlanner: received None, putting None to out_q and exiting"
                )
                self.out_q.put(None)
                break

        logger.debug(
            f"StaticPlanner: exiting logic loop, processed {batch_count} batches"
        )

    def _plan(
        self,
        avails_of_1: list[set[int]],
        needs_of_2: list[set[int]],
    ) -> list[StPlan]:
        """规划内存（Memory）的P2P、CPU预取和CPU写回。"""
        logger.debug("StaticPlanner: _plan called")

        n_gpus = self.n_gpus
        # fetchs_of_2 need to be filled
        fetchs_of_2: list[set[int]] = [set() for _ in range(n_gpus)]
        plans_of_1 = []

        total_to_cache = 0
        total_to_fetch = 0

        for n in range(n_gpus):
            to_cache_of_1 = avails_of_1[n] & needs_of_2[n]
            plan_of_1 = StPlan(
                to_cache=to_cache_of_1,
                fetch=self.fetchs[n],
            )
            fetchs_of_2[n] = needs_of_2[n] - avails_of_1[n]
            plans_of_1.append(plan_of_1)

            total_to_cache += len(to_cache_of_1)
            total_to_fetch += len(self.fetchs[n])
            logger.debug(
                f"StaticPlanner: GPU {n} plan - to_cache={len(to_cache_of_1)}, fetch={len(self.fetchs[n])}, next_fetch={len(fetchs_of_2[n])}"
            )

        self.fetchs = fetchs_of_2
        logger.debug(
            f"StaticPlanner: _plan complete, total_to_cache={total_to_cache}, total_to_fetch={total_to_fetch}"
        )
        return plans_of_1
