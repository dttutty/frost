import os
import threading
import traceback
from queue import Queue
from typing import List, Set, Tuple

import numpy as np

from schemas import Plan

from ..meta_batcher import MetaBundle


class StaticPlanner(threading.Thread):
    """
    这个是对那种mailbox_update=mem_update的planner
    """

    gpus_distance: np.ndarray
    n_gpus: int

    def __init__(
        self,
        gpus_distance: np.ndarray,
        in_q: Queue[MetaBundle | None],
        out_q: Queue[Tuple[MetaBundle, List[Plan]] | None],
        stop_ev: threading.Event,
    ):
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
        self.pending_infos = None

    def run(self):
        try:
            while not self.stop_ev.is_set():
                self.logic()
        except Exception:
            print("!!! Planner Thread Crashed !!!")
            traceback.print_exc()
            self.stop_ev.set()
            os._exit(1)

    def logic(self):
        new_infos = self.in_q.get(block=True)
        if new_infos is not None:
            if self.pending_infos is None:
                current_needs, _ = new_infos
                self.pending_infos = new_infos
                self.prefetch = current_needs
            else:
                current_needs = self.pending_infos
                next_needs, _ = new_infos
                plan = self._plan(current_needs, next_needs)

                self.out_q.put(plan)
                self.pending_infos = new_infos

        elif new_infos is None:
            assert self.pending_infos is not None
            current_needs = self.pending_infos
            # 没有下一个 batch 了，next_needs 为空
            next_needs = [set() for _ in range(self.n_gpus)]

            plan = self._plan(current_needs, next_needs)
            self.out_q.put(plan)
            self.pending_infos = None

    def _plan(
        self,
        avails: List[Set[int]],
        needs: List[Set[int]],
    ) -> List[Plan]:
        """规划内存（Memory）的P2P、CPU预取和CPU写回。"""

        update_avails, all_update_avails = dedup_updates(
            update_avails,
            reverse=True,
        )

        n_gpus = self.n_gpus

        # === 解决痛点 2：中间结果存入 self ===
        leftovers, fresh_p2p, fresh_p2p_cts = handle_update(needs, update_avails)

        # 1. 预处理：确定每个 GPU 实际持有的 "Stale" 数据
        # 如果提供了 fresh 集合，必须从持有列表中排除它们（因为 fresh 优先用于 fresh plan）
        avails = [avail - all_update_avails for avail in avails]
        needs = [need - all_update_avails for need in needs]

        needs, stale_p2p, stale_p2p_cts = handle_remaining(
            needs,  # <-- 从 self 读取
            avails,
            self.gpus_distance,
        )  # 这种没有新mem的节点,直接发送旧的mem就可以(由distance最近的发)

        plans = []
        p2p = [[set() for _ in range(n_gpus)] for _ in range(n_gpus)]

        prefetch = self.prefetch
        self.prefetch = needs

        for src in range(n_gpus):
            for dst in range(n_gpus):
                intersection = fresh_p2p[src][dst] & stale_p2p[src][dst]
                assert len(intersection) == 0, (
                    "Fresh and Stale P2P sets should be disjoint."
                )
                p2p[src][dst] = fresh_p2p[src][dst] | stale_p2p[src][dst]

            rec_cts = fresh_p2p_cts[:, src] + stale_p2p_cts[:, src]
            send_cts = fresh_p2p_cts[src, :] + stale_p2p_cts[src, :]

            plan = Plan(
                p2p=p2p[src],
                prefetch=prefetch[src],
                writeback=leftovers[src],
                rec_cts=rec_cts,  # 两个长度相等,可以逐位相加
                send_cts=send_cts,  # 两个长度相等,可以逐位相加
            )
            plans.append(plan)
        return plans
