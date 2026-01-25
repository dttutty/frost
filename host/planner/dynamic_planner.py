import logging
import threading
from queue import Queue
from typing import Literal

import numpy as np

from schemas.alias import MetasDyNeed
from schemas.datatypes.plan import DyPlan
from thread_base import ThreadBase


logger = logging.getLogger(__name__)


def handle_remaining(
    needs: list[set[int]],
    avails: list[set[int]],
    gpus_distance: np.ndarray,
) -> tuple[list[set[int]], list[list[set[int]]]]:
    """
    (纯 Python Set 版本)
    计算陈旧数据的 P2P 计划和 CPU 预取。
    逻辑：距离优先的贪心匹配 (Greedy Matching by Distance)。
    Receiver-Centric
    """
    n_gpus = len(gpus_distance)

    # 初始化结果容器
    p2p_plan: list[list[set[int]]] = [
        [set() for _ in range(n_gpus)] for _ in range(n_gpus)
    ]

    # 3. 核心逻辑：按距离排序的贪心匹配
    # 我们希望优先从距离为 0 (自己), 1 (NVLink), 2 (PCIe/QPI) ... 的地方获取
    unique_distances = sorted(np.unique(gpus_distance))

    for dist in unique_distances:
        # 找到所有距离为 dist 的 (sender, receiver) 对
        # np.argwhere 对于 8x8 这种小矩阵非常快
        pairs = np.argwhere(gpus_distance == dist)

        for s, r in pairs:
            # 获取 Sender 拥有的
            available = avails[s]
            # 获取 Receiver 目前还缺少的 (注意：这里是动态更新的 remaining_needs)
            need: set[int] = needs[r]

            # 交集：sender 有 且 receiver 缺
            to_transfer = available & need

            if to_transfer:
                # 1. 加入传输计划
                p2p_plan[s][r].update(to_transfer)

                # 2. 关键步骤：贪心消除
                # 既然在这个距离找到了，就不需要再去更远的地方找了 -> 从需求中移除
                need.difference_update(to_transfer)

                # 3. 计数

    return needs, p2p_plan


def handle_update(
    needs: list[set[int]],
    avails: list[set[int]],
) -> tuple[
    list[set[int]],  # leftovers_on_gpu
    list[list[set[int]]],  # p2p_transfer_plan
]:
    """
    (纯 Python Set 版本 - 逻辑简化)
    计算 P2P 传输计划：谁有(fresh)，谁需要(needed)，谁发给谁。
    Sender-Centric
    """
    n_gpus = len(needs)

    # 1. 全局 Fresh 集合: 所有 GPU 目前持有的节点并集
    # 等价于: np.any(fresh_held_mask, axis=0)

    # 2. 计算 Stale Needs (缺少的节点)
    # 逻辑: 我需要的 - 全局任何人有fresh的 = 我必须去Stale存储distance优先拿的 (Stale) +需要从CPU预取的
    # 等价于: needed_mask & ~all_fresh_nids_mask

    # 初始化结果容器
    p2p_plan: list[list[set[int]]] = [
        [set() for _ in range(n_gpus)] for _ in range(n_gpus)
    ]
    leftovers = []

    # 使用 numpy 数组以保持返回类型兼容性

    # 3. 计算 P2P 计划 (双层循环: Sender -> Receiver)
    # 逻辑: Sender 有的 & Receiver 需要的 = 传输内容
    # 对于 GPU 数量 (通常 < 16) 来说，O(N^2) 的循环开销甚至小于 Numpy 的启动开销
    for s in range(n_gpus):
        new_update = avails[s]

        # 记录 Sender 发送出去的所有东西，用于计算 leftover
        everything_sender_sends = set()

        for r in range(n_gpus):
            need = needs[r]

            # 交集: S 有且 R 需要
            # 等价于: fresh_send_mask & fresh_need_mask
            to_transfer = new_update & need

            if to_transfer:
                p2p_plan[s][r].update(to_transfer)
                everything_sender_sends.update(to_transfer)

                # 更新计数

        # 4. 计算 Leftovers (Sender 持有有，但没人要的)
        # 等价于: fresh_held_mask & ~all_nids_to_send_mask
        leftover = new_update - everything_sender_sends
        leftovers.append(leftover)

    return (
        leftovers,
        p2p_plan,
    )


def dedup_updates(
    updates: list[set[int]], reverse: bool = True
) -> tuple[list[set[int]], set[int]]:
    """
    (简化逻辑版)
    对每个 GPU 持有的节点列表进行去重，确保一个节点只由一个 GPU 负责。
    优先保留迭代顺序中靠前的 GPU 中的节点。

    Args:
        updates: List[Set[int]], 列表索引对应 rank/gpu_id
        reverse:
            True  -> 倒序优先 (Rank N-1 优先保留节点, Rank 0 只能捡剩下的)
            False -> 正序优先 (Rank 0 优先保留节点)
    """
    n_gpus = len(updates)
    new_updates: list[set[int]] = [set() for _ in range(n_gpus)]
    if n_gpus == 0:
        return new_updates, set()
    all_update: set[int] = set()
    # seen_nodes is the set of nodes that have already been assigned to higher-priority GPUs
    iterator = range(n_gpus - 1, -1, -1) if reverse else range(n_gpus)
    for i in iterator:
        curr_rank_update_nodes = updates[i]
        new_updates[i] = curr_rank_update_nodes - all_update
        all_update.update(curr_rank_update_nodes)

    return new_updates, all_update


class DynamicPlanner(ThreadBase):
    """
    这个是对那种mailbox_update=mem_update的planner
    """

    gpus_distance: np.ndarray
    n_gpus: int

    def __init__(
        self,
        gpus_distance: np.ndarray,
        in_q: Queue[MetasDyNeed | None | Literal["end_of_epoch"]],
        out_q: Queue[list[DyPlan] | None],
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="dynamic_planner", stop_ev=stop_ev)
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
        self.pending: tuple[list[set[int]], list[set[int]]] | None = None

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            in_q_data = self.in_q.get(block=True)
            if isinstance(in_q_data, tuple):
                if self.pending is None:
                    needs, _ = in_q_data
                    self.pending = in_q_data
                    self.fetchs_of_1 = needs
                else:
                    needs, updates = self.pending
                    next_needs, _ = in_q_data
                    plan = self._plan(needs, updates, next_needs)
                    self.out_q.put(plan)
                    self.pending = in_q_data
            elif in_q_data == "end_of_epoch":
                assert self.pending is not None
                needs, updates = self.pending
                # 没有下一个 batch 了，next_needs 为空
                next_needs = [set() for _ in range(self.n_gpus)]
                plan = self._plan(needs, updates, next_needs)
                # this time will generate a plan, but next time will not generate plan
                self.out_q.put(plan)
                self.pending = None
            else:
                self.out_q.put(None)
                break

    def _plan(
        self,
        avails_of_1: list[set[int]],
        update_avails_of_1: list[set[int]],
        needs_of_2: list[set[int]],
    ) -> list[DyPlan]:
        """规划内存（Memory）的P2P、CPU预取和CPU写回。"""

        update_avails_of_1, all_update_avails_of_1 = dedup_updates(
            update_avails_of_1, reverse=True
        )

        n_gpus = self.n_gpus

        # === 解决痛点 2：中间结果存入 self ===
        leftovers_of_1, fresh_p2p_1to2 = handle_update(needs_of_2, update_avails_of_1)

        # 1. 预处理：确定每个 GPU 实际持有的 "Stale" 数据
        # 如果提供了 fresh 集合，必须从持有列表中排除它们（因为 fresh 优先用于 fresh plan）
        avails_of_1 = [avail - all_update_avails_of_1 for avail in avails_of_1]
        needs_of_2 = [need - all_update_avails_of_1 for need in needs_of_2]

        needs_of_2, stale_p2p_1to2 = handle_remaining(
            needs_of_2,  # <-- 从 self 读取
            avails_of_1,
            self.gpus_distance,
        )  # 这种没有新mem的节点,直接发送旧的mem就可以(由distance最近的发)

        plans = []
        p2p: list[list[set[int]]] = [
            [set() for _ in range(n_gpus)] for _ in range(n_gpus)
        ]
        p2p_cts = np.zeros((n_gpus, n_gpus), dtype=np.int32)

        fetchs_of_1 = self.fetchs_of_1
        self.fetchs_of_1 = needs_of_2

        for src in range(n_gpus):
            for dst in range(n_gpus):
                if __debug__:
                    intersection = fresh_p2p_1to2[src][dst] & stale_p2p_1to2[src][dst]
                    assert len(intersection) == 0, (
                        "Fresh and Stale P2P sets should be disjoint."
                    )
                p2p[src][dst] = fresh_p2p_1to2[src][dst] | stale_p2p_1to2[src][dst]
                p2p_cts[src, dst] = len(p2p[src][dst])

        for src in range(n_gpus):
            plan = DyPlan(
                p2p=p2p[src],
                fetch=fetchs_of_1[src],
                writeback=leftovers_of_1[src],
                rec_cts=p2p_cts[:, src],  # 两个长度相等,可以逐位相加
                send_cts=p2p_cts[src, :],  # 两个长度相等,可以逐位相加
            )
            plans.append(plan)
        return plans
