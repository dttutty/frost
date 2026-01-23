import logging
import threading
from queue import Queue
from typing import List, Set, Tuple

import numpy as np

from schemas.datatypes.plan import DyPlan
from thread_base import ThreadBase

from ..meta_batcher import MetaBundle


logger = logging.getLogger(__name__)


def handle_remaining(
    needs: List[Set[int]],
    avails: List[Set[int]],
    gpus_distance: np.ndarray,
) -> Tuple[List[Set[int]], List[List[Set[int]]], np.ndarray]:
    """
    (纯 Python Set 版本)
    计算陈旧数据的 P2P 计划和 CPU 预取。
    逻辑：距离优先的贪心匹配 (Greedy Matching by Distance)。
    Receiver-Centric
    """
    n_gpus = len(gpus_distance)
    p2p_counts = np.zeros((n_gpus, n_gpus), dtype=np.int32)

    # 初始化结果容器
    p2p_plan: List[List[Set[int]]] = [
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
            need: Set[int] = needs[r]

            # 交集：sender 有 且 receiver 缺
            to_transfer = available & need

            if to_transfer:
                # 1. 加入传输计划
                p2p_plan[s][r].update(to_transfer)

                # 2. 关键步骤：贪心消除
                # 既然在这个距离找到了，就不需要再去更远的地方找了 -> 从需求中移除
                need.difference_update(to_transfer)

                # 3. 计数
                p2p_counts[s, r] = len(to_transfer)  # 记录 s 发给 r 的数量

    return needs, p2p_plan, p2p_counts


def handle_update(
    needs: List[Set[int]],
    avails: List[Set[int]],
) -> Tuple[
    List[Set[int]],  # leftovers_on_gpu
    List[List[Set[int]]],  # p2p_transfer_plan
    np.ndarray,
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
    p2p_plan = [[set() for _ in range(n_gpus)] for _ in range(n_gpus)]
    leftovers = []

    # 使用 numpy 数组以保持返回类型兼容性
    p2p_counts = np.zeros((n_gpus, n_gpus), dtype=np.int32)

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
                p2p_counts[s, r] = len(to_transfer)

        # 4. 计算 Leftovers (Sender 持有有，但没人要的)
        # 等价于: fresh_held_mask & ~all_nids_to_send_mask
        leftover = new_update - everything_sender_sends
        leftovers.append(leftover)

    return (
        leftovers,
        p2p_plan,
        p2p_counts,
    )


def dedup_updates(
    updates: List[Set[int]], reverse: bool = True
) -> Tuple[List[Set[int]], Set[int]]:
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
    new_updates: List[Set[int]] = [set() for _ in range(n_gpus)]
    if n_gpus == 0:
        return new_updates, set()
    all_update: Set[int] = set()
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
        in_q: Queue[MetaBundle | None],
        out_q: Queue[Tuple[MetaBundle, List[DyPlan]] | None],
        stop_ev: threading.Event,
    ):
        super().__init__(name="dynamic_planner")
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

    def logic(self):
        while not self.stop_ev.is_set():
            new_infos = self.in_q.get(block=True)
            if new_infos is not None:
                if self.pending_infos is None:
                    current_needs, _ = new_infos
                    self.pending_infos = new_infos
                    self.fetch = current_needs
                else:
                    current_needs, current_updates = self.pending_infos
                    next_needs, _ = new_infos
                    plan = self._plan(current_needs, current_updates, next_needs)

                    self.out_q.put(plan)
                    self.pending_infos = new_infos

            elif new_infos is None:
                assert self.pending_infos is not None
                current_needs, current_updates = self.pending_infos
                # 没有下一个 batch 了，next_needs 为空
                next_needs = [set() for _ in range(self.n_gpus)]

                plan = self._plan(current_needs, current_updates, next_needs)
                self.out_q.put(plan)
                self.pending_infos = None

    def _plan(
        self,
        avails: List[Set[int]],
        update_avails: List[Set[int]],
        needs: List[Set[int]],
    ) -> List[DyPlan]:
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

        fetch = self.fetch
        self.fetch = needs

        for src in range(n_gpus):
            for dst in range(n_gpus):
                intersection = fresh_p2p[src][dst] & stale_p2p[src][dst]
                assert len(intersection) == 0, (
                    "Fresh and Stale P2P sets should be disjoint."
                )
                p2p[src][dst] = fresh_p2p[src][dst] | stale_p2p[src][dst]

            rec_cts = fresh_p2p_cts[:, src] + stale_p2p_cts[:, src]
            send_cts = fresh_p2p_cts[src, :] + stale_p2p_cts[src, :]

            plan = DyPlan(
                p2p=p2p[src],
                fetch=fetch[src],
                writeback=leftovers[src],
                rec_cts=rec_cts,  # 两个长度相等,可以逐位相加
                send_cts=send_cts,  # 两个长度相等,可以逐位相加
            )
            plans.append(plan)
        return plans
