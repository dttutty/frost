import logging
import threading

import numpy as np
import torch

from frost.aliases.host import DynamicPlanInQ, DynamicPlanOutQ, MetasDyNeed
from frost.datatypes.plan import DynamicPlan
from frost.runtime._thread_base import ThreadBase
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


def handle_remaining(
    needs: list[TensorSet],
    avails: list[TensorSet],
    gpus_distance: np.ndarray,
) -> tuple[list[TensorSet], list[list[TensorSet]]]:
    """
    (TensorSet 版本)
    计算陈旧数据的 P2P 计划和 CPU 预取。
    逻辑：距离优先的贪心匹配 (Greedy Matching by Distance)。
    Receiver-Centric
    """
    n_gpus = len(gpus_distance)

    # --- 0. 准备空 TensorSet ---
    # 为了保证 PyTorch 运算不报错，需获取正确的 device 和 dtype
    ref_ts = None
    if n_gpus > 0:
        if len(needs[0]) > 0:
            ref_ts = needs[0]
        elif len(avails[0]) > 0:
            ref_ts = avails[0]

    if ref_ts is not None:
        empty_tensor = torch.tensor(
            [], dtype=ref_ts.data.dtype, device=ref_ts.data.device
        )
    else:
        empty_tensor = torch.tensor([], dtype=torch.long)  # Fallback

    empty_ts = TensorSet(empty_tensor)

    # --- 1. 初始化结果容器 ---
    # p2p_plan[s][r] 表示 s 发给 r 的数据
    p2p_plan: list[list[TensorSet]] = [
        [empty_ts for _ in range(n_gpus)] for _ in range(n_gpus)
    ]

    # --- 2. 核心逻辑：按距离排序的贪心匹配 ---
    unique_distances = sorted(np.unique(gpus_distance))

    for dist in unique_distances:
        # 找到所有距离为 dist 的 (sender, receiver) 对
        # results in array of shape (N, 2) -> [[s1, r1], [s2, r2], ...]
        pairs = np.argwhere(gpus_distance == dist)

        for s, r in pairs:
            # 优化：如果 Receiver 已经没什么需要的了，直接跳过
            if len(needs[r]) == 0:
                continue

            # 优化：如果 Sender 没什么可提供的，直接跳过
            if len(avails[s]) == 0:
                continue

            # 获取 Sender 拥有的
            available = avails[s]

            # 获取 Receiver 目前还缺少的
            # 注意：needs[r] 在循环过程中会不断变小
            current_need = needs[r]

            # 交集：sender 有 且 receiver 缺
            # 使用 TensorSet.__and__ (&)
            to_transfer = available & current_need

            if len(to_transfer) > 0:
                # 1. 加入传输计划
                # 逻辑: p2p_plan[s][r].update(to_transfer)
                # TensorSet 写法: 必须赋值回去，因为 | 返回新对象
                p2p_plan[s][r] = p2p_plan[s][r] | to_transfer

                # 2. 关键步骤：贪心消除
                # 逻辑: need.difference_update(to_transfer)
                # TensorSet 写法: 计算差集并**回写到列表**
                needs[r] = current_need - to_transfer

    # 返回更新后的 needs (即剩下的必须去 CPU 拿的部分) 和 P2P 计划
    return needs, p2p_plan


def handle_update(
    needs: list[TensorSet],
    avails: list[TensorSet],
) -> tuple[
    list[TensorSet],  # leftovers_on_gpu
    list[list[TensorSet]],  # p2p_transfer_plan
]:
    """
    (TensorSet 版本)
    计算 P2P 传输计划：Sender-Centric
    """
    n_gpus = len(needs)

    # --- 0. 准备工作：创建一个兼容的 Empty TensorSet ---
    # 我们需要一个空的 TensorSet 来初始化，且 device/dtype 要匹配，
    # 否则 PyTorch 做运算时可能会报错。
    ref_ts = None
    if n_gpus > 0:
        if len(needs[0]) > 0:
            ref_ts = needs[0]
        elif len(avails[0]) > 0:
            ref_ts = avails[0]

    if ref_ts is not None:
        empty_tensor = torch.tensor(
            [], dtype=ref_ts.data.dtype, device=ref_ts.data.device
        )
    else:
        # Fallback: 默认 CPU Long
        empty_tensor = torch.tensor([], dtype=torch.long)

    empty_ts = TensorSet(empty_tensor)

    # --- 1. 初始化结果容器 ---
    # 注意：这里创建一个 n x n 的矩阵，填充 empty_ts
    p2p_plan: list[list[TensorSet]] = [
        [empty_ts for _ in range(n_gpus)] for _ in range(n_gpus)
    ]
    leftovers: list[TensorSet] = []

    # --- 2. 计算 P2P 计划 (双层循环: Sender -> Receiver) ---
    for s in range(n_gpus):
        new_update = avails[s]

        # 记录 Sender (s) 发送给所有人的节点总量，用于最后算剩下的
        everything_sender_sends = empty_ts

        for r in range(n_gpus):
            # 只有当 s != r 时才发生网络传输
            # (虽然 s==r 求交集逻辑上也成立，但在 P2P 计划里通常指跨卡传输)

            need = needs[r]

            # 交集: S 有 & R 需要
            # 使用 TensorSet 的 __and__ (&)
            to_transfer = new_update & need

            if len(to_transfer) > 0:
                # 记录传输计划
                # 因为只遍历一次 (s,r)，直接赋值即可
                p2p_plan[s][r] = to_transfer

                # 累加到 total sent
                # 使用 TensorSet 的 __or__ (|)
                everything_sender_sends = everything_sender_sends | to_transfer

        # --- 3. 计算 Leftovers (Sender 持有，但没人要的) ---
        # 逻辑: 我有的 - 我送出去的
        # 使用 TensorSet 的 __sub__ (-)
        leftover = new_update - everything_sender_sends
        leftovers.append(leftover)

    return leftovers, p2p_plan


def dedup_updates(
    updates: list[TensorSet], reverse: bool = True
) -> tuple[list[TensorSet], TensorSet]:
    """
    使用 TensorSet 实现的去重逻辑。
    """
    n_gpus = len(updates)

    # 创建一个空的 TensorSet 作为初始值
    # 注意：确保 dtype 与你的节点 ID 类型一致（通常是 torch.long）
    empty_tensor = torch.tensor([], dtype=torch.long)
    # 如果是在 CUDA 上，记得 .cuda()，或者确保 updates 里的 tensor 都在同一个设备上
    if n_gpus > 0 and updates[0].data.is_cuda:
        empty_tensor = empty_tensor.to(updates[0].data.device)

    empty_ts = TensorSet(empty_tensor)

    # 初始化结果列表
    new_updates: list[TensorSet] = [empty_ts] * n_gpus

    if n_gpus == 0:
        return new_updates, empty_ts

    all_update = empty_ts

    # 确定迭代顺序
    iterator = range(n_gpus - 1, -1, -1) if reverse else range(n_gpus)

    for i in iterator:
        curr_rank_update_nodes = updates[i]

        # 1. 计算当前 GPU 独有的节点：(当前节点) - (已经被高优先级 GPU 抢走的节点)
        # 利用了 TensorSet.__sub__ (-)
        # 结果 new_updates[i] 会是一个新的 TensorSet 对象
        new_updates[i] = curr_rank_update_nodes - all_update

        # 2. 将当前节点加入到"已分配"集合中
        # 利用了 TensorSet.__or__ (|)
        # 注意：这里是重新赋值，因为 | 运算符返回的是一个新的 TensorSet
        all_update = all_update | curr_rank_update_nodes

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
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
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
        self.pending: MetasDyNeed | None = None

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            in_q_data = self.in_q.get(block=True)
            if in_q_data is None:
                self.out_q.put(None)
                break
            elif in_q_data == "end_of_epoch":
                assert self.pending is not None
                needs, updates, metas = self.pending
                # 没有下一个 batch 了，next_needs 为空
                if needs and len(needs) > 0:
                    ref_t = needs[0].data
                    empty_tensor = torch.tensor(
                        [], dtype=ref_t.dtype, device=ref_t.device
                    )
                else:
                    empty_tensor = torch.tensor([], dtype=torch.long)  # 默认 fallback
                empty_ts = TensorSet(empty_tensor)
                next_needs: list[TensorSet] = [empty_ts for _ in range(self.n_gpus)]
                plan = self._plan(needs, updates, next_needs)
                # this time will generate a plan, but next time will not generate plan
                self.out_q.put(plan)
                self.pending = None
                continue
            if self.pending is None:
                needs, updates, metas = in_q_data
                self.pending = needs, updates, metas
                self.fetchs_of_1 = needs
            else:
                needs, updates, metas = self.pending
                plan = self._plan(needs, updates, in_q_data[0])
                self.out_q.put(plan)
                self.pending = in_q_data

    def _plan(
        self,
        avails_of_1: list[TensorSet],
        update_avails_of_1: list[TensorSet],
        needs_of_2: list[TensorSet],
    ) -> list[DynamicPlan]:
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
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        empty_tensor = torch.tensor([], dtype=torch.long, device=device)
        empty_ts = TensorSet(empty_tensor)

        # 2. 初始化 NxN 矩阵
        # 因为 TensorSet 是不可变的，所有格子共享同一个 empty_ts 对象完全没问题
        p2p: list[list[TensorSet]] = [
            [empty_ts for _ in range(n_gpus)] for _ in range(n_gpus)
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
            plan = DynamicPlan(
                p2p=p2p[src],
                fetch=fetchs_of_1[src],
                writeback=leftovers_of_1[src],
                recv_counts=p2p_cts[:, src],  # 两个长度相等,可以逐位相加
                send_counts=p2p_cts[src, :],  # 两个长度相等,可以逐位相加
            )
            plans.append(plan)
        return plans
