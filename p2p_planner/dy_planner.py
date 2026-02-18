import logging
import random
import threading
from queue import Queue
from typing import Literal

import numpy as np
import torch

from schemas.alias import MetasDyNeed
from schemas.datatypes.misc import Meta
from schemas.datatypes.plan import DyPlan
from tensor_set import TensorSet
from thread_base import ThreadBase


logger = logging.getLogger(__name__)


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


def dedup_updates_random(
    updates: list[TensorSet], seed: int | None = None
) -> tuple[list[TensorSet], TensorSet]:
    """
    Implements random deduplication using TensorSet.
    """
    n_gpus = len(updates)

    # Create an empty TensorSet for initialization
    # Ensure dtype matches node IDs (usually torch.long)
    empty_tensor = torch.tensor([], dtype=torch.long)

    # Move to the correct device if necessary
    if n_gpus > 0 and updates[0].data.is_cuda:
        empty_tensor = empty_tensor.to(updates[0].data.device)

    empty_ts = TensorSet(empty_tensor)

    # Initialize the result list with empty sets
    new_updates: list[TensorSet] = [empty_ts] * n_gpus

    if n_gpus == 0:
        return new_updates, empty_ts

    all_update = empty_ts

    # 1. Generate a random processing order
    # Using a list implies CPU operations, which is fine for the loop control
    indices = list(range(n_gpus))

    random.shuffle(indices)

    # 2. Iterate through GPUs in the random order
    for i in indices:
        curr_rank_update_nodes = updates[i]

        # Calculate unique nodes for the current GPU: current - seen
        # Assign result to the original index 'i', not the loop iteration index
        new_updates[i] = curr_rank_update_nodes - all_update

        # Add current nodes to the 'seen' set
        all_update = all_update | curr_rank_update_nodes

    return new_updates, all_update


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
                next_updates: list[TensorSet] = [empty_ts for _ in range(self.n_gpus)]
                print("------------------------------------------metas[0].batch_idx")
                print(metas[0].batch_idx)
                plan = self._plan(needs, updates, next_needs, next_updates, metas)
                # this time will generate a plan, but next time will not generate plan
                self.out_q.put(plan)
                self.pending = None
                continue
            if self.pending is None:
                needs, updates, metas = in_q_data
                self.pending = in_q_data
                self.fetch = needs
            else:
                needs, updates, metas = self.pending
                print("------------------------------------------metas[0].batch_idx")
                print(metas[0].batch_idx)

                plan = self._plan(needs, updates, in_q_data[0], in_q_data[1], metas)
                self.out_q.put(plan)
                self.pending = in_q_data

    def _plan(
        self,
        all_list: list[TensorSet],
        update_list: list[TensorSet],
        next_all_list: list[TensorSet],
        next_update_list: list[TensorSet],
        metas: list[Meta],
    ) -> list[DyPlan]:
        """规划内存（Memory）的P2P、CPU预取和CPU写回。"""

        n_gpus = self.n_gpus

        dedup_update_list, all_update = dedup_updates(update_list)
        next_all_update = TensorSet.empty_like(all_update.data)
        for update in next_update_list:
            next_all_update = next_all_update | update

        _, p2p = handle_update(next_all_list, dedup_update_list)
        writeback_list = [
            per_gpu_update - next_all_update for per_gpu_update in dedup_update_list
        ]

        # 1. 预处理：确定每个 GPU 实际持有的 "Stale" 数据
        # 如果提供了 fresh 集合，必须从持有列表中排除它们（因为 fresh 优先用于 fresh plan）
        # all_of_1 = [avail - all_update_avails_of_1 for avail in all_of_1]
        next_prefetch = [next_need - all_update for next_need in next_all_list]

        # needs_of_2, stale_p2p_1to2 = handle_remaining(
        #     needs_of_2,  # <-- 从 self 读取
        #     all_of_1,
        #     self.gpus_distance,
        # )  # 这种没有新mem的节点,直接发送旧的mem就可以(由distance最近的发)

        plans = []

        # 2. 初始化 NxN 矩阵
        # 因为 TensorSet 是不可变的，所有格子共享同一个 empty_ts 对象完全没问题

        p2p_cts = np.zeros((n_gpus, n_gpus), dtype=np.int32)

        for src in range(n_gpus):
            for dst in range(n_gpus):
                p2p_cts[src, dst] = len(p2p[src][dst])

        for src in range(n_gpus):
            plan = DyPlan(
                p2p=p2p[src],
                fetch=self.fetch[src],
                writeback=writeback_list[src],
                rec_cts=p2p_cts[:, src],  # 两个长度相等,可以逐位相加
                send_cts=p2p_cts[src, :],  # 两个长度相等,可以逐位相加
            )
            plans.append(plan)

        self.fetch = next_prefetch

        return plans
