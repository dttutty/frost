import logging
import threading
from queue import Queue as Q
from typing import Literal, cast

import numpy as np

from host.planner.dynamic_planner import DynamicPlanner
from host.planner.static_planner import StaticPlanner
from schemas.alias import MetaBundle, MetasDyNeed, MetasStNeed
from schemas.datatypes.plan import BatchNodeStateIO, DyPlan, StPlan
from tensor_set import TensorSet
from thread_base import ThreadBase


logger = logging.getLogger(__name__)


def find_all_gpu_distances_list(gpus_groups: list) -> np.ndarray:
    """
    根据嵌套的 Python 列表结构计算距离矩阵。
    假设 GPU ID 是从 0 开始的连续/非连续整数。

    Args:
        gpus_groups: 一个嵌套的 list, e.g., [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
                     或不平衡的 e.g., [[0, 1], [2, 3], [4]]

    Returns:
        np.ndarray: 一个 (N, N) 的距离矩阵, N = max(gpu_id) + 1。
    """

    # --- 内部辅助函数 ---

    def _find_all_gpu_ids(node: int | list) -> set[int]:
        """递归查找所有叶节点 (GPU IDs)。"""
        if isinstance(node, int):
            return {node}
        if isinstance(node, list):
            # (Use set.union with * to merge sets from children)
            return set().union(*(_find_all_gpu_ids(child) for child in node))
        return set()

    def _get_max_depth(node: int | list) -> int:
        """递归查找最大嵌套深度。"""
        if isinstance(node, int):
            return 0  # Leaf node has depth 0
        if isinstance(node, list) and node:
            # (Add 1 to the max depth of children)
            return 1 + max((_get_max_depth(child) for child in node), default=0)
        return 0  # Empty list or other type

    # --- 主逻辑 ---

    # 1. 找到所有 GPU IDs 并确定矩阵大小 N
    all_ids = _find_all_gpu_ids(gpus_groups)
    if not all_ids:
        return np.array([])

    # (Matrix size is determined by the largest ID found)
    n_gpus = max(all_ids) + 1

    # 2. 初始化矩阵
    distance_matrix = np.zeros((n_gpus, n_gpus), dtype=int)

    # 3. 获取拓扑的最大深度
    max_depth = _get_max_depth(gpus_groups)

    def _fill_distances(current_node: int | list, current_depth: int) -> list[int]:
        """
        递归辅助函数。
        填充距离矩阵，并返回在此节点下找到的 GPU ID 列表。
        """

        # Base Case: Reached a leaf (a single GPU ID)
        if isinstance(current_node, int):
            return [current_node]

        # Recursive Step: Node is a group (list)
        if not isinstance(current_node, list):
            return []

        # 1. 计算此层级兄弟节点间的距离
        # (Top level (depth 0) has highest distance)
        distance = max_depth - current_depth

        # 2. 递归到子节点
        child_gpu_groups: list[list[int]] = []
        for child_node in current_node:
            gpus_in_child = _fill_distances(child_node, current_depth + 1)
            child_gpu_groups.append(gpus_in_child)

        # 3. 填充所有 *跨* 子组对的距离
        for i in range(len(child_gpu_groups)):
            for j in range(i + 1, len(child_gpu_groups)):
                group_a = child_gpu_groups[i]
                group_b = child_gpu_groups[j]

                # (Update matrix for all combinations)
                for gpu_a in group_a:
                    for gpu_b in group_b:
                        distance_matrix[gpu_a, gpu_b] = distance
                        distance_matrix[gpu_b, gpu_a] = distance

        # 4. 返回此节点下所有 GPU ID 的扁平列表
        return [gpu_id for group in child_gpu_groups for gpu_id in group]

    # 5. 从根节点 (depth 0) 开始递归
    _fill_distances(gpus_groups, 0)

    return distance_matrix


class Planner(ThreadBase):
    """
    这个是对那种mailbox_update=mem_update的planner
    """

    def __init__(
        self,
        gpu_topo: list,
        in_q: Q[MetaBundle | None | Literal["end_of_epoch"]],
        out_q: Q[MetaBundle | None | Literal["end_of_epoch"]],
        mem_plan_out_q: Q[list[DyPlan] | None] | None,
        mail_plan_out_q: Q[list[DyPlan] | None] | None,
        nfeat_plan_out_q: Q[list[StPlan] | None] | None,
        stop_ev: threading.Event,
    ) -> None:
        """
        初始化规划器。

        Args:
            gpus_groups: GPU 拓扑结构, 用于计算距离。
        """
        logger.debug(f"Planner: initializing with gpu_topo={gpu_topo}")
        # 1. 存储静态配置 (解决痛点 1 和 4)
        self.gpus_distance = find_all_gpu_distances_list(gpu_topo)
        self.n_gpus = len(self.gpus_distance)
        logger.debug(
            f"Planner: n_gpus={self.n_gpus}, distance_matrix shape={self.gpus_distance.shape}"
        )

        # 2. 为中间结果准备占位符 (解决痛点 2)
        self.meta_bundle_q = in_q
        self.out_q = out_q

        # mem_plan
        self.mem_plan_in_q: None | Q[MetasDyNeed | None | Literal["end_of_epoch"]] = (
            None
        )
        if mem_plan_out_q:
            logger.debug("Planner: initializing mem planner")
            self.mem_plan_in_q = Q()
            assert mem_plan_out_q is not None
            self.mem_planner = DynamicPlanner(
                gpus_distance=self.gpus_distance,
                in_q=self.mem_plan_in_q,
                out_q=mem_plan_out_q,
                stop_ev=stop_ev,
            )

        # mail plan
        self.mail_plan_in_q: None | Q[MetasDyNeed | None | Literal["end_of_epoch"]] = (
            None
        )
        if mail_plan_out_q:
            logger.debug("Planner: initializing mail planner")
            self.mail_plan_in_q = Q()
            assert mail_plan_out_q is not None
            self.mail_planner = DynamicPlanner(
                gpus_distance=self.gpus_distance,
                in_q=self.mail_plan_in_q,
                out_q=mail_plan_out_q,
                stop_ev=stop_ev,
            )

        # nfeat plan
        self.nfeat_plan_in_q: None | Q[MetasStNeed | None | Literal["end_of_epoch"]] = (
            None
        )
        if nfeat_plan_out_q:
            logger.debug("Planner: initializing nfeat planner")
            self.nfeat_plan_in_q = Q()
            assert nfeat_plan_out_q is not None
            self.nfeat_planner = StaticPlanner(
                gpus_distance=self.gpus_distance,
                in_q=self.nfeat_plan_in_q,
                out_q=nfeat_plan_out_q,
                stop_ev=stop_ev,
            )

        self.stop_ev = stop_ev
        logger.debug("Planner: initialization complete")

    def logic(self) -> None:
        batch_count = 0
        logger.debug("Planner: starting logic loop")

        while not self.stop_ev.is_set():
            logger.debug("Planner: waiting for metas from meta_bundle_q")
            metas = self.meta_bundle_q.get(block=True)
            logger.debug(f"Planner: received data type={type(metas).__name__}")

            if metas is None:
                logger.debug("Planner: received None, shutting down sub-planners")
                if self.mem_plan_in_q is not None:
                    logger.debug("Planner: putting None to mem planner")
                    self.mem_plan_in_q.put(None)
                if self.mail_plan_in_q is not None:
                    logger.debug("Planner: putting None to mail planner")
                    self.mail_plan_in_q.put(None)
                if self.nfeat_plan_in_q is not None:
                    logger.debug("Planner: putting None to nfeat planner")
                    self.nfeat_plan_in_q.put(None)
                logger.debug("Planner: putting None to out_q")
                self.out_q.put(None)

                logger.debug("Planner: waiting for sub-planners to join")
                if self.mem_plan_in_q is not None:
                    logger.debug("Planner: joining mem planner")
                    self.mem_planner.join()
                if self.mail_plan_in_q is not None:
                    logger.debug("Planner: joining mail planner")
                    self.mail_planner.join()
                if self.nfeat_plan_in_q is not None:
                    logger.debug("Planner: joining nfeat planner")
                    self.nfeat_planner.join()
                logger.debug("Planner: all sub-planners joined, exiting")
                break

            if metas == "end_of_epoch":
                logger.debug(
                    "Planner: received end_of_epoch, forwarding to sub-planners"
                )
                if self.mem_plan_in_q is not None:
                    logger.debug("Planner: putting end_of_epoch to mem planner")
                    self.mem_plan_in_q.put("end_of_epoch")
                if self.mail_plan_in_q is not None:
                    logger.debug("Planner: putting end_of_epoch to mail planner")
                    self.mail_plan_in_q.put("end_of_epoch")
                if self.nfeat_plan_in_q is not None:
                    logger.debug("Planner: putting end_of_epoch to nfeat planner")
                    self.nfeat_plan_in_q.put("end_of_epoch")
                logger.debug("Planner: forwarding end_of_epoch to out_q")
                self.out_q.put("end_of_epoch")
                continue

            batch_count += 1
            logger.debug(
                f"Planner: processing batch #{batch_count}, metas count={len(metas)}"
            )
            assert len(metas) == self.n_gpus

            state_ios: list[BatchNodeStateIO] = []
            for idx, meta in enumerate(metas):
                assert meta.state is not None
                state_ios.append(meta.state)
                logger.debug(
                    f"Planner: batch #{batch_count}, GPU {idx} state collected"
                )

            if self.mem_plan_in_q is not None:
                logger.debug(
                    f"Planner: batch #{batch_count}, putting mem needs/updates to mem planner"
                )
                next_needs = [
                    cast(TensorSet, state_io.mem_needed) for state_io in state_ios
                ]
                next_updates = [
                    cast(TensorSet, state_io.mem_updated) for state_io in state_ios
                ]
                self.mem_plan_in_q.put((next_needs, next_updates))

            if self.mail_plan_in_q is not None:
                logger.debug(
                    f"Planner: batch #{batch_count}, putting mail needs/updates to mail planner"
                )
                next_needs = [
                    cast(TensorSet, state_io.mailbox_needed) for state_io in state_ios
                ]
                next_updates = [
                    cast(TensorSet, state_io.mailbox_updated) for state_io in state_ios
                ]
                self.mail_plan_in_q.put((next_needs, next_updates))

            if self.nfeat_plan_in_q is not None:
                logger.debug(
                    f"Planner: batch #{batch_count}, putting nfeat needs to nfeat planner"
                )
                next_needs = [
                    cast(TensorSet, state_io.nfeat_needed) for state_io in state_ios
                ]
                self.nfeat_plan_in_q.put(next_needs)

            logger.debug(f"Planner: batch #{batch_count}, deleting state from metas")
            for meta in metas:
                del meta.state  # 释放内存

            logger.debug(f"Planner: batch #{batch_count}, putting metas to out_q")
            self.out_q.put(metas)
            logger.debug(f"Planner: batch #{batch_count} complete")

        logger.debug(f"Planner: exiting logic loop, processed {batch_count} batches")
