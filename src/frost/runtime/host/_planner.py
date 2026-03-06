import logging
import threading
from queue import Queue as Q
from typing import cast

import numpy as np

from frost.aliases.host import (
    DynamicPlanInQ,
    DynamicPlanOutQ,
    MetaBundleQ,
    StaticPlanInQ,
    StaticPlanOutQ,
)
from frost.datatypes.access import Access
from frost.runtime._thread_base import ThreadBase
from frost.runtime.host.p2p_planner._dy_planner import DynamicPlanner
from frost.runtime.host.p2p_planner._st_planner import StaticPlanner
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


def find_all_gpu_distances_list(gpus_groups: list[list[int]]) -> np.ndarray:
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

    mem_plan_in_q: None | DynamicPlanInQ
    mail_plan_in_q: None | DynamicPlanInQ
    nfeat_plan_in_q: None | StaticPlanInQ

    def __init__(
        self,
        gpu_topo: list[int],
        in_q: MetaBundleQ,
        out_q: MetaBundleQ,
        mem_plan_out_q: DynamicPlanOutQ | None,
        mail_plan_out_q: DynamicPlanOutQ | None,
        nfeat_plan_out_q: StaticPlanOutQ | None,
        stop_ev: threading.Event,
    ) -> None:
        """
        初始化规划器。

        Args:
            gpus_groups: GPU 拓扑结构, 用于计算距离。
        """
        super().__init__(name="planner", stop_ev=stop_ev)
        # 1. 存储静态配置 (解决痛点 1 和 4)
        self.gpus_distance = find_all_gpu_distances_list(gpu_topo)
        self.n_gpus = len(self.gpus_distance)

        # 2. 为中间结果准备占位符 (解决痛点 2)
        self.meta_bundle_q = in_q
        self.out_q = out_q

        # mem_plan
        self.mem_plan_in_q: None | DynamicPlanInQ = None
        if mem_plan_out_q:
            self.mem_plan_in_q = Q()
            assert mem_plan_out_q is not None
            self.mem_planner = DynamicPlanner(
                gpus_distance=self.gpus_distance,
                in_q=self.mem_plan_in_q,
                out_q=mem_plan_out_q,
                stop_ev=stop_ev,
            )

        # mail plan
        self.mail_plan_in_q: None | DynamicPlanInQ = None
        if mail_plan_out_q:
            self.mail_plan_in_q = Q()
            assert mail_plan_out_q is not None
            self.mail_planner = DynamicPlanner(
                gpus_distance=self.gpus_distance,
                in_q=self.mail_plan_in_q,
                out_q=mail_plan_out_q,
                stop_ev=stop_ev,
            )

        # nfeat plan
        self.nfeat_plan_in_q: None | StaticPlanInQ = None
        if nfeat_plan_out_q:
            self.nfeat_plan_in_q = Q()
            assert nfeat_plan_out_q is not None
            self.nfeat_planner = StaticPlanner(
                gpus_distance=self.gpus_distance,
                in_q=self.nfeat_plan_in_q,
                out_q=nfeat_plan_out_q,
                stop_ev=stop_ev,
            )

        self.stop_ev = stop_ev

        self.mem_planner.start() if mem_plan_out_q else None
        self.mail_planner.start() if mail_plan_out_q else None
        self.nfeat_planner.start() if nfeat_plan_out_q else None

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            metas = self.meta_bundle_q.get(block=True)

            if metas is None:
                self._cleanup()
                break

            if metas == "end_of_epoch":
                self._clean_epoch()
                continue

            logger.info(
                f"[Planner] Received batch {metas[0].batch_idx} with {len(metas)} metas"
            )
            assert len(metas) == self.n_gpus

            access_list: list[Access] = []
            for idx, meta in enumerate(metas):
                assert meta.access is not None
                access_list.append(meta.access)

            if self.mem_plan_in_q is not None:
                next_needs = [cast(TensorSet, io.required_mem) for io in access_list]
                next_updates = [cast(TensorSet, io.mem_update) for io in access_list]
                self.mem_plan_in_q.put((next_needs, next_updates, metas))

            if self.mail_plan_in_q is not None:
                next_needs = [cast(TensorSet, io.required_box) for io in access_list]
                next_updates = [cast(TensorSet, io.box_update) for io in access_list]
                self.mail_plan_in_q.put((next_needs, next_updates, metas))

            if self.nfeat_plan_in_q is not None:
                next_needs = [cast(TensorSet, io.required_nfeat) for io in access_list]
                self.nfeat_plan_in_q.put(next_needs)

            for meta in metas:
                meta.access = None  # 释放内存

            self.out_q.put(metas)

    def _cleanup(self) -> None:
        if self.mem_plan_in_q is not None:
            self.mem_plan_in_q.put(None)
        if self.mail_plan_in_q is not None:
            self.mail_plan_in_q.put(None)
        if self.nfeat_plan_in_q is not None:
            self.nfeat_plan_in_q.put(None)
        self.out_q.put(None)

        if self.mem_plan_in_q is not None:
            self.mem_planner.join()
        if self.mail_plan_in_q is not None:
            self.mail_planner.join()
        if self.nfeat_plan_in_q is not None:
            self.nfeat_planner.join()

    def _clean_epoch(self) -> None:
        if self.mem_plan_in_q is not None:
            self.mem_plan_in_q.put("end_of_epoch")
        if self.mail_plan_in_q is not None:
            self.mail_plan_in_q.put("end_of_epoch")
        if self.nfeat_plan_in_q is not None:
            self.nfeat_plan_in_q.put("end_of_epoch")
        self.out_q.put("end_of_epoch")
