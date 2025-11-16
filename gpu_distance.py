from typing import List, Optional, Set, Tuple, Union

import numpy as np


def compute_whats_last_mail_gonna_updated_by_gpu_no_mailbox(pos_edges):
    return compute_whats_mem_gonna_updated_by_gpu(pos_edges)


def compute_whats_last_mail_gonna_updated_by_gpu_has_mailbox(mfgss):
    raise NotImplementedError(
        "你需要实现 compute_whats_last_mail_gonna_updated_by_gpu_has_mailbox"
    )


def compute_whats_mail_gonna_updated_by_gpu(mail_delivery_block):
    raise NotImplementedError("你需要实现 compute_whats_mail_gonna_updated_by_gpu")


def compute_whats_need_by_gpu(mfgss) -> Set:
    raise NotImplementedError("你需要实现 compute_whats_need_by_gpu")
    s = set()
    return s


def compute_whats_mem_gonna_updated_by_gpu(pos_edgess):
    gpus_number = len(pos_edgess)
    this_gpu_freshs = [set() for _ in range(gpus_number)]

    # --- 修复: 循环范围必须包含 0 ---
    # 修复: 使用 gpu_id, 而不是未定义的 'id'
    # 修复: 直接从 'edge' 获取 src/dst
    for gpu_id in range(gpus_number):
        for edge in pos_edgess[gpu_id]:
            # edge 应该是一个 (src, dst) 元组
            src = edge[0]
            dst = edge[1]
            this_gpu_freshs[gpu_id].add(src)
            this_gpu_freshs[gpu_id].add(dst)
    return this_gpu_freshs


def deduplicate_this_root_nodess(
    this_root_nodess: List[Set[int]], reverse: bool = True
) -> List[Set[int]]:
    """
    (JIT 向量化版本)
    对每个 GPU 持有的节点列表进行去重，确保一个节点只由一个 GPU 负责。
    """
    n_gpus = len(this_root_nodess)
    if n_gpus == 0:
        return []

    # 1. JIT 向量化
    (fresh_held_mask,), local_nids_map, _ = _vectorize_inputs_jit(this_root_nodess)

    if local_nids_map.size == 0:
        return [set() for _ in range(n_gpus)]

    n_active_nids = local_nids_map.shape[0]

    # 2. 向量化计算
    already_assigned_mask = np.zeros(n_active_nids, dtype=bool)
    dedup_mask = np.zeros_like(fresh_held_mask)

    gpu_id_iterator = range(n_gpus - 1, -1, -1) if reverse else range(n_gpus)

    for i in gpu_id_iterator:
        current_nids_mask = fresh_held_mask[i]
        newly_assigned_mask = current_nids_mask & ~already_assigned_mask
        dedup_mask[i] = newly_assigned_mask
        already_assigned_mask |= current_nids_mask

    # 3. 反向量化
    return _devectorize_mask_to_list_set(dedup_mask, local_nids_map)


def resolve_next_batch_fresh_data_sources(
    needed_by_gpu: List[Set[int]],
    fresh_nids_held_by_gpu: List[Set[int]],
    strategy: str = "latest",
) -> Tuple[
    List[Set[int]],  # stale_needs_by_gpu
    List[List[Set[int]]],  # p2p_transfer_plan
    Set[int],  # all_fresh_nids
    List[Set[int]],  # leftovers_on_gpu
    np.ndarray,  # rec_cts
    np.ndarray,  # send_cts
]:
    """
    (JIT 向量化版本)
    此函数处理稀疏的全局 NID (几亿) 和稠密的活动 NID (几千)。
    """

    # 1. JIT 向量化 (稀疏 List[Set] -> 稠密 [G, U_active] 掩码)
    (needed_mask, fresh_held_mask), local_nids_map, _ = _vectorize_inputs_jit(
        needed_by_gpu, fresh_nids_held_by_gpu
    )

    if local_nids_map.size == 0:
        # 处理空输入的边缘情况
        n_gpus = len(needed_by_gpu)
        empty_set_list = [set() for _ in range(n_gpus)]
        empty_ll_set = [[set() for _ in range(n_gpus)] for _ in range(n_gpus)]
        empty_counts = np.zeros(n_gpus, dtype=int)
        return (
            empty_set_list,
            empty_ll_set,
            set(),
            empty_set_list,
            empty_counts,
            empty_counts,
        )

    # 2. 核心计算 (在 *小型* 稠密掩码上)

    # 2A. 计算 Stale 和 Fresh 需求
    all_fresh_nids_mask = np.any(fresh_held_mask, axis=0)  # [U_active]
    required_fresh_nids_mask = needed_mask & all_fresh_nids_mask  # [G, U_active]
    stale_needs_mask = needed_mask & ~all_fresh_nids_mask  # [G, U_active]

    # 2B. 计算 P2P 传输计划
    fresh_send_mask = np.expand_dims(fresh_held_mask, axis=1)  # [S, 1, U_active]
    fresh_need_mask = np.expand_dims(
        required_fresh_nids_mask, axis=0
    )  # [1, R, U_active]
    p2p_plan_mask = fresh_send_mask & fresh_need_mask  # [S, R, U_active]

    # 2C. 计算 Leftovers
    all_nids_to_send_mask = np.any(p2p_plan_mask, axis=1)  # [S, U_active]
    leftover_mask = fresh_held_mask & ~all_nids_to_send_mask  # [G, U_active]

    # 2D. 计算 Counts
    p2p_counts_matrix = np.sum(p2p_plan_mask, axis=2, dtype=np.int32)  # [S, R]
    send_cts = p2p_counts_matrix.sum(axis=1)  # [S]
    rec_cts = p2p_counts_matrix.sum(axis=0)  # [R]

    # 3. 反向量化 (稠密 [G, U_active] 掩码 -> 稀疏 List[Set])
    stale_needs_by_gpu = _devectorize_mask_to_list_set(stale_needs_mask, local_nids_map)
    p2p_transfer_plan = _devectorize_p2p_mask_to_ll_set(p2p_plan_mask, local_nids_map)
    all_fresh_nids = set(local_nids_map[all_fresh_nids_mask])  # 全局 NID
    leftovers_on_gpu = _devectorize_mask_to_list_set(leftover_mask, local_nids_map)

    return (
        stale_needs_by_gpu,
        p2p_transfer_plan,
        all_fresh_nids,
        leftovers_on_gpu,
        rec_cts,
        send_cts,
    )


def resolve_next_batch_stale_data_sources(
    all_nids_held_by_gpu: List[Set[int]],
    stale_needs_by_gpu: List[Set[int]],  # <-- 统一了参数名和文档
    gpus_distance: np.ndarray,
    all_fresh_nids: Optional[Set[int]],
) -> Tuple[List[Set[int]], List[List[Set[int]]], np.ndarray, np.ndarray]:
    """
    解析下一批次的“陈旧数据”需求。
    它会优先尝试从最近的、持有该陈旧数据的P2P GPU获取，
    最后返回仍无法满足的需求（必须从CPU预取）。

    Args:
        all_nids_held_by_gpu: 每个GPU当前持有的 *所有* 节点ID。
        stale_needs_by_gpu: 每个GPU需要的 *陈旧* 节点ID。 (此函数不会修改它)
        gpus_distance: GPU间的距离矩阵。
        all_fresh_nids: 所有 *新鲜* 节点的集合。

    Returns:
        Tuple (cpu_prefetch_needs_by_gpu, stale_p2p_transfer_plan):
        1. cpu_prefetch_needs_by_gpu: 最终仍未被满足的需求 (需从CPU获取)。
        2. stale_p2p_transfer_plan: 陈旧数据的P2P传输计划 [sender][receiver]。
    """

    # 1. 计算每个GPU实际持有的 *陈旧* 节点 (不修改输入)
    if all_fresh_nids is not None:
        stale_nids_held_by_gpu = [
            held_set - all_fresh_nids for held_set in all_nids_held_by_gpu
        ]
    else:
        stale_nids_held_by_gpu = [s.copy() for s in all_nids_held_by_gpu]

    n_gpus = len(stale_nids_held_by_gpu)

    # 2. 初始化 P2P 传输计划
    stale_p2p_transfer_plan: List[List[Set]] = [
        [set() for _ in range(n_gpus)] for _ in range(n_gpus)
    ]

    # --- 【 Bug 修复 】 ---
    # 3. 创建 'stale_needs_by_gpu' 的深拷贝。
    #    我们将修改这个拷贝，而不是原始输入。
    cpu_prefetch_needs_by_gpu = [s.copy() for s in stale_needs_by_gpu]
    # --- 【 修复结束 】 ---

    # 4. 按距离从小到大（0, 1, 2...）满足需求
    unique_distances = sorted(np.unique(gpus_distance))

    for distance in unique_distances:
        for sender in range(n_gpus):
            for receiver in range(n_gpus):
                # 如果(sender, receiver)是当前距离
                if gpus_distance[sender, receiver] == distance:
                    # 计算 sender 能为 receiver 提供的节点
                    # (读取我们创建的拷贝)
                    transferable_nids = (
                        stale_nids_held_by_gpu[sender]
                        & cpu_prefetch_needs_by_gpu[receiver]
                    )

                    # 只有在确实有可传输的节点时才记录
                    if transferable_nids:
                        # 记录到 P2P 计划
                        stale_p2p_transfer_plan[sender][receiver] = transferable_nids

                        # --- 【 Bug 修复 】 ---
                        # 从 receiver 的“待获取列表”(的拷贝)中移除已满足的节点
                        cpu_prefetch_needs_by_gpu[receiver] -= transferable_nids
                        # --- 【 修复结束 】 ---

    # 5. 循环结束后，返回这个被修改过的 *拷贝*
    p2p_counts_matrix = np.array(
        [[len(s) for s in row] for row in stale_p2p_transfer_plan]
    )
    send_cts = p2p_counts_matrix.sum(axis=1)
    rec_cts = p2p_counts_matrix.sum(axis=0)
    return cpu_prefetch_needs_by_gpu, stale_p2p_transfer_plan, rec_cts, send_cts


def find_all_gpu_distances_list(gpus_groups: list) -> np.ndarray:
    """
    (List Version)
    根据嵌套的 Python 列表结构计算距离矩阵。
    假设 GPU ID 是从 0 开始的连续/非连续整数。

    Args:
        gpus_groups: 一个嵌套的 list, e.g., [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
                     或不平衡的 e.g., [[0, 1], [2, 3], [4]]

    Returns:
        np.ndarray: 一个 (N, N) 的距离矩阵, N = max(gpu_id) + 1。
    """

    # --- 内部辅助函数 ---

    def _find_all_gpu_ids(node: Union[int, list]) -> Set[int]:
        """递归查找所有叶节点 (GPU IDs)。"""
        if isinstance(node, int):
            return {node}
        if isinstance(node, list):
            # (Use set.union with * to merge sets from children)
            return set().union(*(_find_all_gpu_ids(child) for child in node))
        return set()

    def _get_max_depth(node: Union[int, list]) -> int:
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

    def _fill_distances(
        current_node: Union[int, list], current_depth: int
    ) -> List[int]:
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
        child_gpu_groups: List[List[int]] = []
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


class MultiGPUDataPlanner:
    """
    编排多GPU训练中的内存、特征和Mailbox的数据传输计划。
    """

    def __init__(
        self,
        gpus_groups: list,
        mail_strategy: str = "latest",
        mail_delivery_to: str = "self",
    ):
        """
        初始化规划器。

        Args:
            gpus_groups: GPU 拓扑结构, 用于计算距离。
            mail_delivery_to: "self", "neighbor"
            mail_strategy:  "self"可以是"latest", "mean","neighbor"可以是"latest", "all",
        """
        # 1. 存储静态配置 (解决痛点 1 和 4)
        self.gpus_distance: np.ndarray = find_all_gpu_distances_list(gpus_groups)
        self.n_gpus: int = len(self.gpus_distance)
        self.mail_strategy = mail_strategy
        self.mail_delivery_to = mail_delivery_to

        # 2. 为中间结果准备占位符 (解决痛点 2)
        self._this_needs: List[Set[int]] = []
        self._next_needs: List[Set[int]] = []
        self._all_fresh_nids: Set[int] = set()
        self._stale_needs: List[Set[int]] = []

    def plan_step(
        self,
        this_mfgsss: list,
        next_mfgsss: list,
        pos_edges: list,
        mail_delivery_blocks: Optional[list] = None,
    ):
        """
        为当前训练步骤规划所有数据流动。
        """
        # 3. 一次性计算通用需求 (解决痛点 3)
        self._this_needs = [compute_whats_need_by_gpu(m) for m in this_mfgsss]
        self._next_needs = [compute_whats_need_by_gpu(m) for m in next_mfgsss]

        # 4. 依次调用各个子规划器
        # (注意：这里的执行顺序很重要，因为它们共享中间状态)

        mem_plan = self._plan_memory(pos_edges)
        mail_plan = self._plan_mail(pos_edges, mail_delivery_blocks)
        feat_plan = self._plan_features()

        return mem_plan, mail_plan, feat_plan

    def _plan_memory(self, pos_edges: list, strategy: str = "latest"):
        """规划内存（Memory）的P2P、CPU预取和CPU写回。"""

        # 计算特定于内存的更新
        changed_nodes = compute_whats_mem_gonna_updated_by_gpu(pos_edges)
        if strategy == "latest":
            changed_nodes = deduplicate_this_root_nodess(changed_nodes)

        # === 解决痛点 2：中间结果存入 self ===
        stale_needs, fresh_p2p, all_fresh, leftovers, fresh_rec_cts, fresh_send_cts = (
            resolve_next_batch_fresh_data_sources(
                self._next_needs, changed_nodes, strategy
            )
        )
        # 暂存中间结果
        self._stale_needs = stale_needs
        self._all_fresh_nids = all_fresh

        cpu_prefetch, stale_p2p, rec_cts, send_cts = (
            resolve_next_batch_stale_data_sources(
                self._this_needs,
                self._stale_needs,  # <-- 从 self 读取
                self.gpus_distance,
                self._all_fresh_nids,  # <-- 从 self 读取
            )
        )  # 这种没有新mem的节点,直接发送旧的mem就可以(由distance最近的发)

        return {
            "fresh_p2p": fresh_p2p,
            "stale_p2p": stale_p2p,
            "cpu_prefetch": cpu_prefetch,
            "cpu_writeback": leftovers,
            "rec_cts": fresh_rec_cts + rec_cts,  # 两个长度相等,可以逐位相加
            "send_cts": fresh_send_cts + send_cts,  # 两个长度相等,可以逐位相加
        }

    def _plan_mail(self, pos_edges: list, mail_delivery_blocks: Optional[list]):
        """规划 Mail (Mailbox / Mail) 的传输。"""

        # === 解决痛点 4：封装策略 ===
        if self.mail_delivery_to == "self" and self.mail_strategy:
            assert mail_delivery_blocks is None
            assert self.mail_strategy in ["latest", "mean"]
            return self._plan_memory(pos_edges, strategy=self.mail_strategy)

        elif self.mail_delivery_to == "neighbor":
            assert mail_delivery_blocks is not None
            assert self.mail_strategy in ["latest", "all"]
            # 不能用this_needs和next_needs对比
            this_batch_changed_nodess = [
                compute_whats_mail_gonna_updated_by_gpu(m) for m in mail_delivery_blocks
            ]
            if self.mail_strategy == "latest":
                this_batch_changed_nodess = deduplicate_this_root_nodess(
                    this_batch_changed_nodess
                )
            (
                stale_needs_by_gpu,
                fresh_p2p_transfer_plan,
                all_fresh_nids,
                leftovers_on_gpu,
                rec_cts,
                send_cts,
            ) = resolve_next_batch_fresh_data_sources(
                self._next_needs, this_batch_changed_nodess, self.mail_strategy
            )
            return {
                "fresh_p2p": fresh_p2p_transfer_plan,
                "cpu_writeback": leftovers_on_gpu,
                "rec_cts": rec_cts,
                "send_cts": send_cts,
            }

    def _plan_for_mailbox(self):
        # mailbox，只有“陈旧” P2P 和 CPU 预取,最后由目的地GPU把mail写入mailbox即可
        return self._plan_features()

    def _plan_features(self):
        """规划节点特征（Node Features）的传输。"""

        # 特征没有“新鲜”概念，只有“陈旧” P2P 和 CPU 预取
        cpu_prefetch, stale_p2p, rec_cts, send_cts = (
            resolve_next_batch_stale_data_sources(
                self._this_needs,  # 持有者：所有 this_batch 需要的节点
                self._next_needs,  # 需求者：所有 next_batch 需要的节点
                self.gpus_distance,
                all_fresh_nids=None,  # 关键：没有 fresh 节点
            )
        )

        return {
            "stale_p2p": stale_p2p,
            "cpu_prefetch": cpu_prefetch,
            "rec_cts": rec_cts,
            "send_cts": send_cts,
        }
