from typing import List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from dgl.heterograph import DGLBlock

from schemas.datatypes import Meta, Plan, PlanSet


def _mail_delivery_tensor(
    unique_root_nodes: torch.Tensor, mail_delivery_block: DGLBlock
) -> torch.Tensor:
    src_local_id, dst_local_id = mail_delivery_block.edges(
        order="eid"
    )  # find out all edges
    global_src = mail_delivery_block.srcdata["ID"][src_local_id]
    global_dst = mail_delivery_block.dstdata["ID"][dst_local_id]

    uniq_global_src = torch.unique(global_src)
    uniq_global_dst = torch.unique(global_dst)
    is_in_root = torch.isin(global_src, unique_root_nodes)
    # uniq_global_dst to set
    uniq_global_dst_set = set(uniq_global_dst.tolist())
    roots_sorted = torch.sort(unique_root_nodes).values
    # roots_sorted to set
    roots_sorted_set = set(roots_sorted.tolist())

    src_sorted = torch.sort(global_src).values

    final_src = global_src[is_in_root]
    final_dst = global_dst[is_in_root]
    tensor = torch.stack([final_src, final_dst], dim=0)
    return tensor


def stale_data_planer(
    all_nodes: List[Set[int]],
    stale_needs_by_gpu: List[Set[int]],
    gpus_distance: np.ndarray,
    all_fresh: Optional[Set[int]],
) -> Tuple[List[Set[int]], List[List[Set[int]]], np.ndarray]:
    """
    (纯 Python Set 版本)
    计算陈旧数据的 P2P 计划和 CPU 预取。
    逻辑：距离优先的贪心匹配 (Greedy Matching by Distance)。
    Receiver-Centric
    """
    n_gpus = len(all_nodes)
    p2p_counts = np.zeros((n_gpus, n_gpus), dtype=np.int32)
    assert n_gpus > 0

    # 1. 预处理：确定每个 GPU 实际持有的 "Stale" 数据
    # 如果提供了 fresh 集合，必须从持有列表中排除它们（因为 fresh 优先用于 fresh plan）
    if all_fresh:
        stale_held_by_gpu = [held_set - all_fresh for held_set in all_nodes]
    else:
        # 如果没有 fresh 集合，那所有持有的都可以视为 stale 候选项
        # 浅拷贝一次，避免后续修改影响原数据（虽然这里只读取）
        stale_held_by_gpu = all_nodes

    # 2. 复制需求列表（因为我们会在贪心匹配过程中不断“划掉”已满足的需求）
    # remaining_needs 最终将只包含 "谁都没有，必须去 CPU 拿" 的节点
    remaining_needs_by_gpu: List[Set[int]] = [s.copy() for s in stale_needs_by_gpu]

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
            available = stale_held_by_gpu[s]
            # 获取 Receiver 目前还缺少的 (注意：这里是动态更新的 remaining_needs)
            need: Set[int] = remaining_needs_by_gpu[r]

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

    # 4. 此时 remaining_needs_by_gpu 里剩下的就是 P2P 搞不定的，只能去 CPU 拿
    prefetch = remaining_needs_by_gpu

    return prefetch, p2p_plan, p2p_counts


def fresh_data_planner(
    next_all_list: List[Set[int]],
    mem_changed_nodes: List[Set[int]],
    all_ranks_fresh_nodes: Set[int],
) -> Tuple[
    List[Set[int]],  # stale_needs_by_gpu
    List[List[Set[int]]],  # p2p_transfer_plan
    List[Set[int]],  # leftovers_on_gpu
    np.ndarray,
]:
    """
    (纯 Python Set 版本 - 逻辑简化)
    计算 P2P 传输计划：谁有(fresh)，谁需要(needed)，谁发给谁。
    Sender-Centric
    """
    n_gpus = len(next_all_list)

    # 1. 全局 Fresh 集合: 所有 GPU 目前持有的节点并集
    # 等价于: np.any(fresh_held_mask, axis=0)

    assert set().union(*mem_changed_nodes) == all_ranks_fresh_nodes

    # 2. 计算 Stale Needs (缺少的节点)
    # 逻辑: 我需要的 - 全局任何人有fresh的 = 我必须去Stale存储distance优先拿的 (Stale) +需要从CPU预取的
    # 等价于: needed_mask & ~all_fresh_nids_mask
    stale_needs_by_gpu = [needed - all_ranks_fresh_nodes for needed in next_all_list]

    # 初始化结果容器
    p2p_plan = [[set() for _ in range(n_gpus)] for _ in range(n_gpus)]
    leftovers_on_gpu = []

    # 使用 numpy 数组以保持返回类型兼容性
    p2p_counts = np.zeros((n_gpus, n_gpus), dtype=np.int32)

    # 3. 计算 P2P 计划 (双层循环: Sender -> Receiver)
    # 逻辑: Sender 有的 & Receiver 需要的 = 传输内容
    # 对于 GPU 数量 (通常 < 16) 来说，O(N^2) 的循环开销甚至小于 Numpy 的启动开销
    for s in range(n_gpus):
        available = mem_changed_nodes[s]

        # 记录 Sender 发送出去的所有东西，用于计算 leftover
        everything_sender_sends = set()

        for r in range(n_gpus):
            need = next_all_list[r]

            # 交集: S 有且 R 需要
            # 等价于: fresh_send_mask & fresh_need_mask
            to_transfer = available & need

            if to_transfer:
                p2p_plan[s][r].update(to_transfer)
                everything_sender_sends.update(to_transfer)

                # 更新计数
                p2p_counts[s, r] = len(to_transfer)

        # 4. 计算 Leftovers (Sender 持有有，但没人要的)
        # 等价于: fresh_held_mask & ~all_nids_to_send_mask
        leftover = available - everything_sender_sends
        leftovers_on_gpu.append(leftover)

    return (
        stale_needs_by_gpu,
        p2p_plan,
        leftovers_on_gpu,
        p2p_counts,
    )


def calc_future_writeback(
    current_updates_by_rank: List[Set[int]],  # B2, B3 更新了什么
    future_needs_by_rank: List[Set[int]],  # B4, B5 需要什么
) -> List[Set[int]]:
    """
    实现逻辑 3:
    如果 B2,3 更新了某数据，但 B4,5 不需要它，则 B2,3 需要将其 Writeback 到 CPU。
    """
    n_gpus = len(current_updates_by_rank)

    # 1. 汇总未来的所有需求 (B4 + B5 的并集)
    #    不管是谁需要，只要 GPU Cluster 里有人需要，就不能且不需要 Writeback
    all_future_needs = set().union(*future_needs_by_rank)

    writeback_plans = []
    for rank in range(n_gpus):
        # 该 Rank 产生的新数据
        my_updates = current_updates_by_rank[rank]

        # 逻辑：我产生的 - 未来需要的 = 只有 CPU 需要 (Writeback)
        to_writeback = my_updates - all_future_needs
        writeback_plans.append(to_writeback)

    return writeback_plans


def dedup_this_root_nodess(
    mem_update_nodes_all_ranks: List[Set[int]], reverse: bool = True
) -> Tuple[List[Set[int]], Set[int]]:
    """
    (简化逻辑版)
    对每个 GPU 持有的节点列表进行去重，确保一个节点只由一个 GPU 负责。
    优先保留迭代顺序中靠前的 GPU 中的节点。

    Args:
        u_mem_update_p_rank: List[torch.Tensor], 列表索引对应 rank/gpu_id
        reverse:
            True  -> 倒序优先 (Rank N-1 优先保留节点, Rank 0 只能捡剩下的)
            False -> 正序优先 (Rank 0 优先保留节点)
    """
    n_gpus = len(mem_update_nodes_all_ranks)
    return_value: List[Set[int]] = [set() for _ in range(n_gpus)]
    if n_gpus == 0:
        return return_value, set()
    seen_nodes: Set[int] = set()
    # seen_nodes is the set of nodes that have already been assigned to higher-priority GPUs
    iterator = range(n_gpus - 1, -1, -1) if reverse else range(n_gpus)
    for i in iterator:
        curr_rank_update_nodes = mem_update_nodes_all_ranks[i]
        return_value[i] = curr_rank_update_nodes - seen_nodes
        seen_nodes.update(curr_rank_update_nodes)

    return return_value, seen_nodes


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


def _comp_need_edges(blockss: List[List[DGLBlock]]) -> torch.Tensor:
    all_eid_list = []
    for blocks in blockss:
        for block in blocks:
            all_eid_list.append(block.edata["ID"])
    if len(all_eid_list) == 0:
        return torch.empty(0, dtype=torch.long)
    all_eid_tensor = torch.cat(all_eid_list)
    unique_all_eid_tensor = torch.unique(all_eid_tensor)
    return unique_all_eid_tensor


def compute_whats_need_by_gpu(blocks: List[DGLBlock]) -> torch.Tensor:
    if not blocks:
        return torch.empty(0, dtype=torch.long)
    all_nid_list = []
    for block in blocks:
        all_nid_list.append(block.srcdata["ID"])
    all_nid_tensor = torch.cat(all_nid_list)
    return torch.unique(all_nid_tensor)


def _comp_need_nodes(meta: Meta) -> tuple[Set, None | Set[int], None | Set[int]]:
    """
    Helper method to process a single batch.

    It calculates dependencies (needed features, memory updates, mail updates)
    and stores intermediate results (like pos_edges) back into the batch object.

    Returns:
        tuple: (whats_need, whose_mem_gonna_updated, whats_mail_gonna_updated)
    """

    root_nodes = torch.unique(meta.pos_edge)
    mem_update_nodes = mailbox_update_nodes = None
    mem_update_nodes = root_nodes

    assert meta.mdb is not None
    meta.mdt = _mail_delivery_tensor(root_nodes, mail_delivery_block=meta.mdb)
    mailbox_update_nodes = torch.unique(meta.mdt)

    all_nodes = compute_whats_need_by_gpu(meta.blockss[0])

    return (
        set(all_nodes.tolist()),
        set(mem_update_nodes.tolist()) if mem_update_nodes is not None else None,
        set(mailbox_update_nodes.tolist())
        if mailbox_update_nodes is not None
        else None,
    )


class ApanPlanner:
    """
    编排多GPU训练中的内存、特征和Mailbox的数据传输计划。
    """

    def __init__(
        self,
        gpus_groups: list,
        mail_strategy: Optional[Literal["mean", "last"]] = "last",
    ):
        """
        初始化规划器。

        Args:
            gpus_groups: GPU 拓扑结构, 用于计算距离。
            mail_strategy:  "self"可以是"latest", "mean","neighbors"可以是"latest", "all",
        """
        # 1. 存储静态配置 (解决痛点 1 和 4)
        self.gpus_distance: np.ndarray = find_all_gpu_distances_list(gpus_groups)
        self.n_gpus: int = len(self.gpus_distance)
        self.mail_strategy = mail_strategy
        self.mail_deliver_to = "neighbors"

        # 2. 为中间结果准备占位符 (解决痛点 2)
        self.mail_all_node_set: List[Set[int]]
        self.mailbox_all_node_set: List[Set[int]]
        self.mem_update_set: List[Set[int]] | None = None
        self.mailbox_update_set: List[Set[int]] | None = None
        self.initialized: bool = False
        self.mem_prefetch: List[Set[int]] = []
        self.mail_prefetch: List[Set[int]] = []

    def reset(self):
        self.mail_all_node_set = []
        self.mailbox_all_node_set = []
        self.mem_update_set = None
        self.mailbox_update_set = None
        self.initialized = False
        self.mem_prefetch = []
        self.mail_prefetch = []

    def plan_next(
        self,
        meta_list: List[Meta],
        is_last_macrobatch: bool,
    ) -> List[Meta]:
        # Use integer division (//) instead of float division (/)
        ngpus = self.n_gpus

        def comp_needs_4_one_macrobatch(metas: List[Meta]):
            mem_all_nodes_all_ranks = []
            mem_update_nodes_all_ranks = []
            mailbox_update_nodes_all_ranks = []

            for meta in metas:
                # Use the helper function
                (
                    all_nodes_per_rank,
                    all_mem_update_nodes_per_rank,
                    all_mailbox_update_nodes_per_rank,
                ) = _comp_need_nodes(meta)

                mem_all_nodes_all_ranks.append(all_nodes_per_rank)
                assert mem_update_nodes_all_ranks is not None
                assert mailbox_update_nodes_all_ranks is not None
                mem_update_nodes_all_ranks.append(all_mem_update_nodes_per_rank)
                mailbox_update_nodes_all_ranks.append(all_mailbox_update_nodes_per_rank)

            return (
                mem_all_nodes_all_ranks,
                mem_update_nodes_all_ranks,
                mailbox_update_nodes_all_ranks,
            )

        # --- 1. Handle Planner Initialization (if needed) ---
        # This part processes the *first half* of batches (0 to ngpus-1)
        # only if the planner hasn't been initialized.
        if not self.initialized:
            (
                self.mail_all_node_set,
                self.mem_update_set,
                self.mailbox_update_set,
            ) = comp_needs_4_one_macrobatch(meta_list[:ngpus])

            self.mem_prefetch.extend(self.mail_all_node_set)
            assert self.mem_update_set is not None
            assert self.mailbox_update_set is not None
            self.mail_prefetch.extend(self.mail_all_node_set)

        # --- 2. Plan for the next step ---
        next_uniq_all_nodes = []
        next_uniq_mem_update_nodes = []
        next_uniq_mailbox_update_list = []

        if is_last_macrobatch:
            # last macrobatch is only half size of normal macrobatch
            assert len(meta_list) == self.n_gpus
            for batch in meta_list:
                next_uniq_all_nodes.append(set())
                next_uniq_mem_update_nodes.append(set())
                next_uniq_mailbox_update_list.append(set())

        else:
            (
                next_uniq_all_nodes,
                next_uniq_mem_update_nodes,
                next_uniq_mailbox_update_list,
            ) = comp_needs_4_one_macrobatch(meta_list[ngpus:])

        # Call the main step planner using data from the second half
        mem_plan, box_plan = self.plan_step(next_uniq_all_nodes)
        self.mail_all_node_set = next_uniq_all_nodes
        self.mem_update_set = next_uniq_mem_update_nodes
        self.mailbox_update_set = next_uniq_mailbox_update_list

        # --- 3. Assign plans ---
        # Assign the generated plan to the *first half* of the batches
        # (which will be executed in the next stage/iteration).
        for b_id, batch in enumerate(meta_list[:ngpus]):
            batch.plan_set = PlanSet(
                box=box_plan[b_id],
                mem=mem_plan[b_id],
            )

            batch.plan_set = batch.plan_set.tensorfy()
            batch.plan_set = batch.plan_set.sort()
        #     print(fields(td_list[b_id].plan_set))
        # print(fields(td_list[0].plan_set))
        # print(fields(td_list[1].plan_set))
        self.initialized = True

        return meta_list

    def plan_step(
        self,
        next_all_node_set: List[Set[int]],
    ) -> Tuple[
        Union[List[Plan], None],
        Union[List[Plan], None],
    ]:
        """
        为当前训练步骤规划所有数据流动。
        """
        mail_plan = mem_plan = None
        assert self.mem_update_set is not None
        assert self.mailbox_update_set is not None
        which_gpu_hold_the_latest_which_memory, all_ranks_fresh_mem_nodes = (
            dedup_this_root_nodess(
                self.mem_update_set,
                reverse=True,
            )
        )
        mem_plan = self._plan_memory_or_mailbox(
            self.mail_all_node_set,
            which_gpu_hold_the_latest_which_memory,
            next_all_node_set,
            all_ranks_fresh_mem_nodes,
            "memory",
        )
        which_gpu_hold_the_latest_which_mailbox, all_ranks_fresh_mailbox_nodes = (
            dedup_this_root_nodess(
                self.mailbox_update_set,
                reverse=True,
            )
        )  # for mulitple GPU, we only keep the last one to update the mailbox
        mail_plan = self._plan_memory_or_mailbox(
            self.mail_all_node_set,
            which_gpu_hold_the_latest_which_mailbox,
            next_all_node_set,
            all_ranks_fresh_mailbox_nodes,
            mem_or_mail="mail",
        )

        return mem_plan, mail_plan

    def _plan_memory_or_mailbox(
        self,
        all_nodes: List[Set[int]],
        update_nodes: List[Set[int]],
        next_all_nodes: List[Set[int]],
        all_ranks_fresh_nodes: Set[int],
        mem_or_mail: str = "memory",
    ) -> List[Plan]:
        """规划内存（Memory）的P2P、CPU预取和CPU写回。"""

        # 计算特定于内存的更新

        # === 解决痛点 2：中间结果存入 self ===
        stale_needs, fresh_p2p, leftovers, fresh_p2p_counts = fresh_data_planner(
            next_all_nodes, update_nodes, all_ranks_fresh_nodes
        )

        prefetch, stale_p2p, p2p_counts = stale_data_planer(
            all_nodes,
            stale_needs,  # <-- 从 self 读取
            self.gpus_distance,
            all_ranks_fresh_nodes,  # <-- 从 self 读取
        )  # 这种没有新mem的节点,直接发送旧的mem就可以(由distance最近的发)
        if mem_or_mail == "memory":
            self.mem_prefetch.extend(prefetch)
        else:
            self.mail_prefetch.extend(prefetch)
        plans = []
        for src_gpu in range(self.n_gpus):
            p2p = [set() for _ in range(self.n_gpus)]
            for dst_gpu in range(self.n_gpus):
                intersection = fresh_p2p[src_gpu][dst_gpu] & stale_p2p[src_gpu][dst_gpu]
                assert len(intersection) == 0, (
                    "Fresh and Stale P2P sets should be disjoint."
                )
                p2p[dst_gpu] = fresh_p2p[src_gpu][dst_gpu] | stale_p2p[src_gpu][dst_gpu]
            if mem_or_mail == "memory":
                this_batch_prefetch = self.mem_prefetch.pop(0)
            elif mem_or_mail == "mail":
                this_batch_prefetch = self.mail_prefetch.pop(0)
            else:
                raise ValueError(f"Unknown mem_or_mail: {mem_or_mail}")
            current_rec_cts = fresh_p2p_counts[:, src_gpu] + p2p_counts[:, src_gpu]
            current_send_cts = fresh_p2p_counts[src_gpu, :] + p2p_counts[src_gpu, :]
            # --- 验证逻辑开始 ---
            if __debug__:  # 可以改成 if __debug__: 在生产环境跳过
                for other_gpu in range(self.n_gpus):
                    # A. 验证 Send: 我发给别人的 Set 长度 vs 我的 Send Count
                    # p2p[other_gpu] 是 src_gpu 准备发给 other_gpu 的数据集合
                    actual_send_len = len(p2p[other_gpu])
                    recorded_send_cnt = current_send_cts[other_gpu]
                    assert actual_send_len == recorded_send_cnt, (
                        f"Send Mismatch [GPU {src_gpu} -> GPU {other_gpu}]: "
                        f"Set has {actual_send_len} items, but Count is {recorded_send_cnt}"
                    )

                    # B. 验证 Recv: 别人发给我的 Set 长度 vs 我的 Recv Count
                    # 注意：'p2p' 变量只包含发送计划。
                    # 为了验证接收，我们需要看 '别人' (other_gpu) 准备发给 '我' (src_gpu) 的集合
                    incoming_data_set = (
                        fresh_p2p[other_gpu][src_gpu] | stale_p2p[other_gpu][src_gpu]
                    )
                    actual_recv_len = len(incoming_data_set)
                    recorded_recv_cnt = current_rec_cts[other_gpu]
                    assert actual_recv_len == recorded_recv_cnt, (
                        f"Recv Mismatch [GPU {other_gpu} -> GPU {src_gpu}]: "
                        f"Incoming Set has {actual_recv_len} items, but Count is {recorded_recv_cnt}"
                    )
            plan = Plan(
                p2p=p2p,
                prefetch=this_batch_prefetch,
                writeback=leftovers[src_gpu],
                rec_cts=current_rec_cts,  # 两个长度相等,可以逐位相加
                send_cts=current_send_cts,  # 两个长度相等,可以逐位相加
            )
            plans.append(plan)
        return plans
