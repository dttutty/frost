# <!--
# d2h thread
# 知道下个个batch需要什么数据

# 知道什么数据应该从别的GPU要
# 数据组合完成后就要把之前的数据删掉

# 知道下
#  -->

from typing import List, Any

import torch
import torch.distributed as dist



def split_data_to_tensors_send_to_self():
    # next_batch_plan_nids are all nodes: supporting nodes + root nodes

    # this batch_plan_nids are all nodes: supporting nodes(who has no mail) + root nodes
    # 最开始: 把所有节点的memory更新一下
    # 这里其实就可以把memory拿去存盘了,还有mail也可以生成了(接下)
    # 然后用更新的memory来计算src pos_dst neg_dst的emb
    # (接上)哪些memory需要存盘:所有节点的memory,所有节点的mail
    # need to write back to disk nodes: only positive root nodes - next_batch_plan_nids
    pass

def split_data_to_tensors_send_to_neigh():
    # next_batch_plan_nids are all nodes: supporting nodes + root nodes
    # this batch_plan_nids are all nodes: supporting nodes + root nodes
    # need to write back to disk nodes: only 1st layer suppoting nodes - next_batch_plan_nids
    pass
def split_data_to_tensors(
    next_batch_plan_nids: List[torch.Tensor],  # len = num_devices
    my_nids: List[torch.Tensor],  # len = num_devices
    priority_matrix: List[
        List[int]
    ],  # shape = [num_devices][num_devices], priority_matrix[dst][src]
) -> tuple[list[list[torch.Tensor]], list[list[int]]], Any:
    """
    计算从各源设备 -> 各目标设备 的数据发送计划:
      from_to[src][dst] = LongTensor(应由src发送给dst的节点ID)
    同时返回 missing_per_dst[dst] = list(未找到的ID)
    """
    num_devices = len(next_batch_plan_nids)
    assert num_devices == len(my_nids), "list长度不一致"
    assert num_devices == len(priority_matrix), "priority_matrix 维度不匹配"
    for row in priority_matrix:
        assert len(row) == num_devices, "priority_matrix 每行长度应等于设备数"

    # 初始化结果矩阵
    from_to: list[list[torch.Tensor]] = [
        [torch.empty(0, dtype=torch.long) for _ in range(num_devices)]
        for _ in range(num_devices)
    ]

    # 初始化缺失ID列表
    missing_per_dst: list[list[int]] = [[] for _ in range(num_devices)]

    # 源设备当前持有的ID集合
    src_sets = [set(t.tolist()) for t in my_nids]

    # 对每个目标设备处理
    for dst in range(num_devices):
        need_list = next_batch_plan_nids[dst].tolist()
        remaining = set(need_list)

        # 按优先级从各源拣ID
        src_order = sorted(
            range(num_devices), key=lambda s: (priority_matrix[dst][s], s)
        )
        for src in src_order:
            if not remaining:
                break
            picked = [
                nid for nid in need_list if nid in remaining and nid in src_sets[src]
            ]
            if picked:
                from_to[src][dst] = torch.tensor(picked, dtype=torch.long)
                remaining.difference_update(picked)

        # 剩余没找到的ID
        if remaining:
            missing_per_dst[dst] = sorted(list(remaining))

    uniq_next_batch_plan_nids: set[int] = set().union(
        *[t.tolist() for t in next_batch_plan_nids]
    )

    # 初始化结果列表（每个设备一个）
    need_to_write_back_batch_plan_nids: list[list[int]] = [
        list() for _ in range(len(my_nids))
    ]

    # 计算每个设备中“当前批次但下批不再需要”的节点ID
    for i, ids in enumerate(my_nids):
        curr_set = set(ids.tolist())
        need_to_write_back_batch_plan_nids[i] = sorted(
            curr_set - uniq_next_batch_plan_nids
        )

    return from_to, missing_per_dst, need_to_write_back_batch_plan_nids


from typing import List, Tuple


def build_plan_on_rank0(
    gathered_src: List[
        torch.Tensor
    ],  # len = D, rank i 的 this_batch_plan_nids（在CPU上）
    gathered_need: List[
        torch.Tensor
    ],  # len = D, rank i 的 next_batch_plan_nids（在CPU上）
    priority_matrix: List[List[int]],  # [D][D], priority[dst][src] 越小越优先
) -> Tuple[list[dict], list[list[int]]]:
    """
    返回:
      send_plan_per_src[src] = { dst: LongTensor(ids_to_send_from_src_to_dst on CPU) }
      missing_per_dst[dst]   = list[int]
    """
    D = len(gathered_need)
    # 用你已有的函数:
    from_to, missing = split_data_to_tensors(
        next_batch_plan_nids=[t.clone() for t in gathered_need],
        my_nids=[t.clone() for t in gathered_src],
        priority_matrix=priority_matrix,
    )

    # 按 src 组织成易于下游发送的结构
    send_plan_per_src: list[dict] = [dict() for _ in range(D)]
    for src in range(D):
        for dst in range(D):
            if from_to[src][dst].numel() > 0:
                send_plan_per_src[src][dst] = from_to[src][dst].cpu().contiguous()
    return send_plan_per_src, missing


def scatter_plan(send_plan_per_src):
    """
    rank0: 传入 send_plan_per_src（长度 = world_size，每个元素是该src的dict{dst: ids(LongTensor-CPU)}）
    非rank0: 传入 None
    返回:
      my_send_plan: dict{dst: LongTensor(ids)}  —— 本rank需要发送的ID（分dst）
    """
    world = dist.get_world_size()
    out_list = [None for _ in range(world)] if dist.get_rank() == 0 else [None]
    if dist.get_rank() == 0:
        # send_plan_per_src[i] 将发给 rank i
        dist.scatter_object_list(out_list, scatter_list=send_plan_per_src, src=0)
    else:
        dist.scatter_object_list(out_list, scatter_list=None, src=0)
    return out_list[0]  # dict{dst: LongTensor on CPU}


def exchange_features_with_all_to_all(
    my_send_plan: dict,  # {dst: LongTensor(ids on CPU)}
    feature_table_device: torch.device,  # 本rank特征表所在GPU
    fetch_fn,  # 给你一个 ids(LongTensor on GPU)，返回对应 features(2D/ND Tensor on GPU)
):
    # 1) 准备每个 dst 的特征块（放到本rank GPU上）
    device = feature_table_device
    dsts = sorted(my_send_plan.keys())
    blocks = []
    split_sizes = []
    for dst in dsts:
        ids_cpu = my_send_plan[dst]
        ids = ids_cpu.to(device, non_blocking=True)
        feats = fetch_fn(ids)  # -> Tensor on device
        blocks.append(feats)
        split_sizes.append(feats.shape[0])  # 按样本数切分

    if len(blocks) == 0:
        # 没有要发送的，发送空
        send_cat = torch.empty(
            (0,), device=device, dtype=torch.float32
        )  # 按你的特征维度构造
        split_sizes = [0] * dist.get_world_size()
    else:
        send_cat = torch.cat(blocks, dim=0)

    # 2) all_to_all_single（需要对方也按同样顺序准备 split_sizes；最简单做法是大家都事先 all_gather 一下 split_sizes）
    world = dist.get_world_size()
    # 先把 split_sizes 交换一下，得到每个 src->我 的大小，便于分配接收缓冲
    my_sizes_tensor = torch.tensor([0] * world, dtype=torch.int64, device=device)
    for i, dst in enumerate(dsts):
        my_sizes_tensor[dst] = split_sizes[i]

    all_sizes = [
        torch.zeros(world, dtype=torch.int64, device=device) for _ in range(world)
    ]
    dist.all_gather(all_sizes, my_sizes_tensor)
    recv_sizes = torch.stack(all_sizes, dim=0)[
        :, dist.get_rank()
    ].tolist()  # [src] -> rowsize

    # 根据 recv_sizes 分配接收缓冲
    rows = sum(recv_sizes)
    feat_dim = send_cat.shape[1] if send_cat.ndim > 1 else 1  # 按你的实际特征维度
    recv_cat = torch.empty((rows, feat_dim), device=device, dtype=send_cat.dtype)

    dist.all_to_all_single(
        output=recv_cat,
        input=send_cat,
        output_split_sizes=recv_sizes,
        input_split_sizes=split_sizes,
    )

    # 现在 recv_cat 里按 src 顺序依次堆叠了你需要接收的特征；如需按 (src, 顺序) 拆分可据 recv_sizes 再切回去
    return recv_cat


# def route_three_way(
#     x_to0: torch.Tensor, x_to1: torch.Tensor, x_to2: torch.Tensor, group=None
# ):
#     """
#     当前 rank 的三块数据（发给 r0/r1/r2），都在同一 device、同 dtype。
#     返回 (recv_from0, recv_from1, recv_from2)
#     """
#     # 1) 准备输入：按 rank 顺序 [to0, to1, to2] 拼接
#     in_chunks = [x_to0.contiguous(), x_to1.contiguous(), x_to2.contiguous()]
#     in_buf = torch.cat(in_chunks, dim=0)
#     in_sizes = [c.numel() for c in in_chunks]  # 以元素个数为单位
#     world = dist.get_world_size(group)

#     # 2) 各 rank 交换“发出切片的大小”，以便计算 out_buf 的拆分（不均等大小时需要）
#     #    注意：NCCL 后端现在支持 all_to_all_single 的 split_sizes（各 rank 可不等长）。
#     in_sizes_t = torch.tensor(in_sizes, device=in_buf.device, dtype=torch.int64)
#     sizes_gather = [torch.empty_like(in_sizes_t) for _ in range(world)]
#     dist.all_gather(sizes_gather, in_sizes_t, group=group)
#     # sizes_gather[r][k] = 来自 rank r 发往 rank k 的元素数

#     # 3) 计算我将要收到的总量（来自所有源的“发给我”的切片之和）
#     my_rank = dist.get_rank(group)
#     out_sizes = [sizes_gather[src][my_rank].item() for src in range(world)]
#     total_out_elems = sum(out_sizes)

#     # 4) 分配输出缓冲并进行 all_to_all_single
#     out_buf = torch.empty(total_out_elems, dtype=in_buf.dtype, device=in_buf.device)

#     # 注意：PyTorch 的 all_to_all_single 以“元素个数”来解释 split_sizes
#     dist.all_to_all_single(
#         output=out_buf,
#         input=in_buf,
#         output_split_sizes=out_sizes,  # 我将从各 src 收到多少元素
#         input_split_sizes=in_sizes,  # 我发给各 dst 多少元素（按 [to0,to1,to2]）
#         group=group,
#     )

#     # 5) 把 out_buf 按来源 rank 切回三块（从 rank0/1/2 来的）
#     o0 = out_buf[0 : out_sizes[0]]
#     o1 = out_buf[out_sizes[0] : out_sizes[0] + out_sizes[1]]
#     o2 = out_buf[sum(out_sizes[:2]) : sum(out_sizes)]
#     return o0, o1, o2
