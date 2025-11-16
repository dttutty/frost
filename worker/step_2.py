import torch
import torch.distributed as dist


# ---------------------------
# 共享：把本地 (node_ids, feats, feats_ts) 路由给各自 owner，owner 收齐后返回
# ---------------------------
@torch.no_grad()
def route_to_owner(
    node_ids: torch.Tensor,
    feats: torch.Tensor,
    feats_ts: torch.Tensor,
    owner_of,  # function: (node_ids: LongTensor) -> LongTensor in [0, world_size)
    world_size: int,
):
    """
    输入（本 rank 局部）:
      node_ids: [K_local], int64
      feats:    [K_local, D]
      feats_ts: [K_local]  (时间戳/步号，可为整型或浮点)
    按 owner_of(node_id) 路由到对应 owner（all_to_all_single 变长安全）。
    返回（本 rank 作为 owner 收到的）:
      recv_ids  : [K_recv]
      recv_feats: [K_recv, D]
      recv_ts   : [K_recv]
    非 owner rank 返回空张量（K_recv=0）。
    """
    assert node_ids.dim() == 1
    assert feats.dim() == 2 and feats.size(0) == node_ids.size(0)
    assert feats_ts.dim() == 1 and feats_ts.size(0) == node_ids.size(0)

    if world_size == 1:
        # 退化：无需通信
        return node_ids, feats, feats_ts

    rank = dist.get_rank()
    device = node_ids.device
    K_local = node_ids.size(0)
    D = feats.size(1)

    # 1) 按 owner 分桶
    owners = owner_of(node_ids)  # [K_local] in [0, world_size)
    buckets_ids, buckets_x, buckets_ts = [], [], []
    send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    for dst in range(world_size):
        m = owners == dst
        ids_dst = node_ids[m]
        x_dst = feats[m]
        ts_dst = feats_ts[m]
        buckets_ids.append(ids_dst)
        buckets_x.append(x_dst)
        buckets_ts.append(ts_dst)
        send_counts[dst] = ids_dst.numel()

    total_send = int(send_counts.sum().item())
    send_ids = (
        torch.cat(buckets_ids, dim=0)
        if total_send > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )
    send_x = (
        torch.cat(buckets_x, dim=0)
        if total_send > 0
        else torch.empty(0, feats.size(1), dtype=feats.dtype, device=device)
    )
    send_ts = (
        torch.cat(buckets_ts, dim=0)
        if total_send > 0
        else torch.empty(0, dtype=feats_ts.dtype, device=device)
    )

    # 2) 交换各自的 send_counts，得出我应接收多少（recv_counts）
    # all_gather_into_tensor 会把每个 rank 的 send_counts 行收集成 [world_size, world_size]
    all_counts = torch.empty(world_size, world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(all_counts, send_counts)
    recv_counts = all_counts[:, rank].contiguous()
    total_recv = int(recv_counts.sum().item())

    # 3) 申请接收缓冲
    recv_ids = torch.empty(total_recv, dtype=torch.long, device=device)
    recv_x = torch.empty(total_recv, D, dtype=feats.dtype, device=device)
    recv_ts = torch.empty(total_recv, dtype=feats_ts.dtype, device=device)

    in_splits = send_counts.tolist()
    out_splits = recv_counts.tolist()

    # 4) 变长 all_to_all_single：三路并行（id / feat / ts）
    dist.all_to_all_single(
        recv_ids, send_ids, out_split_sizes=out_splits, in_split_sizes=in_splits
    )
    dist.all_to_all_single(
        recv_x, send_x, out_split_sizes=out_splits, in_split_sizes=in_splits
    )
    dist.all_to_all_single(
        recv_ts, send_ts, out_split_sizes=out_splits, in_split_sizes=in_splits
    )

    return recv_ids, recv_x, recv_ts


# ---------------------------
# 聚合：owner 上对相同 node_id 求均值
# ---------------------------
@torch.no_grad()
def aggregate_owner_mean(recv_ids: torch.Tensor, recv_feats: torch.Tensor):
    """
    输入为 route_to_owner() 在本 owner rank 的输出：
      recv_ids  : [K_recv] int64
      recv_feats: [K_recv, D]
    输出：
      owned_nodes: [M] 唯一 node_id（排序升序）
      owned_means: [M, D] 按 node 聚合后的均值
    """
    device = recv_ids.device
    D = recv_feats.size(1)
    K = recv_ids.numel()
    if K == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, D, dtype=recv_feats.dtype, device=device),
        )

    uniq, inv = torch.unique(
        recv_ids, return_inverse=True, sorted=True
    )  # uniq: [M], inv: [K] in [0..M)
    M = uniq.size(0)
    sums = torch.zeros(M, D, dtype=recv_feats.dtype, device=device)
    sums.index_add_(0, inv, recv_feats)
    counts = torch.bincount(inv, minlength=M).clamp_min(1).to(sums.dtype).unsqueeze(1)
    means = sums / counts
    return uniq, means


# ---------------------------
# 聚合：owner 上按时间戳选择最新（最大 ts）
# ---------------------------
@torch.no_grad()
def aggregate_owner_latest(
    recv_ids: torch.Tensor, recv_feats: torch.Tensor, recv_ts: torch.Tensor
):
    """
    对每个 node 选择 ts 最大的一条记录。
    返回：
      owned_nodes : [M]
      latest_feats: [M, D]
      latest_ts   : [M]
    """
    device = recv_ids.device
    D = recv_feats.size(1)
    K = recv_ids.numel()
    if K == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, D, dtype=recv_feats.dtype, device=device),
            torch.empty(0, dtype=recv_ts.dtype, device=device),
        )

    # 为了高效地 per-node 取 argmax(ts)，先对 (node_id, ts) 排序，再做分段最大
    # 排序 key: (node_id, ts) 升序；同 ts 冲突可以再用原序/次序规则
    sort_idx = (
        torch.lexsort((recv_ts, recv_ids))
        if hasattr(torch, "lexsort")
        else torch.argsort(recv_ids * 0 + 0)
    )  # 兼容提示；多数环境无 torch.lexsort
    # 若无 lexsort，可改为：
    sort_idx = (
        torch.argsort(recv_ids * 1_000_000_000 + recv_ts)
        if recv_ts.dtype.is_floating_point is False
        else torch.argsort(
            torch.stack([recv_ids, recv_ts], dim=1).to(torch.float64).sum(dim=1)
        )
    )
    ids_sorted = recv_ids[sort_idx]
    ts_sorted = recv_ts[sort_idx]
    x_sorted = recv_feats[sort_idx]

    # 找到每个 node 的分段边界
    uniq, first_idx = torch.unique_consecutive(
        ids_sorted,
        return_counts=False,
        return_inverse=False,
        return_counts_indices=False,
        return_index=True,
    )
    # 对每段取 ts 最大项的索引：可以用 segment-wise argmax
    # 简化：对每段做一个“滚动比较”，我们用 group 最大：先找每段的最大 ts，再用掩码收集对应位置
    # 计算每段长度
    counts = torch.diff(
        torch.cat([first_idx, torch.tensor([ids_sorted.numel()], device=device)])
    )
    # 每段最大 ts
    max_ts = torch.zeros_like(uniq, dtype=ts_sorted.dtype, device=device)
    start = 0
    max_pos = torch.empty_like(uniq, dtype=torch.long, device=device)
    for i, c in enumerate(counts.tolist()):
        if c == 0:
            max_pos[i] = start
            continue
        seg_ts = ts_sorted[start : start + c]
        # 取段内最大
        rel = torch.argmax(seg_ts)
        max_pos[i] = start + rel
        start += c

    latest_nodes = uniq
    latest_ts = ts_sorted[max_pos]
    latest_feats = x_sorted[max_pos]
    return latest_nodes, latest_feats, latest_ts


# ---------------------------
# Mailbox：owner 上为每个 node 保留所有 (feat, ts)（Ragged/CSR 形式）
# ---------------------------
@torch.no_grad()
def build_owner_mailbox(
    recv_ids: torch.Tensor, recv_feats: torch.Tensor, recv_ts: torch.Tensor
):
    """
    返回稀疏邮箱：
      nodes   : [M] 唯一 node_id（升序）
      indptr  : [M+1] 累积索引，区间 indptr[i]:indptr[i+1] 是第 i 个 node 的所有消息
      feats_c : [K, D]  依照上面的拼接顺序
      ts_c    : [K]
    """
    device = recv_ids.device
    D = recv_feats.size(1)
    K = recv_ids.numel()
    if K == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.empty(0, D, dtype=recv_feats.dtype, device=device),
            torch.empty(0, dtype=recv_ts.dtype, device=device),
        )

    # 先按 node_id（可再按 ts）排序，保证每个 node 的消息是连续块
    sort_idx = torch.argsort(
        recv_ids
    )  # 如需按时间排序可用 torch.argsort(torch.stack([recv_ids, recv_ts],1), dim=0)
    ids_sorted = recv_ids[sort_idx]
    x_sorted = recv_feats[sort_idx]
    ts_sorted = recv_ts[sort_idx]

    nodes, first_idx = torch.unique_consecutive(ids_sorted, return_index=True)
    counts = torch.diff(
        torch.cat([first_idx, torch.tensor([ids_sorted.numel()], device=device)])
    )
    indptr = torch.zeros(nodes.numel() + 1, dtype=torch.long, device=device)
    indptr[1:] = torch.cumsum(counts, dim=0)

    return nodes, indptr, x_sorted, ts_sorted
