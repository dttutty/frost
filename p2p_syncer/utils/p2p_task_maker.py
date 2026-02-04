import logging
import sys
from typing import Literal, TypeVar

import numpy
import torch

from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.misc import Meta
from schemas.datatypes.plan import DyPlan
from schemas.datatypes.task import P2PTask


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")
logger = logging.getLogger(__name__)


def make_p2p_task(
    data_td_nid: AllType,
    plan: DyPlan,
    strategy: Literal["last", "mean", "all"],
    device: torch.device,
    meta: Meta | None = None,
) -> P2PTask:
    """
    优化版: 利用 is_sorted 属性使用 searchsorted 加速查找
    """
    p2p = plan.p2p
    rec_cts = plan.rec_cts
    send_cts = plan.send_cts

    nid = data_td_nid.nid
    data = data_td_nid.data
    ts = data_td_nid.ts

    # 确保源数据是有序的，这是 searchsorted 的前提
    # 如果 data_td_nid.is_sorted 不为 True，这里的优化会得到错误结果
    if not getattr(data_td_nid, "is_sorted", False):
        # 如果还没排序，这里必须排一下，或者回退到 isin
        # 考虑到你之前的逻辑，这里应该是 sorted 的，加个 assert 确保安全
        # assert False, "data_td_nid must be sorted for this optimization"
        # 如果实在不敢保证，可以临时排序（但会有开销），建议在上游保证 sorted
        data_td_nid = data_td_nid.sort_by_nid()
        nid = data_td_nid.nid
        data = data_td_nid.data
        ts = data_td_nid.ts

    buckets_ids, buckets_data, buckets_ts = [], [], []
    sizes = numpy.zeros(len(p2p), dtype=int)

    # 预先获取 source nid 的大小，避免反复访问
    src_len = nid.numel()

    for dst_gpu_id, _ in enumerate(p2p):
        nids_to_send_tensor = p2p[dst_gpu_id].data
        expected_count = nids_to_send_tensor.numel()

        if expected_count == 0:
            sizes[dst_gpu_id] = 0
            continue

        # --- [优化开始] 使用 searchsorted ---

        # 1. 查找插入位置 (Indices)
        # nid 必须是 1D sorted tensor
        # nids_to_send_tensor 不需要 sorted，但如果是 sorted 会更快一点点
        indices = torch.searchsorted(nid, nids_to_send_tensor)

        # 2. 边界检查 & 值匹配检查
        # searchsorted 返回的是插入位置，可能越界(==src_len)，或者该位置的值不相等

        # 先把越界的下标修剪一下，防止下一步索引报错 (clamp 到最后一个元素)
        indices_clamped = indices.clamp(max=src_len - 1)

        # 检查找到的位置是否真的是我们要找的值
        # mask_found shape 与 nids_to_send_tensor 相同
        mask_found = nid[indices_clamped] == nids_to_send_tensor

        # 3. 提取有效的索引 (Valid Indices in Source)
        # 只保留那些真正找到的
        valid_gather_indices = indices_clamped[mask_found]

        matched_count = valid_gather_indices.numel()

        # --- [优化结束] ---

        if matched_count != expected_count:
            logger.warning(
                f"make_p2p_task: GPU {dst_gpu_id} expected to send {expected_count} nodes, "
                f"but found {matched_count} in data (total {nid.numel()})."
            )

        # Gather
        # 使用 index_select 或者直接索引 (mask vs indices)
        # 这里 valid_gather_indices 是在源数据中的下标，可以直接取数据
        buckets_ids.append(nid[valid_gather_indices])
        buckets_data.append(data[valid_gather_indices])
        buckets_ts.append(ts[valid_gather_indices])

        sizes[dst_gpu_id] = matched_count

    if len(buckets_ids) > 0:
        send_nid = torch.cat(buckets_ids, dim=0)
        send_data = torch.cat(buckets_data, dim=0)
        send_ts = torch.cat(buckets_ts, dim=0)
    else:
        # 处理没有任何数据要发送的情况
        send_nid = torch.tensor([], dtype=nid.dtype, device=nid.device)
        send_data = torch.tensor([], dtype=data.dtype, device=data.device).reshape(
            0, data.shape[1]
        )
        send_ts = torch.tensor([], dtype=ts.dtype, device=ts.device)

    total_planned_send = sum(send_cts)
    real_shape = send_nid.shape[0]

    if total_planned_send != real_shape:
        # 获取 rank，防止并未初始化 dist 的边缘情况报错，虽然后续都会崩
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = "Unknown"

        name = "make_p2p_task_construction"

        print(f"\n[CRITICAL ERROR] P2P Mismatch on Rank {rank} | Op: {name}")
        print(f"  > Sum of send_cts (Planned): {total_planned_send}")
        print(f"  > Actual Tensor dim0 (Gathered): {real_shape}")
        print(f"  > Tensor Full Shape: {send_nid.shape}")
        print(f"  > Send Cts List (Planned): {send_cts}")
        print(f"  > Actual Gathered Sizes: {sizes}")
        sys.stdout.flush()

        # 抛出异常让程序停在这里
        raise RuntimeError(
            f"AAA Split size mismatch in {name}: planned {total_planned_send} vs actual {real_shape}"
        )

    total_recv = sum(rec_cts)
    recv_nid = torch.empty(total_recv, dtype=nid.dtype, device=device)

    data_shape = list(data.shape)
    data_shape[0] = total_recv
    recv_data = torch.empty(data_shape, dtype=data.dtype, device=device)

    ts_shape = list(ts.shape)
    ts_shape[0] = total_recv
    recv_ts = torch.empty(ts_shape, dtype=ts.dtype, device=device)

    assert numpy.array_equal(sizes, send_cts)

    p2p_task = P2PTask(
        nid2send=send_nid,
        nid2recv=recv_nid,
        data2send=send_data,
        data2recv=recv_data,
        ts2send=send_ts,
        ts2recv=recv_ts,
        rec_cts=rec_cts,
        send_cts=send_cts,
    )

    return p2p_task
