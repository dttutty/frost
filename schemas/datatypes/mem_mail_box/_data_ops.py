from __future__ import annotations

import logging
from typing import TypeVar

import torch
from torch_scatter import scatter_sum

from .box import BoxTsNid
from .mail import MailTsNid
from .mem import MemTsNid


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")
logger = logging.getLogger(__name__)


def compute_last_indices(nid: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """
    通用逻辑：计算去重后应当保留的索引。
    策略：按 nid 分组，保留 ts 最大的那一行。
    返回：indices_to_keep
    """
    if nid.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=nid.device)

    # 1. 双重排序 (Sort by TS desc -> Stable sort by NID)
    perm_ts = torch.argsort(ts, descending=True)
    # stable=True 保证相同 NID 内部，TS 大的排在前面
    perm_nid = torch.argsort(nid[perm_ts], stable=True)

    final_perm = perm_ts[perm_nid]
    sorted_nid = nid[final_perm]

    # 2. 构造掩码 (Keep first of each group)
    mask = torch.empty(sorted_nid.shape[0], dtype=torch.bool, device=nid.device)
    mask[0] = True
    mask[1:] = sorted_nid[1:] != sorted_nid[:-1]

    # 3. 返回保留的索引
    return final_perm[mask]


def keep_last(data: Mem_or_Mail) -> None:
    assert data.ts is not None, "TS must be provided for keep_last"
    indices = compute_last_indices(data.nid, data.ts)

    # 2. 切片更新自己特有的字段
    if indices.numel() < data.nid.numel():  # 只有确实有重复才更新，微小优化
        data.nid = data.nid[indices]
        data.data = data.data[indices]  # MemTsNid 特有的字段
        data.ts = data.ts[indices]

        if data.ct is not None:
            data.ct = data.ct[indices]  # MailTsNid 特有的字段


def keep_mean(data: Mem_or_Mail) -> None:
    """
    聚合策略：对于重复的 nid，计算 data 和 ts 的加权均值。
    同时：
    1. 维护 ct (计数)：如果是首次合并，默认原权重为1；如果有旧权重，则累加。
    2. 维护 ts：计算加权平均时间 (Weighted Mean Time)。
    实现：原地修改。
    """
    assert data.ts is not None, "TS must be provided for keep_mean"

    if data.nid.numel() == 0:
        return

    # 1. ID 映射 (Mapping)
    unique_nids, compact_idx = torch.unique(data.nid, return_inverse=True)
    M = unique_nids.size(0)

    # 2. 处理权重 (Weights)
    if data.ct is None:
        weights = torch.ones_like(data.ts, dtype=torch.float32)
    else:
        weights = data.ct.float()

    # 3. 聚合计数 (Sum Weights) -> 分母
    # [M]
    new_ct = scatter_sum(weights, compact_idx, dim=0, dim_size=M)

    # 4. 聚合数据 (Weighted Sum Data) -> 分子 1
    if data.ct is not None:
        weighted_data = data.data * weights.unsqueeze(-1)
    else:
        weighted_data = data.data

    sum_data = scatter_sum(weighted_data, compact_idx, dim=0, dim_size=M)

    # 5. 聚合时间 (Weighted Sum TS) -> 分子 2  <--- 修改重点在这里
    # 时间也要乘权重 (例如：5条消息在 t=10, 1条消息在 t=20，均值应偏向 10)
    weighted_ts = data.ts * weights
    sum_ts = scatter_sum(weighted_ts, compact_idx, dim=0, dim_size=M)

    # 6. 计算均值 (Divide)
    # 注意: new_ct 至少为 1，除法安全
    new_data = sum_data / new_ct.unsqueeze(-1)
    new_ts = sum_ts / new_ct  # [M] / [M] -> [M]

    # 7. 更新自身
    data.nid = unique_nids
    data.data = new_data
    data.ts = new_ts
    data.ct = new_ct.long()


def merge_with_priority(
    high_priority_data: Mem_or_Mail, low_priority_data: Mem_or_Mail
) -> Mem_or_Mail:
    """
    合并 low_priority_data 和 high_priority_data。
    规则：优先保留 high_priority_data 的数据。如果 nid 在 high_priority_data 中存在，则忽略 high_priority_data 中的对应行。
    """
    # 1. 判空处理：如果 B 是空的，直接返回 A
    if low_priority_data.nid.numel() == 0:
        return type(high_priority_data)(
            nid=high_priority_data.nid,
            data=high_priority_data.data,
            ts=high_priority_data.ts,
        )
    # 如果 A 是空的，直接返回 B
    if high_priority_data.nid.numel() == 0:
        return type(low_priority_data)(
            nid=low_priority_data.nid,
            data=low_priority_data.data,
            ts=low_priority_data.ts,
        )

    # -------------------------------------------------------
    # 核心逻辑: 使用 torch.isin 找出 B 中冲突的节点
    # -------------------------------------------------------
    # mask 为 True 的位置表示：B 中的这个 nid 也在 A 中出现了
    # (注意：self.nid 不需要去重，isin 会自动处理)
    mask_in_high = torch.isin(low_priority_data.nid, high_priority_data.nid)

    # 我们只保留那些 mask 为 False (即不在 A 中) 的数据
    mask_keep_B = ~mask_in_high

    # 2. 对 B 进行过滤
    b_nid_filtered = low_priority_data.nid[mask_keep_B]
    b_mem_filtered = low_priority_data.data[mask_keep_B]
    b_ts_filtered = low_priority_data.ts[mask_keep_B]

    new_ct = None
    if high_priority_data.ct is not None and low_priority_data.ct is not None:
        b_ct_filtered = low_priority_data.ct[mask_keep_B]
        new_ct = torch.cat([high_priority_data.ct, b_ct_filtered], dim=0)

    ret = type(high_priority_data)(
        nid=torch.cat([high_priority_data.nid, b_nid_filtered], dim=0),
        data=torch.cat([high_priority_data.data, b_mem_filtered], dim=0),
        ts=torch.cat([high_priority_data.ts, b_ts_filtered], dim=0),
        ct=new_ct,
    )
    ret = ret.sort_by_nid()

    return ret
