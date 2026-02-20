from __future__ import annotations

import logging
from typing import TypeVar

import torch
from torch_scatter import (
    scatter_max,  # 新增
    scatter_sum,
)

from .box import BoxTsNid
from .mail import MailTsNid
from .mem import MemTsNid


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")
logger = logging.getLogger(__name__)


def compute_last_indices(nid: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    if nid.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=nid.device)

    unique_nids, compact_idx = torch.unique(nid, return_inverse=True)
    M = unique_nids.numel()
    _, argmax = scatter_max(ts, compact_idx, dim=0, dim_size=M)
    return argmax


def keep_last(data: Mem_or_Mail) -> Mem_or_Mail:
    assert data.ts is not None, "TS must be provided for keep_last"
    indices = compute_last_indices(data.nid, data.ts)

    if indices.numel() < data.nid.numel():
        data.nid = data.nid[indices]
        data.data = data.data[indices]
        data.ts = data.ts[indices]

    # compute_last_indices 依赖 unique 的组顺序（默认按 nid 排序）
    data.is_sorted = True
    data.is_unique = True
    return data


def keep_mean(data: Mem_or_Mail) -> Mem_or_Mail:
    """
    无 ct 版本：
    - 对重复 nid：对 data、ts 做普通均值（权重恒为 1）
    - 原地修改
    """
    assert data.ts is not None, "TS must be provided for keep_mean"
    if data.nid.numel() == 0:
        data.is_sorted = True
        data.is_unique = True
        return data

    unique_nids, compact_idx = torch.unique(data.nid, return_inverse=True)
    M = unique_nids.numel()

    # 每组计数
    ones = torch.ones_like(data.ts, dtype=torch.float32)
    ct = scatter_sum(ones, compact_idx, dim=0, dim_size=M)  # [M]

    # data 求和（建议转 float32 更稳）
    sum_data = scatter_sum(data.data.to(torch.float32), compact_idx, dim=0, dim_size=M)

    # ts 求和
    sum_ts = scatter_sum(data.ts.to(torch.float32), compact_idx, dim=0, dim_size=M)

    new_data = sum_data / ct.unsqueeze(-1)
    new_ts = sum_ts / ct

    data.nid = unique_nids
    data.data = new_data.to(
        dtype=data.data.dtype
    )  # 如你想强制输出 float32，可删掉这一行 cast
    data.ts = new_ts.to(dtype=data.ts.dtype)

    data.is_sorted = True
    data.is_unique = True
    return data


def _ensure_sorted_unique_nid(x: Mem_or_Mail) -> Mem_or_Mail:
    """
    确保 x.nid 升序且去重（无 ct 版本）。
    注意：这里是就地修改 x（和你当前风格一致）。
    """
    if not getattr(x, "is_sorted", False):
        x = x.sort_by_nid()

    if not getattr(x, "is_unique", False):
        if x.nid.numel() <= 1:
            x.is_unique = True
            return x

        keep = torch.empty(x.nid.shape[0], dtype=torch.bool, device=x.nid.device)
        keep[0] = True
        keep[1:] = x.nid[1:] != x.nid[:-1]

        if keep.sum().item() != x.nid.numel():
            x.nid = x.nid[keep]
            x.data = x.data[keep]
            if x.ts is not None:
                x.ts = x.ts[keep]

        x.is_unique = True

    return x


def _mask_not_in_sorted_unique(
    high_nid_sorted_unique: torch.Tensor, low_nid: torch.Tensor
) -> torch.Tensor:
    pos = torch.searchsorted(high_nid_sorted_unique, low_nid)
    in_range = pos < high_nid_sorted_unique.numel()
    hit = torch.zeros_like(in_range, dtype=torch.bool)
    hit[in_range] = high_nid_sorted_unique[pos[in_range]] == low_nid[in_range]
    return ~hit


def merge_with_priority(
    high_priority_data: Mem_or_Mail, low_priority_data: Mem_or_Mail
) -> Mem_or_Mail:
    """
    无 ct 版本（且双方都保证 ts 存在）：
    - 优先保留 high；low 中 nid 若出现在 high 中则丢弃
    """
    if low_priority_data.nid.numel() == 0:
        return type(high_priority_data)(
            nid=high_priority_data.nid,
            data=high_priority_data.data,
            ts=high_priority_data.ts,
            is_sorted=high_priority_data.is_sorted,
            is_unique=high_priority_data.is_unique,
        )
    if high_priority_data.nid.numel() == 0:
        return type(low_priority_data)(
            nid=low_priority_data.nid,
            data=low_priority_data.data,
            ts=low_priority_data.ts,
            is_sorted=low_priority_data.is_sorted,
            is_unique=low_priority_data.is_unique,
        )

    # 关键前提：high 必须 sorted+unique（就地处理/或返回新对象取决于你的实现）
    high = _ensure_sorted_unique_nid(high_priority_data)

    # low 不需要排序，直接 membership test
    mask_keep_low = _mask_not_in_sorted_unique(high.nid, low_priority_data.nid)

    ret = type(high)(
        nid=torch.cat([high.nid, low_priority_data.nid[mask_keep_low]], dim=0),
        data=torch.cat([high.data, low_priority_data.data[mask_keep_low]], dim=0),
        ts=torch.cat([high.ts, low_priority_data.ts[mask_keep_low]], dim=0),
        is_sorted=False,
        is_unique=False,
    )
    return ret.sort_by_nid()
