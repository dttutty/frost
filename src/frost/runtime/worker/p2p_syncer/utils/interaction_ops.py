import logging
from typing import TypeVar

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry

AllType = TypeVar("AllType", MemEntry, MailEntry, BoxEntry)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


def generate_interaction_mails(
    root_new_mem_orig: MemEntry, edge_feats: torch.Tensor | None = None
) -> MailEntry:
    """
    这里mem必须和edge一一对应
    Args:
        root_new_mem_orig: MemTsNid 对象 (包含 2N 个节点)
                           前 N 个是 Src，后 N 个是 Dst。
                           mem 形状: [2N, mem_dim]
        edge_feats:        边的特征，形状 [N, edge_dim]
                           对应着 Src[i] -> Dst[i] 的边
    Returns:
        all_mails:         形状 [2N, 2 * mem_dim + edge_dim]
                           对应 root_new_mem_orig 中每个节点的 mail
    """
    # 1. 获取 N 的大小
    total_nodes = root_new_mem_orig.nid.shape[0]
    n = total_nodes // 2

    # 2. 切分 Src 和 Dst 的 Memory
    # root_new_mem_orig.mem 的结构是 [Src_Mem ... | Dst_Mem ...]
    src_mem = root_new_mem_orig.data[:n]  # Self Memory for Src Nodes
    dst_mem = root_new_mem_orig.data[n:]  # Other Memory for Src Nodes (Self for Dst)

    # 3. 生成给 Src 节点的 Mail
    # 逻辑: [自己, 对方, 边] -> [Src, Dst, Edge]
    if edge_feats is not None:
        # 在 interaction_ops.py 第 43 行之前添加：

        mail_for_src = torch.cat([src_mem, dst_mem, edge_feats], dim=1)

        # 4. 生成给 Dst 节点的 Mail
        # 逻辑: [自己, 对方, 边] -> [Dst, Src, Edge]
        # 注意：通常这里要交换 src/dst mem 的位置，保证模型看到的第一部分永远是"自己"
        mail_for_dst = torch.cat([dst_mem, src_mem, edge_feats], dim=1)
    else:
        mail_for_src = torch.cat([src_mem, dst_mem], dim=1)

        # 4. 生成给 Dst 节点的 Mail
        # 逻辑: [自己, 对方, 边] -> [Dst, Src, Edge]
        # 注意：通常这里要交换 src/dst mem 的位置，保证模型看到的第一部分永远是"自己"
        mail_for_dst = torch.cat([dst_mem, src_mem], dim=1)
        # 5. 拼接回 2N 长度，保持顺序一致 (前N个是Src的mail，后N个是Dst的mail)
    all_mails = torch.cat([mail_for_src, mail_for_dst], dim=0)

    mail_ts_nid = MailEntry(
        nid=root_new_mem_orig.nid,
        data=all_mails,
        ts=root_new_mem_orig.ts,  # 这里ts的len正好=2N
    )
    return mail_ts_nid
