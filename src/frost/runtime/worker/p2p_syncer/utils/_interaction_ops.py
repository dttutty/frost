import logging

import torch

from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry

logger = logging.getLogger(__name__)


def generate_interaction_mails(
    root_new_mem_orig: MemEntry,
    edge_feats: torch.Tensor | None = None,
) -> MailEntry:
    """
    Generate interaction mails for each node from updated memories and edge features.
    Memory and edges must correspond one-to-one.

    Args:
        root_new_mem_orig: MemEntry with 2N nodes — first N are Src, last N are Dst.
                           mem shape: [2N, mem_dim]
        edge_feats:        Edge features of shape [N, edge_dim],
                           where edge_feats[i] corresponds to the edge Src[i] -> Dst[i].
    Returns:
        MailEntry of shape [2N, 2 * mem_dim + edge_dim],
        where entry i is the mail for node root_new_mem_orig.nid[i].
    """
    # 1. Derive N
    total_nodes = root_new_mem_orig.nid.shape[0]
    if root_new_mem_orig.data.shape[0] != total_nodes:
        raise ValueError(
            "root_new_mem_orig nid/data rows must match, got "
            f"nid={tuple(root_new_mem_orig.nid.shape)} "
            f"data={tuple(root_new_mem_orig.data.shape)}"
        )
    if root_new_mem_orig.ts.shape[0] != total_nodes:
        raise ValueError(
            "root_new_mem_orig nid/ts rows must match, got "
            f"nid={tuple(root_new_mem_orig.nid.shape)} "
            f"ts={tuple(root_new_mem_orig.ts.shape)}"
        )
    if total_nodes % 2 != 0:
        raise ValueError(
            f"root_new_mem_orig must contain an even [src | dst] layout, got {total_nodes}"
        )
    n = total_nodes // 2

    if edge_feats is not None and edge_feats.shape[0] != n:
        raise ValueError(
            "edge feature rows must match the positive interaction count, got "
            f"edge_feats={tuple(edge_feats.shape)} expected_rows={n}"
        )

    # 2. Split Src and Dst memories
    #    root_new_mem_orig.data layout: [Src_Mem... | Dst_Mem...]
    src_mem = root_new_mem_orig.data[:n]  # self memory for Src nodes
    dst_mem = root_new_mem_orig.data[n:]  # self memory for Dst nodes

    # 3. Mail for Src nodes: [self, other, edge] = [Src_Mem, Dst_Mem, Edge]
    if edge_feats is not None:
        mail_for_src = torch.cat([src_mem, dst_mem, edge_feats], dim=1)

        # 4. Mail for Dst nodes: [self, other, edge] = [Dst_Mem, Src_Peer, Edge]
        mail_for_dst = torch.cat([dst_mem, src_mem, edge_feats], dim=1)
    else:
        mail_for_src = torch.cat([src_mem, dst_mem], dim=1)

        # 4. Mail for Dst nodes: [self, other] = [Dst_Mem, Src_Peer]
        mail_for_dst = torch.cat([dst_mem, src_mem], dim=1)

    # 5. Concatenate back to 2N length; first N entries are Src mails, last N are Dst mails
    all_mails = torch.cat([mail_for_src, mail_for_dst], dim=0)

    mail_ts_nid = MailEntry(
        nid=root_new_mem_orig.nid,
        data=all_mails,
        ts=root_new_mem_orig.ts,  # ts length equals 2N
    )
    return mail_ts_nid
