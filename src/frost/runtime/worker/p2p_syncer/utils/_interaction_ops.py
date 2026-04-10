import logging

import torch

from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry

logger = logging.getLogger(__name__)


def generate_interaction_mails(
    root_new_mem_orig: MemEntry,
    edge_feats: torch.Tensor | None = None,
    peer_mem: torch.Tensor | None = None,
) -> MailEntry:
    """
    Generate interaction mails for each node from updated memories and edge features.
    Memory and edges must correspond one-to-one.

    Args:
        root_new_mem_orig: MemEntry with 2N nodes — first N are Src, last N are Dst.
                           mem shape: [2N, mem_dim]
        edge_feats:        Edge features of shape [N, edge_dim],
                           where edge_feats[i] corresponds to the edge Src[i] -> Dst[i].
        peer_mem:          Optional peer embeddings in the same root layout.  When
                           provided, mail still uses ``root_new_mem_orig.data`` for
                           the self half but uses ``peer_mem`` for the other-node
                           half.  This lets DyRep write graph-attention peer
                           embeddings without changing mailbox width.
    Returns:
        MailEntry of shape [2N, 2 * mem_dim + edge_dim],
        where entry i is the mail for node root_new_mem_orig.nid[i].
    """
    # 1. Derive N
    total_nodes = root_new_mem_orig.nid.shape[0]
    if total_nodes % 2 != 0:
        raise ValueError(
            f"root_new_mem_orig must contain an even [src | dst] layout, got {total_nodes}"
        )
    n = total_nodes // 2

    # 2. Split Src and Dst memories
    #    root_new_mem_orig.data layout: [Src_Mem... | Dst_Mem...]
    src_mem = root_new_mem_orig.data[:n]  # self memory for Src nodes
    dst_mem = root_new_mem_orig.data[n:]  # self memory for Dst nodes

    peer_data = root_new_mem_orig.data if peer_mem is None else peer_mem
    if peer_data.ndim != root_new_mem_orig.data.ndim:
        raise ValueError(
            "peer_mem must have the same rank as memory data: "
            f"got {peer_data.ndim}, need {root_new_mem_orig.data.ndim}"
        )
    if peer_data.shape[0] < total_nodes:
        raise ValueError(
            "peer_mem must contain at least the positive [src | dst] roots: "
            f"got {peer_data.shape[0]}, need {total_nodes}"
        )
    if peer_data.shape[1] != root_new_mem_orig.data.shape[1]:
        raise ValueError(
            "peer_mem feature dim must match memory dim for mailbox width: "
            f"got {peer_data.shape[1]}, need {root_new_mem_orig.data.shape[1]}"
        )
    peer_data = peer_data[:total_nodes]
    src_peer = peer_data[:n]
    dst_peer = peer_data[n:]

    # 3. Mail for Src nodes: [self, other, edge] = [Src_Mem, Dst_Peer, Edge]
    if edge_feats is not None:
        mail_for_src = torch.cat([src_mem, dst_peer, edge_feats], dim=1)

        # 4. Mail for Dst nodes: [self, other, edge] = [Dst_Mem, Src_Peer, Edge]
        mail_for_dst = torch.cat([dst_mem, src_peer, edge_feats], dim=1)
    else:
        mail_for_src = torch.cat([src_mem, dst_peer], dim=1)

        # 4. Mail for Dst nodes: [self, other] = [Dst_Mem, Src_Peer]
        mail_for_dst = torch.cat([dst_mem, src_peer], dim=1)

    # 5. Concatenate back to 2N length; first N entries are Src mails, last N are Dst mails
    all_mails = torch.cat([mail_for_src, mail_for_dst], dim=0)

    mail_ts_nid = MailEntry(
        nid=root_new_mem_orig.nid,
        data=all_mails,
        ts=root_new_mem_orig.ts,  # ts length equals 2N
    )
    return mail_ts_nid
