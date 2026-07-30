"""Route positive-root mail payloads to sampled one-hop neighbors."""

import torch

from frost.datatypes.graph import HierGraph
from frost.datatypes.mem_mail_box.mail import MailEntry


class NeighborhoodMailboxRouter:
    def deliver(self, graph: HierGraph, mail_data: torch.Tensor) -> MailEntry:
        """Broadcast each positive root's mail to its valid sampled neighbors."""
        if not graph.neighbors:
            raise ValueError("HierGraph must have at least one neighbor layer.")

        n_total = graph.root_nid.numel()
        target_len = (n_total // (2 + graph.n_neg)) * 2
        if mail_data.size(0) == n_total:
            mail_data = mail_data[:target_len]
        elif mail_data.size(0) != target_len:
            raise ValueError(
                f"mail_data length {mail_data.size(0)} does not match "
                f"positive-root length {target_len}."
            )

        if not graph.fanout:
            raise ValueError("Neighborhood mail routing requires one-hop fanout.")
        fanout = graph.fanout[0]
        neighbor_nid = graph.neighbors[0].nid.view(target_len, fanout)
        valid = neighbor_nid != -1
        width = mail_data.size(-1)
        return MailEntry(
            nid=neighbor_nid[valid],
            data=mail_data.unsqueeze(1).expand(target_len, fanout, width)[valid],
            ts=graph.root_ts[:target_len]
            .unsqueeze(1)
            .expand(target_len, fanout)[valid],
            is_sorted=False,
            is_unique=False,
        )
