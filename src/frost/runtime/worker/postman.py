import logging

import torch

from frost.datatypes.graph import Graph
from frost.datatypes.mem_mail_box.mail import MailEntry

logger = logging.getLogger(__name__)


class Postman:
    def __init__(
        self,
        n_neg: int,
    ):
        self.n_neg = n_neg  # from hydra config

    def deliver(self, graph: "Graph", mail_data: torch.Tensor) -> "MailEntry":
        """
        Deliver mails from the first 2/3 of root nodes to their 1-hop neighbors.

        Args:
            graph: SeqGraph containing the batch structure.
            mail_data: Tensor of shape [2/3 * N, D] or [N, D] representing the mail payload.
        """
        if not graph.neighbors or len(graph.neighbors) == 0:
            raise ValueError("SeqGraph must have at least 1 layer of neighbors.")

        # 1. Calculate the number of target nodes (first 2/3)
        N_total = graph.root_nid.numel()
        target_len = (N_total // (2 + self.n_neg)) * 2

        # Ensure mail_data size matches the target length
        if mail_data.size(0) == N_total:
            mail_data = mail_data[:target_len]
        elif mail_data.size(0) != target_len:
            raise ValueError(
                f"mail_data length {mail_data.size(0)} does not match target_len {target_len}."
            )

        # 2. Extract sender timestamps
        sender_ts = graph.root_ts[:target_len]

        # 3. Reshape neighbor NIDs to [target_len, S]
        # Neighbors are only sampled for src + pos_dst (target_len nodes),
        # so the tensor has target_len * S elements, not N_total * S.
        S = graph.fanout[0]
        nbr_nids = graph.neighbors[0].nid.view(target_len, S)
        target_nbr_nids = nbr_nids  # Shape: [target_len, S]

        # 4. Create a boolean mask to filter out padding nodes (-1)
        valid_mask = target_nbr_nids != -1  # Shape: [target_len, S]

        # 5. Extract valid destination NIDs using the mask (automatically flattens to 1D)
        new_nids = target_nbr_nids[valid_mask]  # Shape: [K]

        # 6. Broadcast mail_data to neighbors
        # [target_len, D] -> [target_len, S, D] -> Apply Mask -> [K, D]
        D = mail_data.size(-1)
        expanded_mail_data = mail_data.unsqueeze(1).expand(target_len, S, D)
        new_mail_data = expanded_mail_data[valid_mask]  # Shape: [K, D]

        # 7. Broadcast sender timestamps to neighbors
        # [target_len] -> [target_len, S] -> Apply Mask -> [K]
        expanded_ts = sender_ts.unsqueeze(1).expand(target_len, S)
        new_ts = expanded_ts[valid_mask]  # Shape: [K]

        # 8. Return the final MailEntry
        return MailEntry(
            nid=new_nids,
            data=new_mail_data,
            ts=new_ts,
            is_sorted=False,
            is_unique=False,
        )
