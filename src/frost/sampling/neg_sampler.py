import logging

import torch

logger = logging.getLogger(__name__)


class NegLinkSampler:
    def __init__(self, n_nodes: int, n_neg: int) -> None:
        self.n_nodes = n_nodes  # from hydra config
        self.n_neg = n_neg

    def sample(self, src_ids: torch.Tensor) -> torch.Tensor:
        num_neg = self.n_neg
        return torch.randint(0, self.n_nodes, (len(src_ids) * num_neg,))
