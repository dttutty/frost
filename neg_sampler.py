import logging

import torch

from schemas.datatypes.misc import ExtFullGraph


logger = logging.getLogger(__name__)


class NegLinkSampler:
    def __init__(self, g: ExtFullGraph) -> None:
        self.num_nodes = g.indptr.shape[0] - 1

    def sample(
        self, src_ids: torch.Tensor, src_tss: torch.Tensor, num_neg: int = 1
    ) -> torch.Tensor:
        return torch.randint(0, self.num_nodes, (len(src_ids) * num_neg,))
