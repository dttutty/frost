import logging

import torch

from frost.datatypes.access import Access
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


class APANAnalyzer:
    """Access analyzer for the APAN model (1-hop, sample_4_neg=False).

    APAN samples neighbors only for source and positive-destination nodes, not
    for negative destinations.  This analyzer correctly identifies which nodes
    need memory/mailbox/nfeat access by following the same indexing convention as
    ``Postman.deliver()`` and ``merge_1_mail()``.

    Attributes:
        n_neg: Number of negative samples per positive edge, from the Hydra config.
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg  # from hydra config

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ):
        """Compute memory, mailbox, and nfeat access sets for one APAN batch.

        Only src/dst roots update memory; only the 1-hop neighbors of src+dst
        update the mailbox (negative destinations and their neighbors are excluded).

        Args:
            src: Source node IDs of shape ``[n_src]``.
            dst: Positive destination node IDs of shape ``[n_src]``.
            all_nid: All unique node IDs appearing in the sampled flat-graph.
            fanout: Neighbor fanout per hop; must be a 1-element list for APAN.
            valid_all: Flat array of sampled node IDs (including padding ``-1``).

        Returns:
            Access: Named sets for required_mem, mem_update, required_box,
            box_update, and required_nfeat.
        """
        assert valid_all is not None, "APAN requires valid_all for access counting"
        assert fanout is not None, "APAN requires fanout for neighbor sampling"
        assert len(fanout) == 1, "APAN only supports 1-hop sampling"
        n_src = src.shape[0]
        # Root layout: [src(n_src), pos_dst(n_src), neg_dst(n_neg*n_src)]
        # Neighbors (sample_4_neg=False): only src + pos_dst have neighbors
        ngh_start = n_src * (2 + self.n_neg)
        pos_ngh_raw = valid_all[ngh_start : (ngh_start + fanout[0] * n_src * 2)]
        pos_ngh = pos_ngh_raw[pos_ngh_raw != -1]  # remove padding slots

        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            required_box=all_nodes_set,
            box_update=TensorSet(pos_ngh),
            required_nfeat=all_nodes_set,
        )


class NoMemAnalyzer:
    """Access analyzer for memory-free models (e.g. TGAT, GraphMixer, DyGFormer).

    Only node features need to be fetched; there is no memory or mailbox state.

    Attributes:
        n_neg: Number of negative samples per positive edge, from the Hydra config.
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg  # from hydra config

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ):
        """Compute nfeat access set for one batch (no memory or mailbox).

        Args:
            src: Source node IDs of shape ``[n_src]``.
            dst: Positive destination node IDs of shape ``[n_src]``.
            all_nid: All unique node IDs in the sampled flat-graph.
            fanout: Neighbor fanout per hop (not used).
            valid_all: Flat array of sampled node IDs (not used).

        Returns:
            Access: Only ``required_nfeat`` is populated; all other sets are None.
        """
        all_nid = all_nid.clone()  # Clone to avoid mutating the caller's tensor
        all_nodes_set = TensorSet(all_nid)

        return Access(
            required_nfeat=all_nodes_set,
        )


class TGNAnalyzer:
    """Access analyzer for TGN (Temporal Graph Network).

    Both memory and mailbox are updated by src/dst roots; all sampled nodes
    (roots + neighbors) require memory, mailbox, and nfeat to be loaded.

    Attributes:
        n_neg: Number of negative samples per positive edge, from the Hydra config.
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg  # from hydra config

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ):
        """Compute memory, mailbox, and nfeat access sets for one TGN batch.

        Args:
            src: Source node IDs of shape ``[n_src]``.
            dst: Positive destination node IDs of shape ``[n_src]``.
            all_nid: All unique node IDs in the sampled flat-graph.
            fanout: Neighbor fanout per hop (not used directly).
            valid_all: Flat array of sampled node IDs (not used directly).

        Returns:
            Access: All five fields populated; mem_update and box_update contain
            only src+dst (positive roots).
        """
        all_nid = all_nid.clone()  # Clone to avoid mutating the caller's tensor
        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            required_box=all_nodes_set,
            box_update=pos_root,
            required_nfeat=all_nodes_set,
        )


class JODIEAnalyzer:
    """Access analyzer for JODIE (0-hop memory model).

    Similar to TGNAnalyzer: all sampled nodes require memory, mailbox, and nfeat;
    memory and mailbox are updated only by positive src/dst pairs.

    Attributes:
        n_neg: Number of negative samples per positive edge, from the Hydra config.
    """

    def __init__(self, n_neg: int):
        self.n_neg = n_neg  # from hydra config

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
        valid_all: torch.Tensor | None = None,
    ):
        """Compute memory, mailbox, and nfeat access sets for one JODIE batch.

        Args:
            src: Source node IDs of shape ``[n_src]``.
            dst: Positive destination node IDs of shape ``[n_src]``.
            all_nid: All unique node IDs in the sampled flat-graph (src, dst,
                neg_dst).
            fanout: Neighbor fanout per hop (not used; JODIE uses 0-hop sampling).
            valid_all: Flat array of sampled node IDs (not used).

        Returns:
            Access: All five fields populated; mem_update and box_update contain
            only src+dst (positive roots).
        """
        all_nid = all_nid.clone()  # Clone to avoid mutating the caller's tensor
        pos_root = TensorSet(torch.cat([src, dst], dim=0))
        all_nodes_set = TensorSet(all_nid)  # this includes src, dst, and neg_dst

        return Access(
            required_mem=all_nodes_set,
            mem_update=pos_root,
            required_box=all_nodes_set,
            box_update=pos_root,
            required_nfeat=all_nodes_set,
        )
