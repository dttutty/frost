"""Static feature assembler for the H2D pipeline.

Manages the incremental transfer of node and edge features from shared CPU
memory to GPU.  Node features are cached across batches via a rolling delta
(``nfeat_delta``) so that only newly required features are fetched each step.
"""
import logging

import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.meta import Meta
from frost.datatypes.plan import StaticPlan

from ...datatypes.n_feat_nid import NodeFeatNid

logger = logging.getLogger(__name__)


class StaticAssembler:
    """
    Assembles static features (nfeat/efeat) for each batch by merging
    CPU-fetched data with the cached delta from the previous iteration.
    """

    nfeat_delta: NodeFeatNid | None = None

    def __init__(
        self,
        shared_nfeat: torch.Tensor | None,
        shared_efeat: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        self.nfeat_delta = None
        self.device = device
        self.shared_nfeat = shared_nfeat
        self.shared_efeat = shared_efeat

    def prepare_nfeat_cache(
        self,
        nfeat_plan: "StaticPlan",
        meta: "Meta",
    ) -> "NodeFeatNid":
        """Build a sorted ``NodeFeatNid`` lookup table for the current batch.

        Uses the ``StaticPlan`` produced by the host planner:

        * If ``plan.fetch`` is empty, all required features are already in the
          ``nfeat_delta`` cache from the previous batch.
        * Otherwise, fetches the missing rows from ``shared_nfeat``, merges
          them with the cached delta, and updates the cache.

        Args:
            nfeat_plan: Plan specifying which NIDs to fetch vs. cache.
            meta: Batch metadata (used for debugging context).

        Returns:
            Sorted ``NodeFeatNid`` containing all features needed for this batch.
        """
        assert self.shared_nfeat is not None
        to_cache, fetch = nfeat_plan.to_cache.data, nfeat_plan.fetch.data

        # Fast path: All required features are already cached
        if fetch.numel() == 0:
            nfeat_obj = self.nfeat_delta
            assert nfeat_obj is not None, (
                "NFeat delta should not be None when fetch is empty"
            )

            nfeat_obj = nfeat_obj.sort_by_nid()

            # Update cache delta
            if to_cache.numel() > 0:
                self.nfeat_delta = nfeat_obj.filter_by_nid(to_cache)
            else:
                logger.info("⚠️ [DEBUG] No new NIDs to cache, clearing nfeat_delta")
                self.nfeat_delta = None

        else:
            # Standard path: Gather missing features directly from shared memory
            gpu_nid = fetch.to(self.device)
            gpu_data = self.shared_nfeat[fetch].to(self.device)

            nfeat_obj = NodeFeatNid(nid=gpu_nid, nfeat=gpu_data)

            # Merge with existing cached delta if available
            if self.nfeat_delta is not None:
                nfeat_obj = nfeat_obj.merge(self.nfeat_delta)

            nfeat_obj = nfeat_obj.sort_by_nid()

            # Update cache delta for the next iteration
            if to_cache.numel() > 0:
                self.nfeat_delta = nfeat_obj.filter_by_nid(to_cache)
            else:
                self.nfeat_delta = None

        return nfeat_obj

    def fill_nfeat(
        self,
        flat_graph: FlatGraph,
        nfeat_nid: "NodeFeatNid",
        meta: "Meta",
    ) -> FlatGraph:
        """Scatter node features into ``flat_graph.nfeat`` using binary search.

        Assigns the feature vector from ``nfeat_nid`` to every valid (non-``-1``)
        NID in ``flat_graph.nid``.  Raises ``ValueError`` if any NID is absent
        from the lookup table.

        Args:
            flat_graph: The FlatGraph to annotate; modified in-place.
            nfeat_nid: Sorted lookup table mapping NID → feature vector.
            meta: Batch metadata (used for debugging context).

        Returns:
            The same ``flat_graph`` with ``nfeat`` populated.
        """
        assert self.shared_nfeat is not None
        assert nfeat_nid is not None
        assert nfeat_nid.is_sorted
        assert nfeat_nid.nid.numel() > 0, "nfeat_nid.nid is empty"

        # Optional but recommended for strictness:
        # disallow duplicate NIDs in the lookup table.
        assert torch.all(nfeat_nid.nid[1:] > nfeat_nid.nid[:-1]), (
            "nfeat_nid.nid must be strictly increasing (sorted and unique)"
        )

        nodes = flat_graph.nid
        valid_mask = nodes != -1
        valid_nodes = nodes[valid_mask]

        if valid_nodes.numel() == 0:
            raise ValueError("No valid nodes found in flat_graph; all NIDs are -1.")

        feat_dim = nfeat_nid.nfeat.size(-1)
        node_feat = torch.zeros(
            (*nodes.shape, feat_dim),
            device=self.device,
            dtype=nfeat_nid.nfeat.dtype,
        )

        # Binary search on sorted nfeat_nid.nid
        permute_nidx = torch.searchsorted(nfeat_nid.nid, valid_nodes)

        # Prevent out-of-bounds access before equality check
        in_bounds = permute_nidx < nfeat_nid.nid.numel()
        safe_idx = permute_nidx.clamp_max(nfeat_nid.nid.numel() - 1)

        found_mask = in_bounds & (nfeat_nid.nid[safe_idx] == valid_nodes)

        if not torch.all(found_mask):
            missing_nids = torch.unique(valid_nodes[~found_mask])
            preview = missing_nids[:20].tolist()
            raise ValueError(
                f"Found {missing_nids.numel()} NIDs in flat_graph that are missing in nfeat_nid.nid. "
                f"First {len(preview)} missing NIDs: {preview}"
            )

        gathered_nfeat = nfeat_nid.nfeat[permute_nidx]
        node_feat[valid_mask] = gathered_nfeat.to(self.device)

        flat_graph.nfeat = node_feat
        return flat_graph

    def fill_efeat(
        self,
        flat_graph: FlatGraph,
        n_src: int,
        meta: "Meta",
    ) -> FlatGraph:
        """Gather edge features from shared CPU storage into ``flat_graph.efeat``.

        Only source-edge positions (EID != ``-1``, excluding dst/neg positions
        ``n_src:3*n_src``) are filled.  Raises ``ValueError`` if no valid edges
        are found or if any EID is out of bounds.

        Args:
            flat_graph: The FlatGraph to annotate; modified in-place.
            n_src: Number of source nodes in this batch.
            meta: Batch metadata (used for debugging context).

        Returns:
            The same ``flat_graph`` with ``efeat`` populated.
        """
        # Handle efeat reuse for all versions
        assert self.shared_efeat is not None
        # Assign shared_efeat to a local variable for clarity
        efeat = self.shared_efeat

        # Get edges and valid mask
        edges = flat_graph.eid
        valid_mask = edges != -1

        valid_mask[n_src : 3 * n_src] = False

        # Extract valid edge IDs
        valid_eids = edges[valid_mask]
        num_valid = valid_eids.numel()
        feat_dim = efeat.size(-1)

        # Init zero tensor for edge features on target device
        edge_feat = torch.zeros(
            (*edges.shape, feat_dim), device=self.device, dtype=efeat.dtype
        )

        # Return early if no valid edges
        if num_valid == 0:
            raise ValueError("No valid edges found in flat_graph to fill efeat!")

        # ---------- Check EID bounds ----------

        max_eid = valid_eids.max().item()
        min_eid = valid_eids.min().item()
        max_capacity = efeat.shape[0]

        assert max_eid < max_capacity, (
            f"EID OUT OF BOUNDS! Max EID {max_eid} exceeds efeat capacity {max_capacity}"
        )
        assert min_eid >= 0, f"EID OUT OF BOUNDS! Min EID {min_eid} is negative"

        # ---------------------------------------------

        # Gather features on CPU
        cpu_efeat = efeat[valid_eids]

        # Transfer to device
        gpu_valid_efeat = cpu_efeat.to(self.device)

        # Assign features to valid positions
        edge_feat[valid_mask] = gpu_valid_efeat

        flat_graph.efeat = edge_feat

        # Return updated ZipGraph
        return flat_graph
