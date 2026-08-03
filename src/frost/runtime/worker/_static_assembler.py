"""Static feature assembler for the H2D pipeline.

Manages the incremental transfer of node and edge features from shared CPU
memory to GPU.  Node features are cached across batches via a rolling delta
(``nfeat_delta``) so that only newly required features are fetched each step.
"""

import logging

import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.plan import StaticPlan
from frost.models.contracts import NodeSet, RootExpansion

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
        staged_nfeat: torch.Tensor | None = None,
    ) -> "NodeFeatNid":
        """Build a sorted ``NodeFeatNid`` lookup table for the current batch.

        Uses the ``StaticPlan`` produced by the host planner:

        * If ``plan.fetch`` is empty, all required features are already in the
          ``nfeat_delta`` cache from the previous batch.
        * Otherwise, fetches the missing rows from ``shared_nfeat``, merges
          them with the cached delta, and updates the cache.

        Args:
            nfeat_plan: Plan specifying which NIDs to fetch vs. cache.
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
            if staged_nfeat is None:
                feature_ids = fetch.to(self.shared_nfeat.device)
                gpu_data = self.shared_nfeat[feature_ids].to(
                    device=self.device, dtype=torch.float32
                )
            else:
                if staged_nfeat.shape[0] != fetch.numel():
                    raise ValueError(
                        "staged node-feature rows do not match the access plan: "
                        f"{staged_nfeat.shape[0]} != {fetch.numel()}"
                    )
                # This copy is deliberately synchronous: the reusable staging
                # slot can be returned as soon as H2D finishes this batch.
                gpu_data = staged_nfeat.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=False,
                )

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

    def fill_staged_efeat(
        self,
        flat_graph: FlatGraph,
        *,
        staged_efeat: torch.Tensor,
        positions: torch.Tensor,
    ) -> FlatGraph:
        """Scatter packed CPU-staged edge rows into a GPU flat graph."""
        if staged_efeat.shape[0] != positions.numel():
            raise ValueError(
                "staged edge-feature rows do not match their positions: "
                f"{staged_efeat.shape[0]} != {positions.numel()}"
            )
        edge_feat = torch.zeros(
            (flat_graph.eid.shape[0], staged_efeat.shape[1]),
            device=self.device,
            dtype=torch.float32,
        )
        gpu_positions = positions.to(self.device, non_blocking=False)
        gpu_values = staged_efeat.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=False,
        )
        edge_feat[gpu_positions] = gpu_values
        flat_graph.efeat = edge_feat
        return flat_graph

    def fill_nfeat(
        self,
        flat_graph: FlatGraph,
        nfeat_nid: "NodeFeatNid",
        *,
        strict: bool = True,
    ) -> FlatGraph:
        """Scatter node features into ``flat_graph.nfeat`` using binary search.

        Assigns the feature vector from ``nfeat_nid`` to every valid (non-``-1``)
        NID in ``flat_graph.nid``.  When ``strict`` is ``False``, only positions
        present in the lookup table are filled and the rest stay zeroed.

        Args:
            flat_graph: The FlatGraph to annotate; modified in-place.
            nfeat_nid: Sorted lookup table mapping NID → feature vector.
            strict: Whether every valid node in ``flat_graph`` must exist in the
                lookup table. Root-scoped feature plans use ``False`` so only roots are
                filled while neighbor rows remain zeroed.

        Returns:
            The same ``flat_graph`` with ``nfeat`` populated.
        """
        assert self.shared_nfeat is not None
        assert nfeat_nid is not None
        assert nfeat_nid.is_sorted
        assert nfeat_nid.nid.numel() > 0, "nfeat_nid.nid is empty"

        # TensorSet and NodeFeatNid establish this invariant before the lookup
        # reaches a worker.  Revalidate CPU callers for useful unit-test errors,
        # but do not turn the assertion into a CUDA-to-host synchronization in
        # the per-batch hot path.
        if nfeat_nid.nid.device.type == "cpu":
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
        valid_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        # Prevent out-of-bounds access before equality check
        in_bounds = permute_nidx < nfeat_nid.nid.numel()
        safe_idx = permute_nidx.clamp_max(nfeat_nid.nid.numel() - 1)

        found_mask = in_bounds & (nfeat_nid.nid[safe_idx] == valid_nodes)

        if strict:
            # The host access planner guarantees complete coverage for
            # ALL_SAMPLED.  Preserve detailed validation for CPU callers while
            # avoiding a blocking ``torch.all`` on every CUDA batch.
            if nodes.device.type == "cpu" and not torch.all(found_mask):
                missing_nids = torch.unique(valid_nodes[~found_mask])
                preview = missing_nids[:20].tolist()
                raise ValueError(
                    f"Found {missing_nids.numel()} NIDs in flat_graph that are missing in nfeat_nid.nid. "
                    f"First {len(preview)} missing NIDs: {preview}"
                )
            gathered_nfeat = nfeat_nid.nfeat[safe_idx]
            node_feat[valid_positions] = gathered_nfeat
        else:
            # Empty advanced-index assignments are valid, so this remains
            # entirely device-side even when no planned rows are present.
            gathered_nfeat = nfeat_nid.nfeat[safe_idx[found_mask]]
            node_feat[valid_positions[found_mask]] = gathered_nfeat

        flat_graph.nfeat = node_feat
        return flat_graph

    def fill_resident_nfeat(
        self,
        flat_graph: FlatGraph,
        *,
        n_src: int,
        n_neg: int,
        fanout: list[int],
        root_expansion: RootExpansion,
        scope: NodeSet,
    ) -> FlatGraph:
        """Gather the contract-selected rows from a full GPU feature table."""
        assert self.shared_nfeat is not None
        if self.shared_nfeat.device != self.device:
            raise ValueError("Resident node features must live on the worker GPU.")

        nodes = flat_graph.nid
        selected = torch.zeros_like(nodes, dtype=torch.bool)
        n_root = n_src * (2 + n_neg)
        if scope is NodeSet.ALL_SAMPLED:
            selected[:] = True
        elif scope is NodeSet.ROOTS:
            selected[:n_root] = True
        elif scope is NodeSet.POSITIVE_ROOTS:
            selected[: 2 * n_src] = True
        elif scope is NodeSet.POSITIVE_NEIGHBORS:
            if root_expansion is not RootExpansion.POSITIVE_ROOTS or not fanout:
                raise ValueError(
                    "positive-neighbor features require positive-root sampling"
                )
            selected[n_root : n_root + 2 * n_src * fanout[0]] = True
        elif scope is not NodeSet.NONE:
            raise ValueError(f"Unsupported resident node-feature scope: {scope}")
        selected &= nodes != -1

        node_feat = torch.zeros(
            (nodes.numel(), self.shared_nfeat.shape[1]),
            device=self.device,
            dtype=torch.float32,
        )
        positions = torch.nonzero(selected, as_tuple=False).squeeze(-1)
        if positions.numel() > 0:
            node_feat[positions] = self.shared_nfeat[nodes[positions].long()].to(
                torch.float32
            )
        flat_graph.nfeat = node_feat
        return flat_graph

    def fill_efeat(
        self,
        flat_graph: FlatGraph,
        n_src: int,
    ) -> FlatGraph:
        """Gather edge features from shared CPU storage into ``flat_graph.efeat``.

        Only source-edge positions (EID != ``-1``, excluding dst/neg positions
        ``n_src:3*n_src``) are filled.  Raises ``ValueError`` if no valid edges
        are found or if any EID is out of bounds.

        Args:
            flat_graph: The FlatGraph to annotate; modified in-place.
            n_src: Number of source nodes in this batch.
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
            (*edges.shape, feat_dim), device=self.device, dtype=torch.float32
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

        feature_ids = valid_eids.to(efeat.device)
        gathered_efeat = efeat[feature_ids]

        # Transfer to device
        gpu_valid_efeat = gathered_efeat.to(device=self.device, dtype=torch.float32)

        # Assign features to valid positions
        valid_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        edge_feat[valid_positions.to(self.device)] = gpu_valid_efeat

        flat_graph.efeat = edge_feat

        # Return updated ZipGraph
        return flat_graph
