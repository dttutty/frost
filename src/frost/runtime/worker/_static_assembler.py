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

from ...datatypes.n_feat_nid import NodeFeatNid, NodeFeatureLookup

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
    ) -> "NodeFeatureLookup":
        """Build sorted fetched/cache lookup sources for the current batch.

        Uses the ``StaticPlan`` produced by the host planner:

        * If ``plan.fetch`` is empty, all required features are already in the
          ``nfeat_delta`` cache from the previous batch.
        * Otherwise, fetches the missing rows from ``shared_nfeat``, merges
          them with the cached delta, and updates the cache.

        Args:
            nfeat_plan: Plan specifying which NIDs to fetch vs. cache.
        Returns:
            Separate sorted fetched and cached tables whose union contains all
            features needed for this batch.
        """
        lookup = self.prepare_nfeat_lookup(nfeat_plan, staged_nfeat)
        self.update_nfeat_cache(lookup, nfeat_plan.to_cache.data)
        return lookup

    def prepare_nfeat_lookup(
        self,
        nfeat_plan: "StaticPlan",
        staged_nfeat: torch.Tensor | None = None,
    ) -> "NodeFeatureLookup":
        """Fetch current rows and pair them with the preceding batch's cache."""
        assert self.shared_nfeat is not None
        fetch = nfeat_plan.fetch.data
        previous_cache = self.nfeat_delta

        # Fast path: All required features are already cached
        if fetch.numel() == 0:
            assert previous_cache is not None, (
                "NFeat delta should not be None when fetch is empty"
            )
            fetched = None

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
                # Keep H2D as a bulk same-dtype DMA, then cast on the GPU.
                # Combining the device and dtype conversion makes CUDA read
                # scattered host elements through a conversion kernel and is
                # substantially slower for MAG's wide float16 rows. The copy
                # remains synchronous so the reusable staging slot is safe to
                # return immediately after this method completes.
                gpu_data = staged_nfeat.to(
                    device=self.device,
                    non_blocking=False,
                )
                if gpu_data.dtype != torch.float32:
                    gpu_data = gpu_data.to(dtype=torch.float32)

            # StaticPlan.fetch is a TensorSet, hence ascending and unique; the
            # device transfer preserves that order.
            fetched = NodeFeatNid(nid=gpu_nid, nfeat=gpu_data, is_sorted=True)

        return NodeFeatureLookup(fetched=fetched, cached=previous_cache)

    def update_nfeat_cache(
        self,
        lookup: NodeFeatureLookup,
        to_cache: torch.Tensor,
    ) -> None:
        """Assemble the next cache after the current batch becomes runnable."""
        self.nfeat_delta = (
            self._select_nfeat_rows(lookup, to_cache)
            if to_cache.numel() > 0
            else None
        )

    def _select_nfeat_rows(
        self,
        lookup: NodeFeatureLookup,
        requested_nids: torch.Tensor,
    ) -> NodeFeatNid:
        """Select sorted cache rows without merging wide feature matrices."""
        sources = lookup.sources
        if not sources:
            raise ValueError("Cannot select node features from an empty lookup")

        feat_dim = sources[0].nfeat.size(-1)
        selected = torch.empty(
            (requested_nids.numel(), feat_dim),
            device=self.device,
            dtype=sources[0].nfeat.dtype,
        )
        found_any = torch.zeros_like(requested_nids, dtype=torch.bool)

        for source in sources:
            positions = torch.searchsorted(source.nid, requested_nids)
            in_bounds = positions < source.nid.numel()
            safe_positions = positions.clamp_max(source.nid.numel() - 1)
            found = in_bounds & (source.nid[safe_positions] == requested_nids)
            selected[found] = source.nfeat[safe_positions[found]]
            found_any |= found

        if requested_nids.device.type == "cpu" and not torch.all(found_any):
            missing = requested_nids[~found_any]
            raise ValueError(
                f"Cache plan requested {missing.numel()} unavailable node features"
            )

        return NodeFeatNid(
            nid=requested_nids,
            nfeat=selected,
            is_sorted=True,
        )

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
        nfeat_lookup: "NodeFeatureLookup",
        *,
        strict: bool = True,
    ) -> FlatGraph:
        """Scatter node features into ``flat_graph.nfeat`` using binary search.

        Assigns feature vectors from the fetched/cache lookup sources to every
        valid (non-``-1``) NID in ``flat_graph.nid``.  When ``strict`` is
        ``False``, only positions present in either source are filled and the
        rest stay zeroed.

        Args:
            flat_graph: The FlatGraph to annotate; modified in-place.
            nfeat_lookup: Sorted fetched/cache tables mapping NID to features.
            strict: Whether every valid node in ``flat_graph`` must exist in the
                lookup table. Root-scoped feature plans use ``False`` so only roots are
                filled while neighbor rows remain zeroed.

        Returns:
            The same ``flat_graph`` with ``nfeat`` populated.
        """
        assert self.shared_nfeat is not None
        sources = nfeat_lookup.sources
        if not sources:
            raise ValueError("Node feature lookup is empty")

        # TensorSet and NodeFeatNid establish this invariant before the lookup
        # reaches a worker.  Revalidate CPU callers for useful unit-test errors,
        # but do not turn the assertion into a CUDA-to-host synchronization in
        # the per-batch hot path.
        for source in sources:
            assert source.is_sorted
            assert source.nid.numel() > 0, "node feature source is empty"
            if source.nid.device.type == "cpu":
                assert torch.all(source.nid[1:] > source.nid[:-1]), (
                    "node feature source IDs must be strictly increasing"
                )

        nodes = flat_graph.nid
        valid_mask = nodes != -1
        valid_nodes = nodes[valid_mask]

        if valid_nodes.numel() == 0:
            raise ValueError("No valid nodes found in flat_graph; all NIDs are -1.")

        feat_dim = sources[0].nfeat.size(-1)
        node_feat = torch.zeros(
            (*nodes.shape, feat_dim),
            device=self.device,
            dtype=sources[0].nfeat.dtype,
        )

        valid_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        found_any = torch.zeros_like(valid_nodes, dtype=torch.bool)

        for source in sources:
            positions = torch.searchsorted(source.nid, valid_nodes)
            in_bounds = positions < source.nid.numel()
            safe_positions = positions.clamp_max(source.nid.numel() - 1)
            found = in_bounds & (source.nid[safe_positions] == valid_nodes)
            node_feat[valid_positions[found]] = source.nfeat[safe_positions[found]]
            found_any |= found

        if strict:
            # The host access planner guarantees complete coverage for
            # ALL_SAMPLED.  Preserve detailed validation for CPU callers while
            # avoiding a blocking ``torch.all`` on every CUDA batch.
            if nodes.device.type == "cpu" and not torch.all(found_any):
                missing_nids = torch.unique(valid_nodes[~found_any])
                preview = missing_nids[:20].tolist()
                raise ValueError(
                    f"Found {missing_nids.numel()} NIDs in flat_graph that are missing in the node feature lookup. "
                    f"First {len(preview)} missing NIDs: {preview}"
                )

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
