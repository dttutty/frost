"""Assembled model that composes memory updater, encoder, postman, and header.

This module wires together all model components (memory updater, GNN encoder,
optional APAN postman, and link predictor header) into a single forward pass
used by the worker during training and evaluation.
"""

from collections.abc import Callable
import logging
from queue import Queue
from typing import cast

import torch
import torch.nn as nn

from frost.ablation import DynamicStateAblation

# Import the newly unified memory updaters
from frost.datatypes.mem_mail_box._data_ops import keep_last, mail_to_box
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.meta import Meta
from frost.datatypes.tasks import BoxTask, DirectWritebackRequest, MemTask
from frost.runtime.protocols import CudaEvent
from frost.models.protocols import MemoryUpdaterProto
from frost.models.contracts import RepresentationKind, RootExpansion
from frost.models.outputs import EncoderResult, RepresentationAdapter
from frost.models.protocols import PostmanProto
from frost.runtime.worker.p2p_syncer.utils._interaction_ops import (
    generate_interaction_mails,
)
from frost.utils.tensor_set import isin_sorted

logger = logging.getLogger(__name__)


class AssembledModel(nn.Module):
    """Full model graph for a single training or evaluation step.

    Chains the following components in order:
      1. Memory updater (optional): updates node memories via RNN/Transformer.
      2. FlatGraph unflattening: converts flat tensor layout to a hierarchical
         HierGraph.
      3. Encoder: computes root-node embeddings from the hierarchical graph.
      4. Header: produces link-prediction logits from the embeddings.

    When a memory updater is present, the updated memory and mailbox states are
    asynchronously pushed back to shared memory via ``mem_task_q`` and
    ``box_task_q``.

    Args:
        encoder: GNN encoder module (e.g., TGN, TGAT, DyGFormer).
        mem_updater: Optional memory updater (MemoryUpdater or
            TransformerMemoryUpdater).  Pass ``None`` for memory-free models.
        header: Prediction head over structured edge representations.
        postman: Optional APAN postman for mailbox delivery.  Pass ``None`` for
            non-APAN models.
        n_neg: Number of negative samples per positive edge.
        root_expansion: Semantic root set expanded into neighborhoods.
    """

    def __init__(
        self,
        encoder: nn.Module,
        mem_updater: MemoryUpdaterProto | None,
        header: nn.Module,
        postman: PostmanProto | None,
        n_neg: int,
        representation: RepresentationKind,
        root_expansion: RootExpansion = RootExpansion.ALL_ROOTS,
        dynamic_state_ablation: DynamicStateAblation | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mem_updater = mem_updater
        self.header = header
        self.representation_adapter = RepresentationAdapter(representation)
        self.postman = postman
        self.dynamic_state_ablation = (
            dynamic_state_ablation
            if dynamic_state_ablation is not None
            else DynamicStateAblation(name="baseline")
        )

        self.n_neg = n_neg  # from hydra config
        self.root_expansion = root_expansion

    @staticmethod
    def _union_planned_p2p_nids(plan) -> torch.Tensor:
        if plan is None or len(plan.p2p) == 0:
            return torch.empty(0, dtype=torch.long)

        if len(plan.p2p) == 1:
            return plan.p2p[0].data

        # Every TensorSet is already sorted and unique.  Concatenate all peers
        # first and deduplicate once instead of rebuilding a sorted TensorSet
        # after each pairwise union.
        return torch.unique(
            torch.cat([per_dst.data for per_dst in plan.p2p]),
            sorted=True,
        )

    @staticmethod
    def _dedupe_box_entry(entry: BoxEntry) -> BoxEntry:
        if entry.nid.numel() == 0:
            entry.is_sorted = True
            entry.is_unique = True
            return entry

        entry = entry.sort_by_nid()
        if entry.is_unique:
            return entry

        keep = torch.ones(entry.nid.numel(), dtype=torch.bool, device=entry.nid.device)
        if entry.nid.numel() > 1:
            keep[:-1] = entry.nid[:-1] != entry.nid[1:]

        if keep.sum().item() != entry.nid.numel():
            entry = BoxEntry(
                nid=entry.nid[keep],
                data=entry.data[keep],
                ts=entry.ts[keep],
                is_sorted=True,
                is_unique=True,
            )
        else:
            entry.is_unique = True

        return entry

    def _select_stale_mem_source(
        self,
        meta: Meta,
        planned_send_nids: torch.Tensor,
    ) -> MemEntry | None:
        flat_graph = meta.flat_graph
        if (
            flat_graph is None
            or flat_graph.mem is None
            or flat_graph.mem_ts is None
            or planned_send_nids.numel() == 0
        ):
            return None

        valid_mask = (flat_graph.nid != -1) & isin_sorted(
            flat_graph.nid,
            planned_send_nids,
        )
        if not torch.any(valid_mask):
            return None

        stale_mem = MemEntry(
            nid=flat_graph.nid[valid_mask],
            data=flat_graph.mem[valid_mask],
            ts=flat_graph.mem_ts[valid_mask],
        )
        return keep_last(stale_mem)

    def _select_stale_box_source(
        self,
        meta: Meta,
        planned_send_nids: torch.Tensor,
    ) -> BoxEntry | None:
        flat_graph = meta.flat_graph
        if (
            flat_graph is None
            or flat_graph.box is None
            or flat_graph.box_ts is None
            or planned_send_nids.numel() == 0
        ):
            return None

        valid_mask = (flat_graph.nid != -1) & isin_sorted(
            flat_graph.nid,
            planned_send_nids,
        )
        if not torch.any(valid_mask):
            return None

        stale_box = BoxEntry(
            nid=flat_graph.nid[valid_mask],
            data=flat_graph.box[valid_mask],
            ts=flat_graph.box_ts[valid_mask],
        )
        return self._dedupe_box_entry(stale_box)

    def forward(
        self,
        meta: Meta,
        *,
        mem_task_q: Queue | None = None,
        box_task_q: Queue | None = None,
        direct_writeback: Callable[[DirectWritebackRequest], None] | None = None,
    ) -> torch.Tensor:
        """Run a full forward pass for one batch.

        Steps:
          1. If a memory updater is configured, update node memories in
             ``meta.flat_graph`` and enqueue ``MemTask`` / ``BoxTask`` objects
             for async global synchronization.
          2. Inflate ``meta.flat_graph`` into the hierarchical ``meta.graph``.
          3. Run the encoder to obtain root-node embeddings.
          4. Run the header to obtain link-prediction logits.

        Args:
            meta: Batch metadata carrier containing the flat graph, plan set,
                and other per-batch attributes.
            mem_task_q: Queue for publishing updated node memories to the
                global memory store.  Required when ``mem_updater`` is set.
            box_task_q: Queue for publishing updated mailbox entries.
                Required when ``mem_updater`` is set.

        Returns:
            Logit tensor of shape ``[n_src * (1 + n_neg), 1]`` containing
            concatenated positive and negative edge scores.
        """
        logger.debug("Processing meta with meta.batch_idx=%s", meta.mini_batch_idx)

        # 1. Update memory using the flattened FlatGraph

        n_src = meta.n_src
        fanout = meta.fanout if meta.fanout is not None else []
        assert meta.flat_graph is not None
        meta.direct_writeback_request = None

        if self.mem_updater is not None:
            plan_set = meta.plan_set
            if self.dynamic_state_ablation.uses_async_dynamic_state_pipeline:
                assert mem_task_q is not None, (
                    "mem_task_q required when mem_updater is set"
                )
                assert box_task_q is not None, (
                    "box_task_q required when mem_updater is set"
                )
                assert plan_set is not None
                assert plan_set.box_plan is not None
                assert plan_set.mem_plan is not None
                mem_send_nids = self._union_planned_p2p_nids(plan_set.mem_plan)
                box_send_nids = self._union_planned_p2p_nids(plan_set.box_plan)
                stale_mem_source = self._select_stale_mem_source(meta, mem_send_nids)
                stale_box_source = self._select_stale_box_source(meta, box_send_nids)
            else:
                stale_mem_source = None
                stale_box_source = None

            # Run unified memory update on the flattened FlatGraph
            meta.flat_graph = self.mem_updater(
                meta.flat_graph,
                meta.n_src * 2,
                n_root_nodes=meta.n_src * (2 + meta.n_neg),
            )
            assert meta.flat_graph is not None
            assert meta.flat_graph.detached_mem is not None
            assert meta.flat_graph.detached_ts is not None
            assert meta.flat_graph.detached_nid is not None

            meta.graph = meta.flat_graph.unflatten(
                n_src=n_src,
                fanout=fanout,
                n_neg=meta.n_neg,
                root_expansion=self.root_expansion,
            )
            # Retrieve detached states for p2p sync

            with torch.no_grad():
                # Construct MemTsNid directly from detached states
                root_new_mem = MemEntry(
                    nid=meta.flat_graph.detached_nid,
                    ts=meta.flat_graph.detached_ts,
                    data=meta.flat_graph.detached_mem,
                )
                root_new_mem_clone = root_new_mem.clone()

                box_pos_efeat = (
                    meta.flat_graph.efeat[0:n_src].clone()
                    if meta.flat_graph.efeat is not None
                    else None
                )
                box_graph = meta.graph if self.postman else None
                box_flat_nid_cpu = meta.flat_nid_cpu if self.postman else None
                if self.dynamic_state_ablation.uses_async_dynamic_state_pipeline:
                    ready_event = cast(CudaEvent, torch.cuda.Event())
                    ready_event.record(stream=torch.cuda.current_stream())

                    # Handle mem sync
                    assert mem_task_q is not None
                    assert plan_set is not None
                    assert plan_set.mem_plan is not None
                    mem_task = MemTask(
                        mini_batch_idx=meta.mini_batch_idx,
                        root_new_mem=root_new_mem,
                        stale_mem=stale_mem_source,
                        mem_plan=plan_set.mem_plan,
                        is_last=meta.is_last_macro_of_epoch,
                    )
                    # Avoid in-place modification issues
                    mem_task_q.put((mem_task, ready_event))

                    assert box_task_q is not None
                    assert plan_set is not None
                    assert plan_set.box_plan is not None
                    box_task = BoxTask(
                        pos_efeat=box_pos_efeat,
                        root_new_mem=root_new_mem_clone,
                        stale_box=stale_box_source,
                        mail_peer_mem=None,
                        mini_batch_idx=meta.mini_batch_idx,
                        box_plan=plan_set.box_plan,
                        is_last=meta.is_last_macro_of_epoch,
                        graph=box_graph,
                        flat_nid_cpu=box_flat_nid_cpu,
                    )
                    box_task_q.put((box_task, ready_event))
                elif self.dynamic_state_ablation.uses_direct_cpu_writeback:
                    direct_mem_filter = None
                    direct_box_filter = None
                    if self.dynamic_state_ablation.filter_writeback_by_plan:
                        assert plan_set is not None
                        assert plan_set.mem_plan is not None
                        assert plan_set.box_plan is not None
                        assert plan_set.mem_plan.writeback is not None
                        assert plan_set.box_plan.writeback is not None
                        direct_mem_filter = plan_set.mem_plan.writeback.data
                        direct_box_filter = plan_set.box_plan.writeback.data
                    direct_box_entry = mail_to_box(
                        generate_interaction_mails(
                            root_new_mem_clone,
                            box_pos_efeat,
                            None,
                        )
                    )
                    request = DirectWritebackRequest(
                        mini_batch_idx=meta.mini_batch_idx,
                        mem_entry=root_new_mem.clone(),
                        box_entry=direct_box_entry,
                        mem_filter_nids=direct_mem_filter,
                        box_filter_nids=direct_box_filter,
                    )
                    if self.dynamic_state_ablation.direct_cpu_writeback_after_mem_update:
                        assert direct_writeback is not None
                        direct_writeback(request)
                    else:
                        meta.direct_writeback_request = request
                else:
                    raise RuntimeError(
                        "Unsupported dynamic_state_ablation runtime configuration."
                    )
        else:
            meta.graph = meta.flat_graph.unflatten(
                n_src=n_src,
                fanout=fanout,
                n_neg=meta.n_neg,
                root_expansion=self.root_expansion,
            )

        # 3. Pass the hierarchical HierGraph to the GNN
        encoder_result = cast(EncoderResult, self.encoder(graph=meta.graph))

        # 4. Predict edges
        edge_representations = self.representation_adapter(
            encoder_result,
            n_src=meta.n_src,
            n_neg=meta.n_neg,
        )
        ret = self.header(edge_representations)
        return cast(torch.Tensor, ret)
