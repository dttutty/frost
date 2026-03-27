"""Assembled model that composes memory updater, encoder, postman, and header.

This module wires together all model components (memory updater, GNN encoder,
optional APAN postman, and link predictor header) into a single forward pass
used by the worker during training and evaluation.
"""

import logging
from queue import Queue
from typing import cast

import torch
import torch.nn as nn

# Import the newly unified memory updaters
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.meta import Meta
from frost.datatypes.tasks import BoxTask, MemTask
from frost.protocols.cuda_event_proto import CudaEvent
from frost.protocols.mem_updater_proto import MemoryUpdaterProto
from frost.protocols.postman_proto import PostmanProto

logger = logging.getLogger(__name__)


class AssembledModel(nn.Module):
    """Full model graph for a single training or evaluation step.

    Chains the following components in order:
      1. Memory updater (optional): updates node memories via RNN/Transformer.
      2. FlatGraph inflation: converts flat tensor layout to a hierarchical Graph.
      3. Encoder: computes root-node embeddings from the inflated Graph.
      4. Header: produces link-prediction logits from the embeddings.

    When a memory updater is present, the updated memory and mailbox states are
    asynchronously pushed back to shared memory via ``mem_task_q`` and
    ``box_task_q``.

    Args:
        encoder: GNN encoder module (e.g., TGN, TGAT, DyGFormer).
        mem_updater: Optional memory updater (MemoryUpdater or
            TransformerMemoryUpdater).  Pass ``None`` for memory-free models.
        header: Prediction head module (e.g., LinkPredictor).
        postman: Optional APAN postman for mailbox delivery.  Pass ``None`` for
            non-APAN models.
        n_neg: Number of negative samples per positive edge.
        sample_4_neg: Whether negative destinations are sampled together with
            source and positive destination (True) or only for src+pos_dst
            (False, APAN mode).
    """

    def __init__(
        self,
        encoder: nn.Module,
        mem_updater: MemoryUpdaterProto | None,
        header: nn.Module,
        postman: PostmanProto | None,
        n_neg: int,
        sample_4_neg: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mem_updater = mem_updater
        self.header = header
        self.postman = postman

        self.n_neg = n_neg  # from hydra config
        self.sample_4_neg = sample_4_neg  # from hydra config

    def forward(
        self,
        meta: Meta,
        *,
        mem_task_q: Queue | None = None,
        box_task_q: Queue | None = None,
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
        logger.debug(f"Processing meta with meta.batch_idx={meta.mini_batch_idx}")

        # 1. Update memory using the flattened ZipSeqGraph

        n_src = meta.n_src
        fanout = meta.fanout if meta.fanout is not None else []
        assert meta.flat_graph is not None

        if self.mem_updater is not None:
            assert mem_task_q is not None, "mem_task_q required when mem_updater is set"
            assert box_task_q is not None, "box_task_q required when mem_updater is set"

            # Run unified memory update on the flattened ZipSeqGraph
            meta.flat_graph = self.mem_updater(meta.flat_graph, meta.n_src * 2)
            assert meta.flat_graph is not None
            assert meta.flat_graph.detached_mem is not None
            assert meta.flat_graph.detached_ts is not None
            assert meta.flat_graph.detached_nid is not None

            meta.graph = meta.flat_graph.unflatten(
                n_src=n_src,
                fanout=fanout,
                n_neg=self.n_neg,
                sample_4_neg=self.sample_4_neg,
            )
            # Retrieve detached states for p2p sync

            with torch.no_grad():
                # Construct MemTsNid directly from detached states
                root_new_mem = MemEntry(
                    nid=meta.flat_graph.detached_nid,
                    ts=meta.flat_graph.detached_ts,
                    data=meta.flat_graph.detached_mem,
                )

                plan_set = meta.plan_set

                assert plan_set is not None
                assert plan_set.box_plan is not None
                assert plan_set.mem_plan is not None
                root_new_mem_clone = root_new_mem.clone()

                ready_event = cast(CudaEvent, torch.cuda.Event())
                ready_event.record(stream=torch.cuda.current_stream())

                # Handle mem sync
                mem_task = MemTask(
                    mini_batch_idx=meta.mini_batch_idx,
                    root_new_mem=root_new_mem,
                    mem_plan=plan_set.mem_plan,
                    is_last=meta.is_last_macro_of_epoch,
                )
                # Avoid in-place modification issues
                mem_task_q.put((mem_task, ready_event))

                assert meta.flat_graph.box is not None
                assert meta.flat_graph.box_ts is not None

                # Handle box sync
                box_task = BoxTask(
                    pos_efeat=meta.flat_graph.efeat[0:n_src].clone()
                    if meta.flat_graph.efeat is not None
                    else None,
                    root_new_mem=root_new_mem_clone,
                    mini_batch_idx=meta.mini_batch_idx,
                    box_plan=plan_set.box_plan,
                    is_last=meta.is_last_macro_of_epoch,
                    graph=meta.graph if self.postman else None,
                    flat_nid_cpu=meta.flat_nid_cpu if self.postman else None,
                )
                box_task_q.put((box_task, ready_event))
        else:
            meta.graph = meta.flat_graph.unflatten(
                n_src=n_src,
                fanout=fanout,
                n_neg=self.n_neg,
                sample_4_neg=self.sample_4_neg,
            )

        # 3. Pass the hierarchical SeqGraph to the GNN
        h = self.encoder(graph=meta.graph)

        # 4. Predict edges
        ret = self.header(h)
        return cast(torch.Tensor, ret)
