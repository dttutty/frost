"""Assembled model that composes memory updater, encoder, postman, and header.

This module wires together all model components (memory updater, GNN encoder,
optional APAN postman, and link predictor header) into a single forward pass
used by the worker during training and evaluation.
"""

import json
import logging
import os
from queue import Queue
from pathlib import Path
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


def _debug_dump_model_dir() -> Path | None:
    raw = os.getenv("FROST_DEBUG_DUMP_MODEL_DIR")
    if raw is None or raw == "":
        return None
    return Path(raw)


def _debug_dump_model_batches() -> set[int] | None:
    raw = os.getenv("FROST_DEBUG_DUMP_MODEL_BATCHES")
    if raw is None or raw.strip() == "":
        return None
    return {int(token.strip()) for token in raw.split(",") if token.strip() != ""}


def _debug_model_checksums_path() -> Path | None:
    raw = os.getenv("FROST_DEBUG_MODEL_CHECKSUMS_PATH")
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    return Path(value)


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
        self._debug_dump_model_dir = _debug_dump_model_dir()
        self._debug_dump_model_batches = _debug_dump_model_batches()

    def _maybe_dump_model_state(
        self,
        *,
        meta: Meta,
        logits: torch.Tensor,
        root_new_mem: MemEntry | None,
    ) -> None:
        dump_dir = self._debug_dump_model_dir
        dump_batches = self._debug_dump_model_batches
        if dump_dir is None or dump_batches is None:
            return
        if meta.mini_batch_idx not in dump_batches:
            return

        dump_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mini_batch_idx": int(meta.mini_batch_idx),
            "split": meta.split,
            "logits": logits.detach().cpu(),
        }
        if root_new_mem is not None:
            payload["root_new_mem_nid"] = root_new_mem.nid.detach().cpu()
            payload["root_new_mem_ts"] = root_new_mem.ts.detach().cpu()
            payload["root_new_mem_data"] = root_new_mem.data.detach().cpu()
        torch.save(payload, dump_dir / f"batch_{meta.mini_batch_idx}.pt")

    def _maybe_append_model_checksums(
        self,
        *,
        meta: Meta,
        logits: torch.Tensor,
        root_new_mem: MemEntry | None,
    ) -> None:
        path = _debug_model_checksums_path()
        if path is None:
            return

        def _sum_or_zero(t: torch.Tensor | None) -> float:
            if t is None or t.numel() == 0:
                return 0.0
            return float(t.detach().float().sum().item())

        def _abssum_or_zero(t: torch.Tensor | None) -> float:
            if t is None or t.numel() == 0:
                return 0.0
            return float(t.detach().float().abs().sum().item())

        def _len_or_zero(t: torch.Tensor | None) -> int:
            if t is None:
                return 0
            return int(t.numel())

        record = {
            "mini_batch_idx": int(meta.mini_batch_idx),
            "cur_epoch": int(meta.cur_epoch),
            "split": meta.split,
            "cur_macro_of_split": int(meta.cur_macro_of_split),
            "cur_macro_of_epoch": int(meta.cur_macro_of_epoch),
            "logits_sum": _sum_or_zero(logits),
            "logits_abs_sum": _abssum_or_zero(logits),
            "root_new_mem_len": _len_or_zero(None if root_new_mem is None else root_new_mem.nid),
            "root_new_mem_nid_sum": _sum_or_zero(
                None if root_new_mem is None else root_new_mem.nid
            ),
            "root_new_mem_ts_sum": _sum_or_zero(
                None if root_new_mem is None else root_new_mem.ts
            ),
            "root_new_mem_sum": _sum_or_zero(
                None if root_new_mem is None else root_new_mem.data
            ),
            "root_new_mem_abs_sum": _abssum_or_zero(
                None if root_new_mem is None else root_new_mem.data
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

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

        # 1. Update memory using the flattened FlatGraph

        n_src = meta.n_src
        fanout = meta.fanout if meta.fanout is not None else []
        assert meta.flat_graph is not None
        root_new_mem_for_dump: MemEntry | None = None

        if self.mem_updater is not None:
            assert mem_task_q is not None, "mem_task_q required when mem_updater is set"
            assert box_task_q is not None, "box_task_q required when mem_updater is set"

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
                root_new_mem_for_dump = root_new_mem

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
                n_neg=meta.n_neg,
                sample_4_neg=self.sample_4_neg,
            )

        # 3. Pass the hierarchical HierGraph to the GNN
        h = self.encoder(graph=meta.graph)

        # 4. Predict edges
        ret = self.header(h)
        self._maybe_dump_model_state(
            meta=meta,
            logits=cast(torch.Tensor, ret),
            root_new_mem=root_new_mem_for_dump,
        )
        self._maybe_append_model_checksums(
            meta=meta,
            logits=cast(torch.Tensor, ret),
            root_new_mem=root_new_mem_for_dump,
        )
        return cast(torch.Tensor, ret)
