"""Host-to-Device (H2D) transfer thread for the worker pipeline.

Moves static features (nfeat, efeat) from shared CPU memory to the GPU for
each incoming batch and constructs per-batch label tensors.  When a
``MemBoxStore`` is absent the output queue feeds directly into ``full_q``;
otherwise it feeds into the ``DynamicAssembler``.
"""

import logging
import threading
from queue import Queue as Q
from typing import Literal, cast

import torch

from frost.datatypes.meta import Meta
from frost.models.contracts import NodeSet, RootExpansion
from frost.runtime.protocols import CudaEvent
from frost.runtime._thread_base import StreamThreadBase, put_shutdown_sentinel
from frost.runtime.worker._static_assembler import StaticAssembler
from frost.runtime.worker._feature_staging import (
    FeatureStagePayload,
    FeatureStageRing,
    StagedFeatureBatch,
)

logger = logging.getLogger(__name__)


class H2D(StreamThreadBase):
    """Background thread that transfers static features and labels from CPU to GPU.

    Dequeues ``Meta`` objects from ``in_q``, fills edge features and (when
    present) node features via ``StaticAssembler``, moves the flat graph to the
    target device, constructs binary labels (ones for positives, zeros for
    negatives), and enqueues the result together with a ``CudaEvent`` into
    ``out_q``.

    Args:
        nfeat: CPU tensor of node features, or ``None`` if the dataset has none.
        efeat: CPU tensor of edge features, or ``None`` if the dataset has none.
        in_q: Input queue producing ``Meta | "end_of_epoch" | None``.
        out_q: Output queue consuming ``(Meta, CudaEvent) | "end_of_epoch" | None``.
        stop_ev: Shared stop event.
        device: Target CUDA device.
    """

    def __init__(
        self,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
        in_q: Q[FeatureStagePayload],
        out_q: Q[tuple[Meta, CudaEvent] | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
        device: torch.device,
        node_feature_scope: NodeSet = NodeSet.ALL_SAMPLED,
        feature_ring: FeatureStageRing | None = None,
        resident_features: bool = False,
        fanout: list[int] | None = None,
        root_expansion: RootExpansion = RootExpansion.ALL_ROOTS,
    ) -> None:
        super().__init__(name="h2d", stop_ev=stop_ev)
        self.nfeat = nfeat
        self.efeat = efeat
        self.device = device
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.node_feature_scope = node_feature_scope
        self.feature_ring = feature_ring
        self.resident_features = resident_features
        self.fanout = fanout or []
        self.root_expansion = root_expansion
        self.st_asm = StaticAssembler(
            shared_nfeat=nfeat, shared_efeat=efeat, device=self.device
        )

    @torch.no_grad()
    def logic(self) -> None:
        """Main loop: transfer each batch from CPU to GPU and forward downstream.

        For each ``Meta`` received from ``in_q``:

        1. Fills edge features if ``efeat`` is configured.
        2. Zeros out negative-edge IDs to prevent label leakage.
        3. Moves the flat graph to the GPU.
        4. Constructs binary label tensor (1 per positive, 0 per negative).
        5. Fills node features if ``nfeat`` is configured.
        6. Records a CUDA event and enqueues ``(meta, event)`` into ``out_q``.
        """
        # batch_count = 0

        while not self.stop_ev.is_set():
            payload = self.in_q.get(block=True)

            if payload is None:
                put_shutdown_sentinel(self.out_q, None)
                break
            elif payload == "end_of_epoch":
                self.out_q.put("end_of_epoch", block=True)
                continue

            staged = payload if isinstance(payload, StagedFeatureBatch) else None
            meta = staged.meta if staged is not None else payload
            if not isinstance(meta, Meta):
                raise TypeError(f"Unexpected H2D payload {meta!r}")

            assert meta.flat_graph is not None
            try:
                if meta.plan_set is not None:
                    meta.plan_set = meta.plan_set.to(self.device)

                # The legacy direct path remains useful for focused unit tests;
                # production feature batches arrive through the staging ring.
                if self.efeat is not None and staged is None:
                    meta.flat_graph = self.st_asm.fill_efeat(
                        meta.flat_graph, n_src=meta.n_src
                    )

                # Root edge features have already been gathered above. Clear every
                # root EID before model execution to prevent the target interaction
                # from leaking through later consumers. The batch can use a
                # different n_neg from the ring buffer's configured maximum.
                n_root = meta.n_src * (2 + meta.n_neg)
                meta.flat_graph.eid[:n_root] = -1

                # move gnn_mfgs to gpu
                meta.flat_graph = meta.flat_graph.to(self.device)

                if staged is not None and staged.edge_values is not None:
                    assert staged.edge_positions is not None
                    meta.flat_graph = self.st_asm.fill_staged_efeat(
                        meta.flat_graph,
                        staged_efeat=staged.edge_values,
                        positions=staged.edge_positions,
                    )

                one_labels = torch.ones(meta.n_src, device=self.device)
                zero_labels = torch.zeros(meta.n_src * meta.n_neg, device=self.device)
                meta.labels = torch.cat([one_labels, zero_labels])
                if meta.neg_valid_mask is not None:
                    meta.neg_valid_mask = meta.neg_valid_mask.to(self.device)

                if self.nfeat is not None:
                    if self.resident_features:
                        meta.flat_graph = self.st_asm.fill_resident_nfeat(
                            meta.flat_graph,
                            n_src=meta.n_src,
                            n_neg=meta.n_neg,
                            fanout=self.fanout,
                            root_expansion=self.root_expansion,
                            scope=self.node_feature_scope,
                        )
                    else:
                        assert (
                            meta.plan_set is not None
                            and meta.plan_set.nfeat_plan is not None
                        )
                        nfeat = self.st_asm.prepare_nfeat_cache(
                            nfeat_plan=meta.plan_set.nfeat_plan,
                            staged_nfeat=(
                                staged.node_values if staged is not None else None
                            ),
                        )

                        meta.flat_graph = self.st_asm.fill_nfeat(
                            flat_graph=meta.flat_graph,
                            nfeat_lookup=nfeat,
                            strict=self.node_feature_scope is NodeSet.ALL_SAMPLED,
                        )
            finally:
                if staged is not None:
                    if self.feature_ring is None:
                        raise RuntimeError(
                            "staged feature payload requires a feature ring"
                        )
                    self.feature_ring.release(staged.slot)

            ready_event = cast(CudaEvent, torch.cuda.Event())
            ready_event.record(torch.cuda.current_stream())
            self.out_q.put((meta, ready_event), block=True)
