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
from frost.models.contracts import NodeSet
from frost.runtime.protocols import CudaEvent
from frost.runtime._thread_base import StreamThreadBase, put_shutdown_sentinel
from frost.runtime.worker._static_assembler import StaticAssembler
from frost.shm.ring_buf import RingBuf

logger = logging.getLogger(__name__)


class H2D(StreamThreadBase):
    """Background thread that transfers static features and labels from CPU to GPU.

    Dequeues ``Meta`` objects from ``in_q``, fills edge features and (when
    present) node features via ``StaticAssembler``, moves the flat graph to the
    target device, constructs binary labels (ones for positives, zeros for
    negatives), and enqueues the result together with a ``CudaEvent`` into
    ``out_q``.

    Args:
        ring_buf: Shared ring buffer (not used for data here, kept for context).
        nfeat: CPU tensor of node features, or ``None`` if the dataset has none.
        efeat: CPU tensor of edge features, or ``None`` if the dataset has none.
        in_q: Input queue producing ``Meta | "end_of_epoch" | None``.
        out_q: Output queue consuming ``(Meta, CudaEvent) | "end_of_epoch" | None``.
        stop_ev: Shared stop event.
        device: Target CUDA device.
        history: Number of historical hops (unused; reserved for future use).
        n_neg: Number of negative samples per positive edge.
    """

    def __init__(
        self,
        ring_buf: RingBuf,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
        in_q: Q[Meta | None | Literal["end_of_epoch"]],
        out_q: Q[tuple[Meta, CudaEvent] | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
        device: torch.device,
        history: int = 1,
        n_neg: int = 1,
        node_feature_scope: NodeSet = NodeSet.ALL_SAMPLED,
    ) -> None:
        super().__init__(name="h2d", stop_ev=stop_ev)
        self.nfeat = nfeat
        self.efeat = efeat
        self.device = device
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.ring_buf = ring_buf
        self.history = history
        self.n_neg = n_neg
        self.node_feature_scope = node_feature_scope
        self.st_asm = StaticAssembler(
            shared_nfeat=nfeat, shared_efeat=efeat, device=self.device
        )
        self.seen_root_nids = torch.tensor([], dtype=torch.long, device="cpu")

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
            meta = self.in_q.get(block=True)

            if meta is None:
                put_shutdown_sentinel(self.out_q, None)
                break
            elif meta == "end_of_epoch":
                self.out_q.put("end_of_epoch", block=True)
                continue

            assert meta.flat_graph is not None

            if meta.plan_set is not None:
                meta.plan_set = meta.plan_set.to(self.device)

            # can run no matter has nfeat or not
            if self.efeat is not None:
                meta.flat_graph = self.st_asm.fill_efeat(
                    meta.flat_graph, n_src=meta.n_src, meta=meta
                )

            # Root edge features have already been gathered above. Clear every
            # root EID before model execution to prevent the target interaction
            # from leaking through later consumers. The batch can use a
            # different n_neg from the ring buffer's configured maximum.
            n_root = meta.n_src * (2 + meta.n_neg)
            meta.flat_graph.eid[:n_root] = -1

            # move gnn_mfgs to gpu
            meta.flat_graph = meta.flat_graph.to(self.device)

            # init x

            one_labels = torch.ones(meta.n_src, device=self.device)
            zero_labels = torch.zeros(meta.n_src * meta.n_neg, device=self.device)
            meta.labels = torch.cat([one_labels, zero_labels])
            if meta.neg_valid_mask is not None:
                meta.neg_valid_mask = meta.neg_valid_mask.to(self.device)

            # move nfeat and efeat to gpu
            if self.nfeat is not None:
                # prepare if has n_feat
                assert (
                    meta.plan_set is not None and meta.plan_set.nfeat_plan is not None
                )
                nfeat = self.st_asm.prepare_nfeat_cache(
                    nfeat_plan=meta.plan_set.nfeat_plan, meta=meta
                )

                meta.flat_graph = self.st_asm.fill_nfeat(
                    flat_graph=meta.flat_graph,
                    nfeat_nid=nfeat,
                    meta=meta,
                    strict=self.node_feature_scope is NodeSet.ALL_SAMPLED,
                )

            ready_event = cast(CudaEvent, torch.cuda.Event())
            ready_event.record(torch.cuda.current_stream())
            self.out_q.put((meta, ready_event), block=True)
