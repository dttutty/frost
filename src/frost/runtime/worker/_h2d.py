import logging
import threading
from queue import Queue as Q
from typing import Literal, cast

import torch

from frost.datatypes.meta import Meta
from frost.protocols.cuda_event_proto import CudaEvent
from frost.runtime._thread_base import StreamThreadBase
from frost.runtime.worker._st_asm import StAssembler
from frost.shm.ring_buf import RingBuf

logger = logging.getLogger(__name__)


class H2D(StreamThreadBase):
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
        self.st_asm = StAssembler(
            shared_nfeat=nfeat, shared_efeat=efeat, device=self.device
        )
        self.seen_root_nids = torch.tensor([], dtype=torch.long, device="cpu")

    @torch.no_grad()
    def logic(self) -> None:
        # batch_count = 0

        while not self.stop_ev.is_set():
            meta = self.in_q.get(block=True)

            if meta is None:
                self.out_q.put(None, block=True)
                break
            elif meta == "end_of_epoch":
                self.out_q.put("end_of_epoch", block=True)
                continue

            assert meta.flat_graph is not None

            if meta.plan_set is not None:
                meta.plan_set = meta.plan_set.to(self.device)

            # can run no matter has nfeat or not
            if self.efeat is not None:
                meta.flat_graph = self.st_asm.fill_mfg_with_efeat(
                    meta.flat_graph, n_src=meta.n_src, meta=meta
                )

            # move gnn_mfgs to gpu
            meta.flat_graph = meta.flat_graph.to(self.device)

            # init x

            one_labels = torch.ones(meta.n_src, device=self.device)
            zero_labels = torch.zeros(meta.n_src * self.n_neg, device=self.device)
            meta.labels = torch.cat([one_labels, zero_labels])

            # move plan to gpu

            # move nfeat and efeat to gpu
            if self.nfeat is not None:
                # prepare if has n_feat
                assert meta.plan_set is not None and meta.plan_set.nfeat is not None
                nfeat = self.st_asm.prepare_nfeat_cache(
                    nfeat_plan=meta.plan_set.nfeat, meta=meta
                )

                meta.flat_graph = self.st_asm.fill_mfg_with_nfeat(
                    flat_graph=meta.flat_graph,
                    nfeat_nid=nfeat,
                    meta=meta,
                )

            ready_event = cast(CudaEvent, torch.cuda.Event())
            ready_event.record(torch.cuda.current_stream())
            self.out_q.put((meta, ready_event), block=True)
