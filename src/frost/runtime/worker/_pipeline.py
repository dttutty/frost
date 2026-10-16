from __future__ import annotations

import logging
import os
from queue import Queue as Q
from typing import Any, cast

import torch

from frost.runtime.worker.types import (
    BoxP2PQ,
    BoxTaskQ,
    BoxWrQ,
    MemP2PQ,
    MemTaskQ,
    MemWrQ,
)
from frost.runtime.worker.types import AssembledQ, DeltaQueue, H2DOutQ, ReceivedQ
from frost.datatypes.result import Result
from frost.runtime._thread_base import join_thread, put_shutdown_sentinel
from frost.runtime.experimental.dynamic_state import TransportMode
from frost.runtime.worker._dynamic_assembler import DynamicAssembler
from frost.runtime.worker._host_to_device import H2D
from frost.runtime.worker._feature_staging import (
    FeaturePrefetcher,
    FeatureStagePayload,
    FeatureStageRing,
)
from frost.runtime.worker._receiver import Receiver
from frost.runtime.worker.p2p_syncer._p2p_comm import P2PQHandler
from frost.runtime.worker.p2p_syncer._task_handlers import (
    MemTaskHandler,
    MessageHandler,
)
from frost.runtime.worker.p2p_syncer._writer import BoxMailWriter, SharedMemWriter
from frost.shm.ring_buf import RingBuf
from frost.shm.control_plane import SharedControlPlane


logger = logging.getLogger(__name__)


def _initial_h2d_pipeline_slots(*, p2p_world_size: int) -> int:
    # A single worker has no cross-GPU overlap benefit.
    return 2 if p2p_world_size == 1 else 4


def _get_memory_worker_queue_size_env(default: int = 1) -> int:
    raw = os.getenv("FROST_MEMORY_WORKER_QUEUE_SIZE")
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 1:
        raise ValueError("FROST_MEMORY_WORKER_QUEUE_SIZE must be >= 1")
    return value


class WorkerPipelineMixin:
    result_q: Q[Result]
    received_q: ReceivedQ
    delta_queue: DeltaQueue | None
    h2d_out_q: H2DOutQ | None
    full_q: AssembledQ
    mem_task_q: MemTaskQ | None
    box_task_q: BoxTaskQ | None
    mem_p2p_q: MemP2PQ | None
    box_p2p_q: BoxP2PQ | None
    mem_wr_q: MemWrQ | None
    box_wr_q: BoxWrQ | None
    dy_asm: DynamicAssembler | None
    mem_handler: MemTaskHandler | None
    mem_wr: SharedMemWriter | None
    box_handler: MessageHandler | None
    box_wr: BoxMailWriter | None
    p2p_q_handler: P2PQHandler | None
    feature_prefetcher: FeaturePrefetcher | None
    feature_ring: FeatureStageRing | None
    feature_q: Q[FeatureStagePayload] | None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _init_pipeline(
        self,
        ring_buf: RingBuf,
        control_plane: SharedControlPlane,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
    ) -> None:
        """Create all queues and background pipeline threads."""
        has_mem = self.mem_box_store is not None
        use_async_dynamic_state_pipeline = (
            has_mem
            and self.dynamic_state_policy.transport is TransportMode.ASYNC_P2P
        )
        memory_q_size = _get_memory_worker_queue_size_env(default=1)
        stateless_q_size = self.no_memory_queue_size if not has_mem else memory_q_size
        self.result_q = Q(maxsize=1)
        self.received_q = Q(maxsize=stateless_q_size)
        self.full_q = Q(maxsize=stateless_q_size)

        has_static_features = nfeat is not None or efeat is not None
        self.feature_prefetcher = None
        self.feature_ring = None
        self.feature_q = None
        if has_static_features and not getattr(self, "single_gpu_resident", False):
            feature_depth = stateless_q_size + 1
            self.feature_ring = FeatureStageRing(depth=feature_depth)
            self.feature_q = Q(maxsize=stateless_q_size)
            self.feature_prefetcher = FeaturePrefetcher(
                nfeat=nfeat,
                efeat=efeat,
                in_q=self.received_q,
                out_q=self.feature_q,
                ring=self.feature_ring,
                stop_ev=self.stop_ev,
            )
            h2d_source_q: Q[FeatureStagePayload] = self.feature_q
        else:
            h2d_source_q = cast(Q[FeatureStagePayload], self.received_q)

        self.receiver = Receiver(
            env=self.env,
            received_q=self.received_q,
            result_q=self.result_q,
            stop_ev=self.stop_ev,
            ring_buf=ring_buf,
            control_plane=control_plane,
        )

        if has_mem:
            assert self.mem_box_store is not None
            # DynamicAssembler keeps a uniform queue-based interface for both the
            # asynchronous P2P path and the direct shared-memory ablations.  The
            # latter bypass delta consumption in ``logic()``, but still require
            # a queue object during construction and orderly shutdown.
            self.delta_queue = Q(maxsize=memory_q_size)
            self.h2d_out_q = Q(maxsize=memory_q_size)
            if use_async_dynamic_state_pipeline:
                self._init_mem_box_threads()
                assert self.delta_queue is not None
            delta_queue = self.delta_queue
            assert delta_queue is not None
            self.dy_asm = DynamicAssembler(
                mem_box_store=self.mem_box_store,
                h2d_out_q=self.h2d_out_q,
                delta_queue=delta_queue,
                full_q=self.full_q,
                stop_ev=self.stop_ev,
                device=self.device,
                h2d_sem=self.h2d_sem,
                n_neg=self.n_neg,
                root_expansion=ring_buf.root_expansion,
                mem_task_q=self.mem_task_q,
                box_task_q=self.box_task_q,
                mem_wr_q=self.mem_wr_q,
                box_wr_q=self.box_wr_q,
                dynamic_state_policy=self.dynamic_state_policy,
                wait_for_resident_write=(
                    self.gpu_resident_state_writer.wait_for_latest_write
                    if getattr(self, "gpu_resident_state_writer", None) is not None
                    else None
                ),
                state_sync=getattr(self, "state_sync", None),
            )
            h2d_dest_q = self.h2d_out_q
        else:
            h2d_dest_q = self.full_q

        self.h2d = H2D(
            nfeat=nfeat,
            efeat=efeat,
            device=self.device,
            node_feature_scope=self.execution_plan.node_feature_scope,
            in_q=h2d_source_q,
            out_q=h2d_dest_q,
            stop_ev=self.stop_ev,
            feature_ring=self.feature_ring,
            resident_features=getattr(self, "single_gpu_resident", False),
            fanout=list(getattr(ring_buf, "fanout", [])),
            root_expansion=ring_buf.root_expansion,
        )

    def _init_mem_box_threads(self) -> None:
        """Create memory/box sync threads. Called only when mem_box_store is present."""
        assert self.mem_updater is not None
        assert self.mem_box_store is not None
        assert self.delta_queue is not None
        assert self.mail_combine is not None
        memory_q_size = _get_memory_worker_queue_size_env(default=1)

        self.mem_task_q = Q(maxsize=memory_q_size)
        self.box_task_q = Q(maxsize=memory_q_size)
        self.mem_p2p_q = Q(maxsize=memory_q_size)
        self.box_p2p_q = Q(maxsize=memory_q_size)
        self.mem_wr_q = Q(maxsize=memory_q_size)
        self.box_wr_q = Q(maxsize=memory_q_size)

        self.mem_handler = MemTaskHandler(
            h2d_sem=self.h2d_sem,
            mem_task_q=self.mem_task_q,
            mem_p2p_q=self.mem_p2p_q,
            stop_ev=self.stop_ev,
            device=self.device,
            mem_wr_q=self.mem_wr_q,
        )
        self.mem_wr = SharedMemWriter(
            shared_data=self.mem_box_store.mem,
            shared_ts=self.mem_box_store.mem_ts,
            h2d_sem=self.h2d_sem,
            stop_ev=self.stop_ev,
            wr_q=self.mem_wr_q,
            device=self.device,
            name="mem_wr",
            state_sync=getattr(self, "state_sync", None),
        )
        self.box_handler = MessageHandler(
            h2d_sem=self.h2d_sem,
            box_task_q=self.box_task_q,
            box_p2p_q=self.box_p2p_q,
            stop_ev=self.stop_ev,
            mail_combine=self.mail_combine,
            device=self.device,
            box_wr_q=self.box_wr_q,
            mailbox_router=self.mailbox_router,
            n_neg=self.n_neg,
        )
        self.box_wr = BoxMailWriter(
            mem_box_store=self.mem_box_store,
            h2d_sem=self.h2d_sem,
            stop_ev=self.stop_ev,
            wr_q=self.box_wr_q,
            device=self.device,
            name="box_wr",
            fanout=self.ring_buf.fanout,
            state_sync=getattr(self, "state_sync", None),
        )
        self.p2p_q_handler = P2PQHandler(
            env=self.env,
            mem_p2p_q=self.mem_p2p_q,
            box_p2p_q=self.box_p2p_q,
            nccl_p2p=self.nccl_p2p,
            p2p_sem=self.p2p_sem,
            stop_ev=self.stop_ev,
            delta_queue=self.delta_queue,
        )

    def start_all_threads(self) -> None:
        """Start all background pipeline threads.

        Must be called once before invoking ``run()``.  Threads are started in
        dependency order: H2D and Receiver first, then the optional memory/box
        sync threads (P2PQHandler, MemTaskHandler, MessageHandler, writers,
        DynamicAssembler) when a ``MemBoxStore`` is present.
        """
        self.h2d.start()
        if self.feature_prefetcher is not None:
            self.feature_prefetcher.start()
        self.receiver.start()
        if self.mem_box_store is not None:
            assert self.dy_asm is not None
            if self.dynamic_state_policy.transport is TransportMode.ASYNC_P2P:
                assert self.p2p_q_handler is not None
                assert self.mem_handler is not None
                assert self.box_handler is not None
                assert self.box_wr is not None
                assert self.mem_wr is not None
                self.p2p_q_handler.start()
                self.mem_handler.start()
                self.box_handler.start()
                self.box_wr.start()
                self.mem_wr.start()
            self.dy_asm.start()

    def _request_thread_shutdown(self) -> None:
        put_shutdown_sentinel(self.received_q, None)
        put_shutdown_sentinel(self.full_q, None)

        if self.feature_q is not None:
            put_shutdown_sentinel(self.feature_q, None)

        if self.h2d_out_q is not None:
            put_shutdown_sentinel(self.h2d_out_q, None)
        if self.delta_queue is not None:
            put_shutdown_sentinel(self.delta_queue, None)
        if self.mem_task_q is not None:
            put_shutdown_sentinel(self.mem_task_q, None)
        if self.box_task_q is not None:
            put_shutdown_sentinel(self.box_task_q, None)
        if self.mem_p2p_q is not None:
            put_shutdown_sentinel(self.mem_p2p_q, None)
        if self.box_p2p_q is not None:
            put_shutdown_sentinel(self.box_p2p_q, None)
        if self.mem_wr_q is not None:
            put_shutdown_sentinel(self.mem_wr_q, None)
        if self.box_wr_q is not None:
            put_shutdown_sentinel(self.box_wr_q, None)

    def _join_pipeline_threads(self, timeout_s: float) -> None:
        join_thread(self.h2d, timeout_s, logger)
        join_thread(self.feature_prefetcher, timeout_s, logger)
        join_thread(self.receiver, timeout_s, logger)
        join_thread(self.mem_handler, timeout_s, logger)
        join_thread(self.mem_wr, timeout_s, logger)
        join_thread(self.box_wr, timeout_s, logger)
        join_thread(self.box_handler, timeout_s, logger)
        join_thread(self.p2p_q_handler, timeout_s, logger)
        join_thread(self.dy_asm, timeout_s, logger)
