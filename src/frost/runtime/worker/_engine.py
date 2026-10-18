"""Worker engine for distributed Temporal GNN training.

Implements the GPU-resident Worker process that drives the train/eval loop.
Each Worker owns one GPU rank, a chain of pipeline threads (Receiver, H2D,
optional DynamicAssembler), and optional P2P memory-sync threads when a
MemBoxStore is configured.
"""

import logging
import os
import threading
from queue import Queue as Q

import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

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
from frost.models.temporal_link_model import TemporalLinkModel
from frost.runtime._thread_base import (
    destroy_process_group,
)
from frost.runtime._early_stopping import is_significant_improvement
from frost.runtime.worker._benchmark_observer import WorkerBenchmarkObserver
from frost.runtime.worker._split_benchmark_observer import (
    WorkerSplitBenchmarkObserver,
)
from frost.runtime.worker._config import WorkerOptions, WorkerResources
from frost.runtime.experimental.dynamic_state import (
    TransportMode,
    resolve_dynamic_state_policy,
)
from frost.runtime.worker._direct_writeback import DirectDynamicStateWriter
from frost.runtime.worker._gpu_resident_writeback import (
    GpuResidentDynamicStateWriter,
)
from frost.runtime.worker._checkpoint import WorkerCheckpointMixin
from frost.runtime.worker._dynamic_assembler import DynamicAssembler
from frost.runtime.worker._h2d_credit import BatchAwareCreditManager
from frost.runtime.worker._pipeline import (
    WorkerPipelineMixin,
    _initial_h2d_pipeline_slots,
)
from frost.runtime.worker._receiver import Receiver as Receiver
from frost.runtime.worker._trainer import WorkerTrainerMixin
from frost.runtime.worker.p2p_syncer._p2p_comm import P2PQHandler
from frost.runtime.worker.p2p_syncer._task_handlers import (
    MemTaskHandler,
    MessageHandler,
)
from frost.runtime.worker.p2p_syncer._writer import BoxMailWriter, SharedMemWriter
from frost.shm.dynamic_state_sync import SharedDynamicStateSync

logger = logging.getLogger(__name__)


class Worker(
    WorkerPipelineMixin,
    WorkerTrainerMixin,
    WorkerCheckpointMixin,
):
    """Distributed training worker that owns the GPU-side pipeline.

    Each worker process corresponds to one GPU rank. It manages a chain of
    background threads (Receiver → H2D → [DynamicAssembler] → main loop) and
    optional P2P memory-sync threads (MemTaskHandler, MessageHandler,
    SharedMemWriter, P2PQHandler) when a ``MemBoxStore`` is present.

    Attributes:
        received_q: Queue carrying ``Meta`` objects freshly received from the host.
        delta_queue: Queue carrying P2P delta tuples; only used when mem_box_store is present.
        h2d_out_q: Intermediate queue between H2D and DynamicAssembler.
        full_q: Queue of fully-assembled ``(Meta, CudaEvent)`` tuples ready for the model.
        mem_task_q: Queue of memory update tasks posted by DynamicStateCoordinator.
        box_task_q: Queue of mailbox update tasks posted by DynamicStateCoordinator.
        mem_p2p_q: Queue of P2P memory descriptors produced by MemTaskHandler.
        box_p2p_q: Queue of P2P mailbox descriptors produced by MessageHandler.
        mem_wr_q: Queue of write-back requests for the node memory store.
        box_wr_q: Queue of write-back requests for the mailbox store.
        dy_asm: DynamicAssembler thread (memory-model path only).
        mem_handler: MemTaskHandler thread (memory-model path only).
        mem_wr: SharedMemWriter thread for node memory (memory-model path only).
        box_handler: MessageHandler thread (memory-model path only).
        box_wr: SharedMemWriter thread for mailbox (memory-model path only).
        p2p_q_handler: P2PQHandler thread (memory-model path only).
        model: TemporalLinkModel wrapped in DDP.
        result_q: Single-element queue used to pass epoch results back to the Receiver.
    """

    # --- Pipeline queues ---
    received_q: ReceivedQ
    delta_queue: DeltaQueue | None = None
    h2d_out_q: H2DOutQ | None = None
    full_q: AssembledQ

    mem_task_q: MemTaskQ | None = None
    box_task_q: BoxTaskQ | None = None
    mem_p2p_q: MemP2PQ | None = None
    box_p2p_q: BoxP2PQ | None = None
    mem_wr_q: MemWrQ | None = None
    box_wr_q: BoxWrQ | None = None

    # --- Background threads (None when mem_box_store is absent) ---
    dy_asm: DynamicAssembler | None = None
    mem_handler: MemTaskHandler | None = None
    mem_wr: SharedMemWriter | None = None
    box_handler: MessageHandler | None = None
    box_wr: BoxMailWriter | None = None
    p2p_q_handler: P2PQHandler | None = None
    direct_state_writer: DirectDynamicStateWriter | None = None
    gpu_resident_state_writer: GpuResidentDynamicStateWriter | None = None
    state_sync: SharedDynamicStateSync | None = None

    model: TemporalLinkModel | DDP
    optimizer: torch.optim.Optimizer
    result_q: Q[Result]

    def __init__(
        self,
        resources: WorkerResources,
        options: WorkerOptions,
    ) -> None:
        # --- Basic config ---
        self.ring_buf = resources.ring_buffer
        self.execution_plan = options.execution_plan
        self.n_neg = options.max_negative_samples
        self.save_to = options.checkpoint_dir
        self.env = resources.env
        self.mailbox_router = resources.mailbox_router
        self.mem_updater = resources.memory_updater
        self.mem_box_store = resources.dynamic_store
        self.nccl = resources.ddp_group
        self.nccl_p2p = resources.p2p_group
        self.device = torch.device(f"cuda:{resources.env.local_rank}")
        self.single_gpu_resident = options.single_gpu_resident
        if self.single_gpu_resident and resources.env.n_gpu != 1:
            raise ValueError("The GPU-resident backend requires exactly one worker.")
        node_features = resources.node_features
        edge_features = resources.edge_features
        if self.single_gpu_resident:
            node_features = (
                None
                if node_features is None
                else node_features.to(self.device, dtype=torch.float32)
            )
            edge_features = (
                None
                if edge_features is None
                else edge_features.to(self.device, dtype=torch.float32)
            )
        self.mail_combine = options.mail_combine
        self.dynamic_state_policy = (
            options.dynamic_state_policy
            if options.dynamic_state_policy is not None
            else resolve_dynamic_state_policy("baseline")
        )
        self.compile_model = options.compile_model
        self.compile_mode = options.compile_mode
        self.amp = options.amp
        self.lr = options.learning_rate
        if options.eval_metric not in {"ap", "mrr"}:
            raise ValueError(
                f"Unsupported eval_metric={options.eval_metric!r}; expected 'ap' or 'mrr'."
            )
        self.eval_metric = options.eval_metric
        self.train_only = options.train_only
        if options.early_stop_tolerance < 0:
            raise ValueError("early_stop_tolerance must be >= 0.")
        self.early_stop_tolerance = options.early_stop_tolerance
        if options.no_memory_queue_size < 1:
            raise ValueError("no_memory_queue_size must be >= 1.")
        self.no_memory_queue_size = options.no_memory_queue_size
        self.curr_epoch = 0
        self.local_best_metric = float("-inf")

        self.benchmark_observer = WorkerBenchmarkObserver(
            env=resources.env,
            device=self.device,
            cpu_pid_list=options.cpu_benchmark_pid_list,
        )
        self.split_benchmark_observer = WorkerSplitBenchmarkObserver(
            env=resources.env,
            device=self.device,
        )
        self._shutdown_done = False

        # --- Synchronization primitives ---
        self.stop_ev = threading.Event()
        self.state_sync = resources.state_sync
        if self.state_sync is not None:
            self.state_sync.stop_ev = self.stop_ev
        self.h2d_sem = BatchAwareCreditManager()
        self.p2p_sem = threading.Semaphore(0)
        if (
            self.mem_box_store is not None
            and self.dynamic_state_policy.transport is TransportMode.DIRECT_CPU
        ):
            self.direct_state_writer = DirectDynamicStateWriter(
                mem_box_store=self.mem_box_store,
                h2d_sem=self.h2d_sem,
            )
        if (
            self.mem_box_store is not None
            and self.dynamic_state_policy.transport is TransportMode.GPU_LOCAL
        ):
            self.gpu_resident_state_writer = GpuResidentDynamicStateWriter(
                mem_box_store=self.mem_box_store,
                h2d_sem=self.h2d_sem,
                mail_combine=self.mail_combine or "last",
            )

        self._init_pipeline(
            resources.ring_buffer,
            resources.control_plane,
            node_features,
            edge_features,
        )
        self._init_model(
            resources.encoder,
            resources.memory_updater,
            resources.predictor,
            options.optimizer,
            resources.ring_buffer,
            compile_model=self.compile_model,
            compile_mode=self.compile_mode,
            use_ddp=not self.single_gpu_resident,
        )
        self._init_metrics()

    def run(self) -> None:
        """Consume assembled batches and dispatch train, eval, and epoch events."""
        self.start_all_threads()
        if self.mem_box_store is not None:
            if self.dynamic_state_policy.max_in_flight_batches is not None:
                max_in_flight_batches = (
                    self.dynamic_state_policy.max_in_flight_batches
                )
                initial_slots = max_in_flight_batches * 2
            else:
                p2p_world_size = dist.get_world_size(self.nccl_p2p)
                initial_slots = _initial_h2d_pipeline_slots(
                    p2p_world_size=p2p_world_size,
                )
                max_in_flight_batches = max(1, initial_slots // 2)
            if initial_slots % 2 != 0:
                raise ValueError(
                    "initial h2d slots must be even because each batch consumes "
                    f"two writeback credits, got {initial_slots}"
                )
            set_capacity = getattr(
                self.h2d_sem, "set_max_in_flight_batches", None
            )
            if callable(set_capacity):
                set_capacity(max_in_flight_batches)
            else:
                for _ in range(initial_slots):
                    self.h2d_sem.release()
            if self.delta_queue is not None:
                self.delta_queue.put("nothing")

        while not self.stop_ev.is_set():
            full = self.full_q.get()
            if isinstance(full, tuple):
                meta, ready_event = full
                self.split_benchmark_observer.on_batch_start(
                    split=meta.split,
                    epoch=self.curr_epoch + 1,
                    events=meta.n_src,
                    scored_pairs=meta.n_src * (1 + meta.n_neg),
                )
                torch.cuda.current_stream().wait_event(ready_event)
                if meta.split == "train":
                    assert meta.labels is not None
                    self.benchmark_observer.on_train_batch_start(meta.n_src)
                    self.train(meta)
                    self.benchmark_observer.on_train_batch_end()
                else:
                    self.eval(meta)
                self.split_benchmark_observer.on_batch_end()
            elif full == "end_of_epoch":
                self.split_benchmark_observer.on_epoch_end()
                self.benchmark_observer.on_epoch_end()
                result = self.on_train_epoch_end()
                selection_metric = (
                    result.val_ap if self.eval_metric == "ap" else result.val_mrr
                )
                if (
                    not self.train_only
                    and is_significant_improvement(
                        selection_metric,
                        self.local_best_metric,
                        self.early_stop_tolerance,
                    )
                ):
                    self.local_best_metric = selection_metric
                    self.save_model(os.path.join(self.save_to, "best.ckpt"))
                self.curr_epoch += 1
                if self.mem_box_store:
                    self._flush_epoch_writebacks()
                    stale_p2p_tokens = self._drain_stale_p2p_guard_tokens()
                    if stale_p2p_tokens > 0:
                        logger.debug(
                            "[Rank %s] Drained %s stale p2p guard token(s) at epoch end",
                            self.env.local_rank,
                            stale_p2p_tokens,
                        )
                    if self.state_sync is not None:
                        self.state_sync.finish_epoch(self.mem_box_store.reset)
                    else:
                        self.mem_box_store.reset()
                # Receiver may release the host to dispatch the next epoch only
                # after every shared-state writer and the reset have completed.
                self.result_q.put(result)
            elif full is None:
                self.split_benchmark_observer.finalize()
                self.benchmark_observer.finalize()
                if (
                    self.mem_box_store is not None
                    and self.dynamic_state_policy.transport is TransportMode.ASYNC_P2P
                ):
                    assert self.box_task_q is not None
                    assert self.mem_task_q is not None
                    assert self.mem_p2p_q is not None
                    self.box_task_q.put(None)
                    self.mem_task_q.put(None)
                    self.mem_p2p_q.put(None)
                break

    def cleanup(self, graceful: bool = True, join_timeout_s: float = 5.0) -> None:
        """Best-effort worker shutdown for both normal and aborted runs."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.stop_ev.set()
        if self.state_sync is not None and not graceful:
            self.state_sync.abort()
        self._request_thread_shutdown()

        # Reset dynamo/triton compiled state before distributed teardown to avoid
        # C++ destructor ordering issues on exit.
        torch._dynamo.reset()

        pg_destroyed = False
        if not graceful:
            destroy_process_group(sync=False, logger_=logger)
            pg_destroyed = True

        self._join_pipeline_threads(join_timeout_s)

        alive_threads = [
            thread
            for thread in (
                self.h2d,
                self.receiver,
                self.mem_handler,
                self.mem_wr,
                self.box_wr,
                self.box_handler,
                self.p2p_q_handler,
                self.dy_asm,
            )
            if thread is not None and thread.is_alive()
        ]
        if alive_threads and not pg_destroyed:
            logger.warning(
                "Forcing process-group teardown because worker threads are still alive: %s",
                [thread.name for thread in alive_threads],
            )
            destroy_process_group(sync=False, logger_=logger)
            pg_destroyed = True
            self._join_pipeline_threads(1.0)

        if not pg_destroyed:
            destroy_process_group(sync=graceful, logger_=logger)

        if self.state_sync is not None:
            self.state_sync.close()

        # If non-daemon threads are still stuck on blocking collective ops
        # (scatter, NCCL), Python's normal exit will hang waiting for them.
        # Force-exit to release the CUDA context immediately.
        still_alive = [
            t
            for t in (
                self.h2d,
                self.receiver,
                self.mem_handler,
                self.mem_wr,
                self.box_wr,
                self.box_handler,
                self.p2p_q_handler,
                self.dy_asm,
            )
            if t is not None and t.is_alive()
        ]
        if still_alive:
            logger.warning(
                "Threads still alive after cleanup: %s — forcing os._exit",
                [t.name for t in still_alive],
            )
            os._exit(1)

    # Public model save/load interface
