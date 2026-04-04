"""Worker engine for distributed Temporal GNN training.

Implements the GPU-resident Worker process that drives the train/eval loop.
Each Worker owns one GPU rank, a chain of pipeline threads (Receiver, H2D,
optional DynamicAssembler), and optional P2P memory-sync threads when a
MemBoxStore is configured.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from queue import Queue as Q
from typing import Literal, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)

from frost.aliases.syncer import (
    BoxP2PQ,
    BoxTaskQ,
    BoxWrQ,
    MemP2PQ,
    MemTaskQ,
    MemWrQ,
)
from frost.aliases.worker import AssembledQ, DeltaQueue, H2DOutQ, ReceivedQ
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.models._assembled_model import AssembledModel
from frost.protocols.mem_updater_proto import MemoryUpdaterProto
from frost.protocols.postman_proto import PostmanProto
from frost.runtime._thread_base import (
    destroy_process_group,
    join_thread,
    put_shutdown_sentinel,
)
from frost.runtime.worker._dynamic_assembler import DynamicAssembler
from frost.runtime.worker._h2d_credit import BatchAwareCreditManager
from frost.runtime.worker._host_to_device import H2D
from frost.runtime.worker._receiver import Receiver as Receiver
from frost.runtime.worker.p2p_syncer._p2p_comm import P2PQHandler
from frost.runtime.worker.p2p_syncer._task_handlers import (
    MemTaskHandler,
    MessageHandler,
)
from frost.runtime.worker.p2p_syncer._writer import BoxMailWriter, SharedMemWriter
from frost.shm.mem_mgr import MemBoxStore
from frost.shm.ring_buf import RingBuf

logger = logging.getLogger(__name__)


def _debug_force_cpu_freshness_single_worker() -> bool:
    flag = os.getenv("FROST_DEBUG_CPU_FRESHNESS_SINGLE_WORKER")
    if flag is None:
        return False
    return flag.lower() in {"1", "true", "yes", "on"}


def _debug_initial_h2d_slots_override() -> int | None:
    raw = os.getenv("FROST_DEBUG_INITIAL_H2D_SLOTS")
    if raw is None:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError("FROST_DEBUG_INITIAL_H2D_SLOTS must be >= 0")
    return value


def _initial_h2d_pipeline_slots(
    *,
    debug_force_cpu_freshness_single_worker: bool,
    p2p_world_size: int,
) -> int:
    override = _debug_initial_h2d_slots_override()
    if override is not None:
        return override

    # Single-worker memory runs have no cross-GPU overlap benefit, and
    # allowing two batches in flight has proven much more timing-sensitive
    # than the serialized one-batch path. Keep that configuration at one
    # batch in flight by default.
    if p2p_world_size == 1:
        return 2
    return 4


def _debug_disable_best_ckpt_save() -> bool:
    flag = os.getenv("FROST_DEBUG_DISABLE_BEST_CKPT_SAVE")
    if flag is None:
        return False
    return flag.lower() in {"1", "true", "yes", "on"}


def _debug_sync_train_step_boundaries() -> bool:
    flag = os.getenv("FROST_DEBUG_SYNC_TRAIN_STEP_BOUNDARIES")
    if flag is None:
        return False
    return flag.lower() in {"1", "true", "yes", "on"}


def _debug_sync_membox_after_train_step() -> bool:
    flag = os.getenv("FROST_DEBUG_SYNC_MEMBOX_AFTER_TRAIN_STEP")
    if flag is None:
        return False
    return flag.lower() in {"1", "true", "yes", "on"}


def _debug_param_checksums_path() -> Path | None:
    raw = os.getenv("FROST_DEBUG_PARAM_CHECKSUMS_PATH")
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    return Path(value)


def _debug_param_checksum_batches() -> set[int] | None:
    raw = os.getenv("FROST_DEBUG_PARAM_CHECKSUM_BATCHES")
    if raw is None or raw.strip() == "":
        return None
    return {int(token.strip()) for token in raw.split(",") if token.strip() != ""}


def _debug_grad_checksums_path() -> Path | None:
    raw = os.getenv("FROST_DEBUG_GRAD_CHECKSUMS_PATH")
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    return Path(value)


def _debug_grad_checksum_batches() -> set[int] | None:
    raw = os.getenv("FROST_DEBUG_GRAD_CHECKSUM_BATCHES")
    if raw is None or raw.strip() == "":
        return None
    return {int(token.strip()) for token in raw.split(",") if token.strip() != ""}


def _debug_step_tensor_dump_dir() -> Path | None:
    raw = os.getenv("FROST_DEBUG_DUMP_STEP_TENSOR_DIR")
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    return Path(value)


def _debug_step_tensor_dump_batches() -> set[int] | None:
    raw = os.getenv("FROST_DEBUG_DUMP_STEP_TENSOR_BATCHES")
    if raw is None or raw.strip() == "":
        return None
    return {int(token.strip()) for token in raw.split(",") if token.strip() != ""}


def _debug_step_tensor_dump_names() -> set[str] | None:
    raw = os.getenv("FROST_DEBUG_DUMP_STEP_TENSOR_NAMES")
    if raw is None or raw.strip() == "":
        return None
    return {token.strip() for token in raw.split(",") if token.strip() != ""}


class P2PGuardCommHook:
    """Wait for P2P memory sync to finish before the first AllReduce of each training step."""

    def __init__(self, p2p_sem: threading.Semaphore) -> None:
        self.p2p_sem = p2p_sem
        self._need_acquire = False

    def arm(self) -> None:
        """Call at the start of each training step to arm the guard for this round."""
        self._need_acquire = True

    def __call__(self, state: dist.ProcessGroup, bucket: dist.GradBucket):  # type: ignore
        if self._need_acquire:
            self.p2p_sem.acquire()
            self._need_acquire = False
        return allreduce_hook(state, bucket)


class Worker:
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
        mem_task_q: Queue of memory update tasks posted by AssembledModel.
        box_task_q: Queue of mailbox update tasks posted by AssembledModel.
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
        model: AssembledModel wrapped in DDP.
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

    model: AssembledModel | DDP
    result_q: Q[Result]

    def __init__(
        self,
        n_neg: int,
        mem_box_store: MemBoxStore | None,
        save_to: str,
        nfeat: None | torch.Tensor,
        efeat: None | torch.Tensor,
        env: Env,
        nccl_ddp: dist.ProcessGroup,
        nccl_p2p: dist.ProcessGroup,
        mem_updater: MemoryUpdaterProto | None,
        encoder: nn.Module,
        predictor: nn.Module,
        ring_buf: RingBuf,
        postman: PostmanProto | None,
        criterion: torch.nn.Module,
        optimizer: Literal["adam", "sgd"],
        lr: float,
        mail_combine: Literal["last", "mean"] | None,
        use_shm_instead: bool = False,
        compile_model: bool = True,
        amp: bool = False,
        no_memory_queue_size: int = 4,
    ) -> None:
        # --- Basic config ---
        self.ring_buf = ring_buf
        self.n_neg = n_neg
        self.save_to = save_to
        self.env = env
        self.postman = postman
        self.mem_updater = mem_updater
        self.mem_box_store = mem_box_store
        self.nccl = nccl_ddp
        self.nccl_p2p = nccl_p2p
        self.device = torch.device(f"cuda:{env.local_rank}")
        self.mail_combine = mail_combine
        self.use_shm_instead = use_shm_instead
        self.compile_model = compile_model
        self.amp = amp
        self.lr = lr
        if no_memory_queue_size < 1:
            raise ValueError("no_memory_queue_size must be >= 1.")
        self.no_memory_queue_size = no_memory_queue_size
        self.curr_epoch = 0
        self.local_best_ap = 0.0
        self._last_train_start_t: float | None = None

        # Throughput tracking (skip first 4 batches as warm-up)
        self._throughput_batch_count = 0
        self._throughput_events = 0
        self._throughput_start_time: float | None = None
        self._shutdown_done = False

        # --- Synchronization primitives ---
        self.stop_ev = threading.Event()
        self.h2d_sem = BatchAwareCreditManager()
        self.p2p_sem = threading.Semaphore(0)

        self._init_pipeline(ring_buf, nfeat, efeat)
        self._init_model(
            encoder,
            mem_updater,
            predictor,
            postman,
            criterion,
            optimizer,
            ring_buf,
            compile_model=self.compile_model,
        )
        self._init_metrics()

    def _init_pipeline(
        self,
        ring_buf: RingBuf,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
    ) -> None:
        """Create all queues and background pipeline threads."""
        has_mem = self.mem_box_store is not None
        stateless_q_size = self.no_memory_queue_size if not has_mem else 1
        self.result_q = Q(maxsize=1)
        self.received_q = Q(maxsize=stateless_q_size)
        self.full_q = Q(maxsize=stateless_q_size)

        self.receiver = Receiver(
            env=self.env,
            received_q=self.received_q,
            result_q=self.result_q,
            stop_ev=self.stop_ev,
            ring_buf=ring_buf,
        )

        if has_mem:
            assert self.mem_box_store is not None
            self.delta_queue = Q(maxsize=1)
            self.h2d_out_q = Q(maxsize=1)
            self._init_mem_box_threads()
            self.dy_asm = DynamicAssembler(
                mem_box_store=self.mem_box_store,
                h2d_out_q=self.h2d_out_q,
                delta_queue=self.delta_queue,
                full_q=self.full_q,
                stop_ev=self.stop_ev,
                device=self.device,
                h2d_sem=self.h2d_sem,
                n_neg=self.n_neg,
                sample_4_neg=ring_buf.sample_4_neg,
                use_shm_instead=self.use_shm_instead,
                mem_task_q=self.mem_task_q,
                box_task_q=self.box_task_q,
                mem_wr_q=self.mem_wr_q,
                box_wr_q=self.box_wr_q,
            )
            h2d_dest_q = self.h2d_out_q
        else:
            h2d_dest_q = self.full_q

        self.h2d = H2D(
            ring_buf=ring_buf,
            nfeat=nfeat,
            efeat=efeat,
            device=self.device,
            n_neg=self.n_neg,
            in_q=self.received_q,
            out_q=h2d_dest_q,
            stop_ev=self.stop_ev,
        )

    def _init_mem_box_threads(self) -> None:
        """Create memory/box sync threads. Called only when mem_box_store is present."""
        assert self.mem_updater is not None
        assert self.mem_box_store is not None
        assert self.delta_queue is not None
        assert self.mail_combine is not None

        self.mem_task_q = Q(maxsize=1)
        self.box_task_q = Q(maxsize=1)
        self.mem_p2p_q = Q(maxsize=1)
        self.box_p2p_q = Q(maxsize=1)
        self.mem_wr_q = Q(maxsize=1)
        self.box_wr_q = Q(maxsize=1)

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
        )
        self.box_handler = MessageHandler(
            h2d_sem=self.h2d_sem,
            box_task_q=self.box_task_q,
            box_p2p_q=self.box_p2p_q,
            stop_ev=self.stop_ev,
            mail_combine=self.mail_combine,
            device=self.device,
            box_wr_q=self.box_wr_q,
            postman=self.postman,
            n_neg=self.n_neg,
            fanout=self.ring_buf.fanout,
        )
        self.box_wr = BoxMailWriter(
            mem_box_store=self.mem_box_store,
            h2d_sem=self.h2d_sem,
            stop_ev=self.stop_ev,
            wr_q=self.box_wr_q,
            device=self.device,
            name="box_wr",
            n_neg=self.n_neg,
            fanout=self.ring_buf.fanout,
        )
        self.p2p_q_handler = P2PQHandler(
            env=self.env,
            mem_p2p_q=self.mem_p2p_q,
            box_p2p_q=self.box_p2p_q,
            nccl_p2p=self.nccl_p2p,
            p2p_sem=self.p2p_sem,
            stop_ev=self.stop_ev,
            delta_queue=self.delta_queue,
            device=self.device,
            use_shm_instead=self.use_shm_instead,
        )

    def _init_model(
        self,
        encoder: nn.Module,
        mem_updater: MemoryUpdaterProto | None,
        predictor: nn.Module,
        postman: PostmanProto | None,
        criterion: torch.nn.Module,
        optimizer: Literal["adam", "sgd"],
        ring_buf: RingBuf,
        compile_model: bool = True,
    ) -> None:
        """Create model, optimizer, and wrap with DDP."""
        self.model = AssembledModel(
            encoder=encoder,
            mem_updater=mem_updater,
            header=predictor,
            n_neg=self.n_neg,
            postman=postman,
            sample_4_neg=ring_buf.sample_4_neg,
        )

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        self.model = self.model.cuda()
        self.model = DDP(
            self.model, process_group=self.nccl, find_unused_parameters=False
        )
        self.p2p_guard = P2PGuardCommHook(self.p2p_sem)
        _guard = self.p2p_guard

        def p2p_guard_comm_hook(state, bucket):
            return _guard(state, bucket)

        self.model.register_comm_hook(state=self.nccl, hook=p2p_guard_comm_hook)
        if compile_model:
            # Compile only the pure-compute submodules (encoder, mem_updater, header).
            # AssembledModel.forward has too many non-tensor operations (Queue puts,
            # CUDA events, dataclass access) that cause graph breaks and recompilation.
            bare = self.model.module
            bare.encoder = torch.compile(bare.encoder)
            if bare.mem_updater is not None:
                bare.mem_updater = torch.compile(bare.mem_updater)
            bare.header = torch.compile(bare.header)
        self.criterion = criterion
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def _init_metrics(self) -> None:
        """Create torchmetrics trackers for training and evaluation."""
        self.train_start2start_ms = MeanMetric(process_group=self.nccl).cuda()
        self.train_loss_tracker = MeanMetric(process_group=self.nccl).cuda()
        self.val_mrr_tracker = MeanMetric(process_group=self.nccl).cuda()
        self.test_mrr_tracker = MeanMetric(process_group=self.nccl).cuda()
        self.val_auc_batch_mean = MeanMetric(process_group=self.nccl).cuda()
        self.val_ap_batch_mean = MeanMetric(process_group=self.nccl).cuda()
        self.test_auc_batch_mean = MeanMetric(process_group=self.nccl).cuda()
        self.test_ap_batch_mean = MeanMetric(process_group=self.nccl).cuda()

        self.val_metrics = MetricCollection(
            {
                "val_loss": MeanMetric(process_group=self.nccl),
                # Use exact distributed metrics here. The binned approximation
                # can shift the selected best epoch enough to materially change
                # the reported JODIE parity numbers.
                "val_auc": BinaryAUROC(thresholds=None, process_group=self.nccl),
                "val_ap": BinaryAveragePrecision(
                    thresholds=None, process_group=self.nccl
                ),
            }
        ).cuda()

        self.test_metrics = MetricCollection(
            {
                "test_auc": BinaryAUROC(thresholds=None, process_group=self.nccl),
                "test_ap": BinaryAveragePrecision(
                    thresholds=None, process_group=self.nccl
                ),
            }
        ).cuda()

    def _flush_epoch_writebacks(self) -> None:
        """Block until async mem/mail handlers, P2P routing, and writers drain.

        Without this flush, the main loop can call ``mem_box_store.reset()``
        while the last batch's async writeback is still in flight, allowing late
        test-split writes or stale P2P guard tokens to leak into the next epoch.
        """
        if self.mem_box_store is None:
            return

        for q in (
            self.mem_task_q,
            self.box_task_q,
            self.mem_p2p_q,
            self.box_p2p_q,
            self.mem_wr_q,
            self.box_wr_q,
        ):
            assert q is not None
            q.join()

    def _drain_stale_p2p_guard_tokens(self) -> int:
        """Drop leftover epoch-end P2P guard tokens.

        The final batch of an epoch still posts one P2P completion token even
        though there is no subsequent train/eval step in the same epoch to
        consume it. If left in the semaphore, the next epoch's first train
        backward can acquire a stale token and skip waiting for its own batch's
        P2P completion.
        """
        drained = 0
        while self.p2p_sem.acquire(blocking=False):
            drained += 1
        return drained

    def _flush_train_step_async_state(self) -> None:
        """Wait until the current train step's async mem/mail sync fully drains.

        This debug path preserves the normal P2P/writeback pipeline, but removes
        cross-step overlap by waiting for:

        - mem/box task handlers to consume the current batch
        - P2P routing to finish the current round
        - mem/box writers to persist local shared-memory updates

        That approximates TGL's step-local memory/mailbox update semantics
        without disabling P2P transport itself.
        """
        if self.mem_box_store is None:
            return

        assert self.mem_task_q is not None
        assert self.box_task_q is not None
        assert self.mem_p2p_q is not None
        assert self.box_p2p_q is not None
        assert self.mem_wr_q is not None
        assert self.box_wr_q is not None

        self.mem_task_q.join()
        self.box_task_q.join()
        self.mem_p2p_q.join()
        self.box_p2p_q.join()
        self.p2p_sem.acquire()
        self.mem_wr_q.join()
        self.box_wr_q.join()

    def on_train_epoch_end(self) -> Result:
        """Aggregate per-epoch metrics across all ranks and return a Result.

        Computes synchronized mean train loss, validation loss/AUC/AP, and test
        AUC/AP via torchmetrics all-reduce, then resets all accumulators.

        Returns:
            Result dataclass containing the aggregated scalar metrics for this epoch.
        """
        avg_s2s = self.train_start2start_ms.compute().item()
        self.train_start2start_ms.reset()
        logger.debug("Average train start->start (ms): %.2f", avg_s2s)
        # Global sync happens only at epoch end for accurate average loss logging

        avg_loss = self.train_loss_tracker.compute()
        val_results = self.val_metrics.compute()
        test_results = self.test_metrics.compute()
        val_mrr = self.val_mrr_tracker.compute().item()
        test_mrr = self.test_mrr_tracker.compute().item()
        val_auc_batch_mean = self.val_auc_batch_mean.compute().item()
        val_ap_batch_mean = self.val_ap_batch_mean.compute().item()
        test_auc_batch_mean = self.test_auc_batch_mean.compute().item()
        test_ap_batch_mean = self.test_ap_batch_mean.compute().item()

        self.val_metrics.reset()
        self.test_metrics.reset()
        self.train_loss_tracker.reset()
        self.val_mrr_tracker.reset()
        self.test_mrr_tracker.reset()
        self.val_auc_batch_mean.reset()
        self.val_ap_batch_mean.reset()
        self.test_auc_batch_mean.reset()
        self.test_ap_batch_mean.reset()

        res = Result(
            train_loss=avg_loss.item(),
            val_loss=val_results["val_loss"].item(),
            val_auc=val_results["val_auc"].item(),
            val_ap=val_results["val_ap"].item(),
            test_auc=test_results["test_auc"].item(),
            test_ap=test_results["test_ap"].item(),
            val_mrr=val_mrr,
            test_mrr=test_mrr,
            val_auc_batch_mean=val_auc_batch_mean,
            val_ap_batch_mean=val_ap_batch_mean,
            test_auc_batch_mean=test_auc_batch_mean,
            test_ap_batch_mean=test_ap_batch_mean,
        )
        self._last_train_start_t = None
        return res

    def train(self, meta: Meta) -> None:
        """Execute one training step for the given batch.

        Runs a forward pass through the assembled model, computes loss, runs
        backward, steps the optimizer, updates the running loss tracker, and
        frees per-batch tensors immediately afterward.

        Args:
            meta: Batch metadata carrying the flat graph, labels, and plan set.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.amp):
            logits = self.model(
                meta,
                mem_task_q=self.mem_task_q,
                box_task_q=self.box_task_q,
            )

            logits_flat = logits.squeeze(-1)
            n_pos = meta.n_src
            loss = self.criterion(
                logits_flat[:n_pos], meta.labels[:n_pos]
            ) + self.criterion(logits_flat[n_pos:], meta.labels[n_pos:])
        if self.mem_box_store is not None and not _debug_sync_membox_after_train_step():
            self.p2p_guard.arm()  # Arm guard: wait for P2P before the first AllReduce
        if _debug_sync_train_step_boundaries():
            torch.cuda.synchronize(self.device)
        self.grad_scaler.scale(loss).backward()
        self._maybe_append_grad_checksums(
            meta=meta,
            loss=loss,
            grad_scale=float(self.grad_scaler.get_scale()),
        )
        self._maybe_dump_step_tensors(meta=meta, phase="grad")

        grad_scale_before = float(self.grad_scaler.get_scale())
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if _debug_sync_train_step_boundaries():
            torch.cuda.synchronize(self.device)
        if self.mem_box_store is not None and _debug_sync_membox_after_train_step():
            self._flush_train_step_async_state()
        grad_scale_after = float(self.grad_scaler.get_scale())
        self._maybe_dump_step_tensors(meta=meta, phase="param")

        self._maybe_append_param_checksums(
            meta=meta,
            loss=loss,
            grad_scale_before=grad_scale_before,
            grad_scale_after=grad_scale_after,
        )

        self.train_loss_tracker.update(loss.detach().float())
        self._release_meta_tensors(meta)

    @torch.no_grad()
    def eval(self, meta: Meta) -> None:
        """Execute one evaluation step (val or test) for the given batch.

        Runs a forward pass in no-grad mode, applies sigmoid, and feeds
        predictions into the appropriate torchmetrics collectors.  For memory
        models, acquires the P2P semaphore to ensure no concurrent P2P I/O
        interferes with the eval state.

        Args:
            meta: Batch metadata; ``meta.split`` must be ``"val"`` or ``"test"``.
        """
        assert meta.labels is not None
        self.model.eval()

        with torch.amp.autocast("cuda", enabled=self.amp):
            logits = self.model(
                meta,
                mem_task_q=self.mem_task_q,
                box_task_q=self.box_task_q,
            )
        if self.mem_box_store is not None:
            self.p2p_sem.acquire()  # Ensure no other P2P task is in flight during eval

        y_pred = logits.float().sigmoid().squeeze(-1)
        if meta.n_neg == 1:
            batch_ap = binary_average_precision(
                y_pred, meta.labels.long(), thresholds=None
            )
            batch_auc = binary_auroc(y_pred, meta.labels.long(), thresholds=None)

        if meta.split == "val":
            logits_flat = logits.float().squeeze(-1)
            n_pos = meta.n_src
            loss = self.criterion(
                logits_flat[:n_pos], meta.labels[:n_pos]
            ) + self.criterion(logits_flat[n_pos:], meta.labels[n_pos:])

            if meta.n_neg == 1:
                self.val_metrics.update(
                    preds=y_pred,
                    target=meta.labels.long(),
                    value=loss,
                )
                self.val_ap_batch_mean.update(batch_ap)
                self.val_auc_batch_mean.update(batch_auc)
            else:
                # 100-negative evaluation logic (MRR)
                # logits_flat is [n_pos * (1 + n_neg)] where first chunk is positives
                y_pos = logits_flat[:n_pos].unsqueeze(1)
                y_neg = logits_flat[n_pos:].view(n_pos, meta.n_neg)
                scores = torch.cat([y_pos, y_neg], dim=1)
                ranks = (scores >= scores[:, 0:1]).sum(dim=1)
                mrr = (1.0 / ranks).mean()
                self.val_mrr_tracker.update(mrr)

        else:
            assert meta.split == "test"
            if meta.n_neg == 1:
                self.test_metrics.update(preds=y_pred, target=meta.labels.long())
                self.test_ap_batch_mean.update(batch_ap)
                self.test_auc_batch_mean.update(batch_auc)
            else:
                logits_flat = logits.float().squeeze(-1)
                n_pos = meta.n_src
                y_pos = logits_flat[:n_pos].unsqueeze(1)
                y_neg = logits_flat[n_pos:].view(n_pos, meta.n_neg)
                scores = torch.cat([y_pos, y_neg], dim=1)
                ranks = (scores >= scores[:, 0:1]).sum(dim=1)
                mrr = (1.0 / ranks).mean()
                self.test_mrr_tracker.update(mrr)

        self._release_meta_tensors(meta)

    def _release_meta_tensors(self, meta: Meta) -> None:
        """Drop per-batch GPU tensors ASAP to reduce delayed CUDA memory recycling."""

        meta.flat_graph = None
        meta.plan_set = None
        meta.labels = None
        meta.graph = None

    def start_all_threads(self) -> None:
        """Start all background pipeline threads.

        Must be called once before invoking ``run()``.  Threads are started in
        dependency order: H2D and Receiver first, then the optional memory/box
        sync threads (P2PQHandler, MemTaskHandler, MessageHandler, writers,
        DynamicAssembler) when a ``MemBoxStore`` is present.
        """
        self.h2d.start()
        self.receiver.start()
        if self.mem_box_store is not None:
            assert self.p2p_q_handler is not None
            assert self.mem_handler is not None
            assert self.box_handler is not None
            assert self.box_wr is not None
            assert self.mem_wr is not None
            assert self.dy_asm is not None
            self.p2p_q_handler.start()
            self.mem_handler.start()
            self.box_handler.start()
            self.box_wr.start()
            self.mem_wr.start()
            self.dy_asm.start()

    def run(self) -> None:
        """Main training loop: consume ``full_q`` and dispatch train/eval/epoch-end.

        Primes the H2D pipeline semaphore and injects an initial ``"nothing"``
        token into ``delta_queue`` for memory models before entering the main
        loop. The H2D credit manager keeps writeback credits ordered by batch,
        so later batches cannot consume shared-memory fetch credits before all
        earlier mem/box writebacks have finished. The loop terminates when
        ``None`` is dequeued.
        """
        self.start_all_threads()
        if self.mem_box_store is not None:
            p2p_world_size = dist.get_world_size(self.nccl_p2p)
            initial_slots = _initial_h2d_pipeline_slots(
                debug_force_cpu_freshness_single_worker=_debug_force_cpu_freshness_single_worker(),
                p2p_world_size=p2p_world_size,
            )
            if initial_slots % 2 != 0:
                raise ValueError(
                    "initial h2d slots must be even because each batch consumes "
                    f"two writeback credits, got {initial_slots}"
                )
            max_in_flight_batches = max(1, initial_slots // 2)
            set_max_in_flight_batches = getattr(
                self.h2d_sem, "set_max_in_flight_batches", None
            )
            if callable(set_max_in_flight_batches):
                set_max_in_flight_batches(max_in_flight_batches)
            else:
                for _ in range(initial_slots):
                    self.h2d_sem.release()
            assert self.delta_queue is not None
            self.delta_queue.put("nothing")

        def trace_handler(p):
            rank = dist.get_rank() if dist.is_initialized() else 0
            output_filename = f"trace_epoch2_rank_{rank}.json"
            p.export_chrome_trace(output_filename)
            logger.debug("Rank %s: Trace saved to %s", rank, output_filename)

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=None,  # None = manual step control
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
            with_flops=False,
        )

        # Number of steps captured in the target epoch
        epoch2_steps_recorded = 0
        total_wait_ms = 0.0
        wait_steps = 0
        while self.stop_ev.is_set() is False:
            # torch.cuda.synchronize()  # Ensure prior CUDA ops are done for accurate timing
            t_wait_start = time.perf_counter()
            full = self.full_q.get()

            t_wait_end = time.perf_counter()
            wait_ms = (t_wait_end - t_wait_start) * 1000.0

            total_wait_ms += wait_ms
            wait_steps += 1

            # if wait_ms > 5.0:  # Skip epoch 0 warmup
            #     logger.debug(
            #         f"[Rank {self.env.local_rank}] Data Starvation! Main loop waited {wait_ms:.2f}ms for full_q"
            #     )

            if isinstance(full, tuple):
                meta, ready_event = full
                torch.cuda.current_stream().wait_event(ready_event)
                if meta.split == "train":  # if training
                    # Throughput tracking: count events before train() releases labels
                    assert meta.labels is not None
                    self._throughput_batch_count += 1
                    if self._throughput_batch_count == 5:
                        self._throughput_start_time = time.perf_counter()
                    if self._throughput_batch_count >= 5:
                        self._throughput_events += meta.labels.numel() // (
                            1 + self.n_neg
                        )

                    # Profiling remains disabled by default.
                    should_profile = False
                    if should_profile:
                        # Start profiler on the first captured step of the target epoch
                        if epoch2_steps_recorded == 0:
                            prof.start()

                    with record_function("Model_Train_Step"):
                        now = time.perf_counter()
                        if self._last_train_start_t is not None:
                            interval_ms = (now - self._last_train_start_t) * 1000.0
                            self.train_start2start_ms.update(
                                torch.tensor(interval_ms, device=self.device)
                            )
                        self._last_train_start_t = now
                        self.train(meta)
                    if should_profile:
                        prof.step()  # Mark end of one profiling step
                        epoch2_steps_recorded += 1

                        # Stop and save after capturing 100 batches
                        if epoch2_steps_recorded >= 100:
                            prof.stop()

                elif meta.split in ["val", "test"]:  # if evaluating
                    self._last_train_start_t = None
                    self.eval(meta)
            if full == "end_of_epoch":
                avg_wait = total_wait_ms / max(1, wait_steps)
                logger.debug(
                    "[Rank %s] Epoch %s Avg Data Wait: %.2fms",
                    self.env.local_rank,
                    self.curr_epoch,
                    avg_wait,
                )
                # print total_wait_ms
                logger.debug(
                    "[Rank %s] Epoch %s Total Data Wait: %.2fms over %s waits",
                    self.env.local_rank,
                    self.curr_epoch,
                    total_wait_ms,
                    wait_steps,
                )

                total_wait_ms = 0.0
                wait_steps = 0

                result = self.on_train_epoch_end()

                self.result_q.put(result)
                val_ap = result.val_ap if result is not None else 0.0
                if val_ap > self.local_best_ap:
                    self.local_best_ap = val_ap
                    if not _debug_disable_best_ckpt_save():
                        self.save_model(os.path.join(self.save_to, "best.ckpt"))
                # dist.barrier() is redundant: on_train_epoch_end() compute() calls
                # already performed 7 NCCL all-reduces that synchronize all ranks.
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
                    self.mem_box_store.reset()
            if full is None:
                if (
                    self._throughput_start_time is not None
                    and self._throughput_events > 0
                ):
                    elapsed = time.perf_counter() - self._throughput_start_time
                    throughput = self._throughput_events / elapsed
                    print(
                        f"[GPU {self.env.local_rank}] Training throughput: "
                        f"{throughput:.1f} events/sec "
                        f"({self._throughput_events} events in {elapsed:.1f}s, "
                        f"excluding first 4 warm-up batches)"
                    )
                if self.mem_box_store is not None:
                    assert self.box_task_q is not None
                    assert self.mem_task_q is not None
                    assert self.mem_p2p_q is not None
                    self.box_task_q.put(None)
                    self.mem_task_q.put(None)
                    self.mem_p2p_q.put(None)
                break

    def _request_thread_shutdown(self) -> None:
        put_shutdown_sentinel(self.received_q, None)
        put_shutdown_sentinel(self.full_q, None)

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
        join_thread(self.receiver, timeout_s, logger)
        join_thread(self.mem_handler, timeout_s, logger)
        join_thread(self.mem_wr, timeout_s, logger)
        join_thread(self.box_wr, timeout_s, logger)
        join_thread(self.box_handler, timeout_s, logger)
        join_thread(self.p2p_q_handler, timeout_s, logger)
        join_thread(self.dy_asm, timeout_s, logger)

    def cleanup(self, graceful: bool = True, join_timeout_s: float = 5.0) -> None:
        """Best-effort worker shutdown for both normal and aborted runs."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.stop_ev.set()
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

    def _unwrap_model(self) -> AssembledModel:
        """Return the bare AssembledModel, stripping torch.compile and DDP wrappers."""
        model = self.model
        # torch.compile wraps the model in an OptimizedModule
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        # DDP wraps the model in a DistributedDataParallel
        if isinstance(model, DDP):
            model = model.module
        return cast(AssembledModel, model)

    def _maybe_append_param_checksums(
        self,
        *,
        meta: Meta,
        loss: torch.Tensor,
        grad_scale_before: float,
        grad_scale_after: float,
    ) -> None:
        path = _debug_param_checksums_path()
        if path is None:
            return

        batches = _debug_param_checksum_batches()
        if batches is not None and meta.mini_batch_idx not in batches:
            return

        total_sum = 0.0
        total_abs_sum = 0.0
        total_sq_sum = 0.0
        total_numel = 0
        param_records: list[dict[str, float | int | str]] = []

        with torch.no_grad():
            for name, value in self._unwrap_model().named_parameters():
                tensor = value.detach().float()
                sum_value = float(tensor.sum().item())
                abs_sum_value = float(tensor.abs().sum().item())
                sq_sum_value = float(tensor.square().sum().item())
                numel_value = int(tensor.numel())
                total_sum += sum_value
                total_abs_sum += abs_sum_value
                total_sq_sum += sq_sum_value
                total_numel += numel_value
                param_records.append(
                    {
                        "name": name,
                        "numel": numel_value,
                        "sum": sum_value,
                        "abs_sum": abs_sum_value,
                        "sq_sum": sq_sum_value,
                    }
                )

        record = {
            "mini_batch_idx": int(meta.mini_batch_idx),
            "cur_epoch": int(meta.cur_epoch),
            "split": meta.split,
            "cur_macro_of_split": int(meta.cur_macro_of_split),
            "cur_macro_of_epoch": int(meta.cur_macro_of_epoch),
            "loss": float(loss.detach().float().item()),
            "grad_scale_before": grad_scale_before,
            "grad_scale_after": grad_scale_after,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "num_params": total_numel,
            "param_sum": total_sum,
            "param_abs_sum": total_abs_sum,
            "param_sq_sum": total_sq_sum,
            "params": param_records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _maybe_append_grad_checksums(
        self,
        *,
        meta: Meta,
        loss: torch.Tensor,
        grad_scale: float,
    ) -> None:
        path = _debug_grad_checksums_path()
        if path is None:
            return

        batches = _debug_grad_checksum_batches()
        if batches is not None and meta.mini_batch_idx not in batches:
            return

        total_sum = 0.0
        total_abs_sum = 0.0
        total_sq_sum = 0.0
        total_numel = 0
        grad_records: list[dict[str, float | int | str]] = []

        with torch.no_grad():
            for name, value in self._unwrap_model().named_parameters():
                if value.grad is None:
                    continue
                tensor = value.grad.detach().float()
                sum_value = float(tensor.sum().item())
                abs_sum_value = float(tensor.abs().sum().item())
                sq_sum_value = float(tensor.square().sum().item())
                numel_value = int(tensor.numel())
                total_sum += sum_value
                total_abs_sum += abs_sum_value
                total_sq_sum += sq_sum_value
                total_numel += numel_value
                grad_records.append(
                    {
                        "name": name,
                        "numel": numel_value,
                        "sum": sum_value,
                        "abs_sum": abs_sum_value,
                        "sq_sum": sq_sum_value,
                    }
                )

        record = {
            "mini_batch_idx": int(meta.mini_batch_idx),
            "cur_epoch": int(meta.cur_epoch),
            "split": meta.split,
            "cur_macro_of_split": int(meta.cur_macro_of_split),
            "cur_macro_of_epoch": int(meta.cur_macro_of_epoch),
            "loss": float(loss.detach().float().item()),
            "grad_scale": grad_scale,
            "num_grad_params": total_numel,
            "grad_sum": total_sum,
            "grad_abs_sum": total_abs_sum,
            "grad_sq_sum": total_sq_sum,
            "grads": grad_records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _maybe_dump_step_tensors(
        self,
        *,
        meta: Meta,
        phase: Literal["grad", "param"],
    ) -> None:
        dump_dir = _debug_step_tensor_dump_dir()
        if dump_dir is None:
            return

        batches = _debug_step_tensor_dump_batches()
        if batches is not None and meta.mini_batch_idx not in batches:
            return

        names = _debug_step_tensor_dump_names()
        if names is None or len(names) == 0:
            return

        dump_dir.mkdir(parents=True, exist_ok=True)
        for name, value in self._unwrap_model().named_parameters():
            if name not in names:
                continue
            if phase == "grad":
                if value.grad is None:
                    continue
                tensor = value.grad.detach()
            else:
                tensor = value.detach()
            safe_name = name.replace(".", "__")
            torch.save(
                {
                    "mini_batch_idx": int(meta.mini_batch_idx),
                    "cur_epoch": int(meta.cur_epoch),
                    "split": meta.split,
                    "phase": phase,
                    "name": name,
                    "tensor": tensor.cpu(),
                },
                dump_dir / f"batch_{meta.mini_batch_idx}_{phase}_{safe_name}.pt",
            )

    # Public model save/load interface
    def save_model(self, path: str) -> None:
        """Save a consistent CPU snapshot of the unwrapped model (rank 0 only).

        The DDP wrapper is stripped so the checkpoint can be loaded without a
        distributed context. The snapshot is materialized on CPU in the caller
        thread before ``torch.save`` so training never overlaps with a
        background thread reading live GPU parameters.

        Args:
            path: File path where the state dict will be written.
        """
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if rank == 0:
            torch.cuda.synchronize(self.device)
            with torch.no_grad():
                cpu_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self._unwrap_model().state_dict().items()
                }
            torch.save(cpu_state_dict, path)

    def load_model(self, path: str) -> None:
        """Load model weights into the existing model / DDP wrapper in place.

        A barrier before loading ensures the file is fully written by rank 0
        before any rank reads it.  A second barrier after loading keeps all
        ranks in sync.

        Args:
            path: File path to the state dict checkpoint.
        """
        # Barrier ensures the file is fully written before any rank reads it
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.nccl)

        state = torch.load(path, map_location="cpu")

        bare_model = self._unwrap_model()
        bare_model.load_state_dict(state, strict=True)
        bare_model.to(self.device)

        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.nccl)

    def load_memory(self, memory_state: dict) -> None:
        """Restore the model's in-memory state from a previously saved dict.

        Delegates to ``AssembledModel.load_memory_state`` via the DDP module
        unwrap.  No-op when ``mem_box_store`` is absent.

        Args:
            memory_state: Dict produced by a prior ``save_memory_state()`` call.
        """
        if self.mem_box_store is not None:
            self._unwrap_model().load_memory_state(memory_state)
