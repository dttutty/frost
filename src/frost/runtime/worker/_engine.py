"""Worker engine for distributed Temporal GNN training.

Implements the GPU-resident Worker process that drives the train/eval loop.
Each Worker owns one GPU rank, a chain of pipeline threads (Receiver, H2D,
optional DynamicAssembler), and optional P2P memory-sync threads when a
MemBoxStore is configured.
"""
import logging
import threading
import time
from queue import Queue as Q
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from frost.aliases.syncer import (
    BoxP2PQ,
    BoxTaskQ,
    BoxWrQ,
    MemP2PQ,
    MemTaskQ,
    MemWrQ,
)
from frost.aliases.worker import AssembledQ, DeltaQ, H2DOutQ, ReceivedQ
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.models._assembled_model import AssembledModel
from frost.protocols.mem_updater_proto import MemoryUpdaterProto
from frost.protocols.postman_proto import PostmanProto
from frost.runtime.worker._dynamic_assembler import DynamicAssembler
from frost.runtime.worker._host_to_device import H2D
from frost.runtime.worker._receiver import Receiver as Receiver
from frost.runtime.worker.p2p_syncer._handlers import BoxTaskHandler, MemTaskHandler
from frost.runtime.worker.p2p_syncer._writer import SharedMemWriter
from frost.runtime.worker.p2p_syncer._p2p_q_handler import P2PQHandler
from frost.shm.mem_mgr import MemBoxStore
from frost.shm.ring_buf import RingBuf

logger = logging.getLogger(__name__)


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
    optional P2P memory-sync threads (MemTaskHandler, BoxTaskHandler,
    SharedMemWriter, P2PQHandler) when a ``MemBoxStore`` is present.

    Attributes:
        received_q: Queue carrying ``Meta`` objects freshly received from the host.
        delta_q: Queue carrying P2P delta tuples; only used when mem_box_store is present.
        h2d_out_q: Intermediate queue between H2D and DynamicAssembler.
        full_q: Queue of fully-assembled ``(Meta, CudaEvent)`` tuples ready for the model.
        mem_task_q: Queue of memory update tasks posted by AssembledModel.
        box_task_q: Queue of mailbox update tasks posted by AssembledModel.
        mem_p2p_q: Queue of P2P memory descriptors produced by MemTaskHandler.
        box_p2p_q: Queue of P2P mailbox descriptors produced by BoxTaskHandler.
        mem_wr_q: Queue of write-back requests for the node memory store.
        box_wr_q: Queue of write-back requests for the mailbox store.
        dy_asm: DynamicAssembler thread (memory-model path only).
        mem_handler: MemTaskHandler thread (memory-model path only).
        mem_wr: SharedMemWriter thread for node memory (memory-model path only).
        box_handler: BoxTaskHandler thread (memory-model path only).
        box_wr: SharedMemWriter thread for mailbox (memory-model path only).
        p2p_q_handler: P2PQHandler thread (memory-model path only).
        model: AssembledModel wrapped in DDP.
        result_q: Single-element queue used to pass epoch results back to the Receiver.
    """

    # --- Pipeline queues ---
    received_q: ReceivedQ
    delta_q: DeltaQ | None = None
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
    box_handler: BoxTaskHandler | None = None
    box_wr: SharedMemWriter | None = None
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
        self.lr = lr
        self.curr_epoch = 0
        self.local_best_auc = 0.0
        self._last_train_start_t: float | None = None

        # --- Synchronization primitives ---
        self.stop_ev = threading.Event()
        self.h2d_sem: threading.Semaphore = threading.Semaphore(0)
        self.p2p_sem = threading.Semaphore(0)

        self._init_pipeline(ring_buf, nfeat, efeat)
        self._init_model(
            encoder, mem_updater, predictor, postman, criterion, optimizer, ring_buf
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
        self.result_q = Q(maxsize=1)
        self.received_q = Q(maxsize=1 if has_mem else 2)
        self.full_q = Q(maxsize=1 if has_mem else 2)

        self.receiver = Receiver(
            env=self.env,
            received_q=self.received_q,
            result_q=self.result_q,
            stop_ev=self.stop_ev,
            ring_buf=ring_buf,
        )

        if has_mem:
            assert self.mem_box_store is not None
            self.delta_q = Q(maxsize=1)
            self.h2d_out_q = Q(maxsize=1)
            self.dy_asm = DynamicAssembler(
                mem_box_store=self.mem_box_store,
                h2d_out_q=self.h2d_out_q,
                delta_q=self.delta_q,
                full_q=self.full_q,
                stop_ev=self.stop_ev,
                device=self.device,
                h2d_sem=self.h2d_sem,
            )
            self._init_mem_box_threads()
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
        assert self.delta_q is not None
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
        self.box_handler = BoxTaskHandler(
            h2d_sem=self.h2d_sem,
            box_task_q=self.box_task_q,
            box_p2p_q=self.box_p2p_q,
            stop_ev=self.stop_ev,
            mail_combine=self.mail_combine,
            device=self.device,
            box_wr_q=self.box_wr_q,
            postman=self.postman,
        )
        self.box_wr = SharedMemWriter(
            shared_data=self.mem_box_store.box,
            shared_ts=self.mem_box_store.box_ts,
            h2d_sem=self.h2d_sem,
            stop_ev=self.stop_ev,
            wr_q=self.box_wr_q,
            device=self.device,
            name="box_wr",
        )
        self.p2p_q_handler = P2PQHandler(
            env=self.env,
            mem_p2p_q=self.mem_p2p_q,
            box_p2p_q=self.box_p2p_q,
            nccl_p2p=self.nccl_p2p,
            p2p_sem=self.p2p_sem,
            stop_ev=self.stop_ev,
            delta_q=self.delta_q,
            device=self.device,
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
        self.criterion = criterion

    def _init_metrics(self) -> None:
        """Create torchmetrics trackers for training and evaluation."""
        self.train_start2start_ms = MeanMetric(process_group=self.nccl).cuda()
        self.train_loss_tracker = MeanMetric(process_group=self.nccl).cuda()

        self.val_metrics = MetricCollection(
            {
                "val_loss": MeanMetric(process_group=self.nccl),
                "val_auc": BinaryAUROC(thresholds=10000, process_group=self.nccl),
                "val_ap": BinaryAveragePrecision(
                    thresholds=10000, process_group=self.nccl
                ),
            }
        ).cuda()

        self.test_metrics = MetricCollection(
            {
                "test_auc": BinaryAUROC(thresholds=10000, process_group=self.nccl),
                "test_ap": BinaryAveragePrecision(
                    thresholds=10000, process_group=self.nccl
                ),
            }
        ).cuda()

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

        self.val_metrics.reset()
        self.test_metrics.reset()
        self.train_loss_tracker.reset()

        res = Result(
            train_loss=avg_loss.item(),
            val_loss=val_results["val_loss"].item(),
            val_auc=val_results["val_auc"].item(),
            val_ap=val_results["val_ap"].item(),
            test_auc=test_results["test_auc"].item(),
            test_ap=test_results["test_ap"].item(),
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
        logits = self.model(
            meta,
            mem_task_q=self.mem_task_q,
            box_task_q=self.box_task_q,
        )

        loss = self.criterion(logits.squeeze(-1), meta.labels)
        if self.mem_box_store is not None:
            self.p2p_guard.arm()  # Arm guard: wait for P2P before the first AllReduce
        loss.backward()

        self.optimizer.step()

        self.train_loss_tracker.update(loss.detach())
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

        logits = self.model(
            meta,
            mem_task_q=self.mem_task_q,
            box_task_q=self.box_task_q,
        )
        if self.mem_box_store is not None:
            self.p2p_sem.acquire()  # Ensure no other P2P task is in flight during eval

        y_pred = logits.sigmoid().squeeze(-1)

        if meta.split == "val":
            loss = self.criterion(logits.squeeze(-1), meta.labels)

            # MeanMetric expects 'value'; classification metrics expect 'preds' and 'target'
            self.val_metrics.update(
                preds=y_pred,
                target=meta.labels.long(),
                value=loss,
            )

        else:
            assert meta.split == "test"
            self.test_metrics.update(preds=y_pred, target=meta.labels.long())

        self._release_meta_tensors(meta)

    def _release_meta_tensors(self, meta: Meta) -> None:
        """Drop per-batch GPU tensors ASAP to reduce delayed CUDA memory recycling."""

        meta.flat_graph = None
        meta.plan_set = None
        meta.labels = None
        meta.graph = None
        meta = None

    def start_all_threads(self) -> None:
        """Start all background pipeline threads.

        Must be called once before invoking ``run()``.  Threads are started in
        dependency order: H2D and Receiver first, then the optional memory/box
        sync threads (P2PQHandler, MemTaskHandler, BoxTaskHandler, writers,
        DynamicAssembler) when a ``MemBoxStore`` is present.
        """
        self.h2d.start()
        self.receiver.start()
        if self.mem_box_store is not None:
            self.p2p_q_handler.start()
            self.mem_handler.start()
            self.box_handler.start()
            self.box_wr.start()
            self.mem_wr.start()
            self.dy_asm.start()

    def run(self) -> None:
        """Main training loop: consume ``full_q`` and dispatch train/eval/epoch-end.

        Primes the H2D pipeline semaphore (4 releases) and injects an initial
        ``"nothing"`` token into ``delta_q`` for memory models before entering the
        main loop.  The loop terminates when ``None`` is dequeued.
        """
        self.start_all_threads()
        if self.mem_box_store is not None:
            # Pre-release 4 slots to prime the H2D pipeline (2 batches in flight)
            self.h2d_sem.release()
            self.h2d_sem.release()
            self.h2d_sem.release()
            self.h2d_sem.release()
            assert self.delta_q is not None
            self.delta_q.put("nothing")

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
        # Guard flag to prevent duplicate profiler start/stop
        profiling_done = False

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
                    # should_profile = self.curr_epoch == 1 and not profiling_done
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
                            profiling_done = True

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
                val_auc = result.val_auc if result is not None else 0.0
                if val_auc > self.local_best_auc:
                    self.local_best_auc = val_auc
                    # Save asynchronously so rank-0 disk I/O doesn't stall the loop
                    threading.Thread(
                        target=self.save_model, args=("best.ckpt",), daemon=True
                    ).start()
                # dist.barrier() is redundant: on_train_epoch_end() compute() calls
                # already performed 7 NCCL all-reduces that synchronize all ranks.
                self.curr_epoch += 1
                if self.mem_box_store:
                    self.mem_box_store.reset()
            if full is None:
                if self.mem_box_store is not None:
                    assert self.box_task_q is not None
                    assert self.mem_task_q is not None
                    assert self.mem_p2p_q is not None
                    self.box_task_q.put(None)
                    self.mem_task_q.put(None)
                    self.mem_p2p_q.put(None)
                break

    def cleanup(self) -> None:
        """Join all background threads and tear down the distributed process group.

        Calls ``join()`` on every thread in reverse dependency order, then
        issues a ``dist.barrier()`` to synchronize all ranks before destroying
        the process group.
        """
        self.h2d.join()
        self.receiver.join()
        if self.mem_handler:
            self.mem_handler.join()
        if self.mem_wr:
            self.mem_wr.join()
        if self.box_wr:
            self.box_wr.join()

        if self.box_handler:
            self.box_handler.join()
        if self.p2p_q_handler:
            self.p2p_q_handler.join()
        if self.dy_asm:
            self.dy_asm.join()
        dist.barrier()
        dist.destroy_process_group()

    # Public model save/load interface
    def save_model(self, path: str) -> None:
        """Save the unwrapped model weights to disk (rank 0 only).

        The DDP wrapper is stripped so the checkpoint can be loaded without
        a distributed context.

        Args:
            path: File path where the state dict will be written.
        """
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if rank == 0:
            # Save the unwrapped model weights so the checkpoint can be loaded without DDP
            model = self.model.module if isinstance(self.model, DDP) else self.model
            torch.save(model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load model weights from a checkpoint and re-wrap with DDP.

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

        # Unwrap DDP before loading because the checkpoint was saved from the bare model
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(state, strict=True)

        model = model.to(self.device)

        # Re-wrap with DDP
        self.model = DDP(model, device_ids=[0])

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
            self.model.module.load_memory_state(memory_state)
