"""Worker engine for distributed Temporal GNN training.

Implements the GPU-resident Worker process that drives the train/eval loop.
Each Worker owns one GPU rank, a chain of pipeline threads (Receiver, H2D,
optional DynamicAssembler), and optional P2P memory-sync threads when a
MemBoxStore is configured.
"""

import logging
import json
import os
import threading
import time
from queue import Queue as Q
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP

from frost.ablation import DynamicStateAblation
from frost.runtime.worker.types import (
    BoxP2PQ,
    BoxTaskQ,
    BoxWrQ,
    MemP2PQ,
    MemTaskQ,
    MemWrQ,
)
from frost.runtime.worker.types import AssembledQ, DeltaQueue, H2DOutQ, ReceivedQ
from frost.datatypes.env import Env
from frost.datatypes.result import Result
from frost.models._assembled_model import AssembledModel
from frost.models.protocols import MemoryUpdaterProto
from frost.models.protocols import PostmanProto
from frost.runtime._thread_base import (
    destroy_process_group,
)
from frost.runtime.worker._direct_writeback import DirectDynamicStateWriter
from frost.runtime.worker._checkpoint import WorkerCheckpointMixin
from frost.runtime.worker._dynamic_assembler import DynamicAssembler
from frost.runtime.worker._h2d_credit import BatchAwareCreditManager
from frost.runtime.worker._pipeline import (
    WorkerPipelineMixin,
    _initial_h2d_pipeline_slots,
)
from frost.runtime.worker._receiver import Receiver as Receiver
from frost.runtime.worker._trainer import WorkerTrainerMixin
from frost.runtime.worker._telemetry import (
    _discover_sibling_rank_pids,
    _get_bool_env,
    _get_non_negative_int_env,
    _get_optional_non_negative_int_env,
    _sample_total_cpu_time_s,
    logical_cpu_count,
)
from frost.runtime.worker.p2p_syncer._p2p_comm import P2PQHandler
from frost.runtime.worker.p2p_syncer._task_handlers import (
    MemTaskHandler,
    MessageHandler,
)
from frost.runtime.worker.p2p_syncer._writer import BoxMailWriter, SharedMemWriter
from frost.shm.mem_mgr import MemBoxStore
from frost.shm.ring_buf import RingBuf

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
    direct_state_writer: DirectDynamicStateWriter | None = None

    model: AssembledModel | DDP
    optimizer: torch.optim.Optimizer
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
        dynamic_state_ablation: DynamicStateAblation | None = None,
        compile_model: bool = True,
        amp: bool = False,
        no_memory_queue_size: int = 4,
        cpu_benchmark_pid_list: list[int] | None = None,
        eval_metric: Literal["ap", "mrr"] = "ap",
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
        self.dynamic_state_ablation = (
            dynamic_state_ablation
            if dynamic_state_ablation is not None
            else DynamicStateAblation(name="baseline")
        )
        self.compile_model = compile_model
        self.amp = amp
        self.lr = lr
        if eval_metric not in {"ap", "mrr"}:
            raise ValueError(
                f"Unsupported eval_metric={eval_metric!r}; expected 'ap' or 'mrr'."
            )
        self.eval_metric = eval_metric
        if no_memory_queue_size < 1:
            raise ValueError("no_memory_queue_size must be >= 1.")
        self.no_memory_queue_size = no_memory_queue_size
        self.curr_epoch = 0
        self.local_best_metric = float("-inf")

        # Throughput benchmark controls:
        # - legacy mode: warmup is applied at the start of every epoch
        # - optional global mode: warmup is applied once across the whole run
        # - measurement stops after target batches when target > 0
        self._throughput_warmup_batches_per_epoch = _get_non_negative_int_env(
            "FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH",
            _get_non_negative_int_env("FROST_THROUGHPUT_WARMUP_BATCHES", 4),
        )
        self._throughput_warmup_batches_total = _get_optional_non_negative_int_env(
            "FROST_THROUGHPUT_WARMUP_BATCHES_TOTAL"
        )
        self._throughput_measure_batches_target = _get_non_negative_int_env(
            "FROST_THROUGHPUT_MEASURE_BATCHES",
            0,
        )
        self._throughput_epoch_train_batch_count = 0
        self._throughput_total_train_batch_count = 0
        self._throughput_measured_batches = 0
        self._throughput_events = 0
        self._throughput_wall_intervals_s: list[float] = []
        self._throughput_gpu_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._throughput_prev_start_event: torch.cuda.Event | None = None
        self._throughput_prev_start_wall_s: float | None = None
        self._throughput_prev_start_wall_unix_s: float | None = None
        self._throughput_prev_batch_events: int | None = None
        self._throughput_window_start_unix_s: float | None = None
        self._throughput_window_end_unix_s: float | None = None
        self._throughput_last_completed_window_start_unix_s: float | None = None
        self._throughput_last_completed_window_end_unix_s: float | None = None
        self._throughput_last_completed_measured_batches = 0
        self._throughput_interval_start_unix_s: list[float] = []
        self._throughput_interval_end_unix_s: list[float] = []
        self._memory_reserved_mib: list[float] = []
        self._memory_allocated_mib: list[float] = []
        self._gpu_util_curve_enabled = _get_bool_env(
            "FROST_GPU_UTIL_CURVE_BENCH", False
        )
        self._cpu_util_enabled = _get_bool_env("FROST_CPU_UTIL_BENCH", False)
        self._cpu_pid_list: list[int] = []
        self._cpu_logical_count = 0
        self._cpu_prev_total_s: float | None = None
        self._cpu_used_cores: list[float] = []
        self._cpu_utilization_pct: list[float] = []
        self._cpu_wall_interval_s: list[float] = []
        if self._cpu_util_enabled:
            if self.env.local_rank == 0:
                self._cpu_pid_list = (
                    [int(pid) for pid in cpu_benchmark_pid_list]
                    if cpu_benchmark_pid_list
                    else _discover_sibling_rank_pids()
                )
                self._cpu_logical_count = logical_cpu_count()
        self._shutdown_done = False

        # --- Synchronization primitives ---
        self.stop_ev = threading.Event()
        self.h2d_sem = BatchAwareCreditManager()
        self.p2p_sem = threading.Semaphore(0)
        if (
            self.mem_box_store is not None
            and self.dynamic_state_ablation.uses_direct_cpu_writeback
        ):
            self.direct_state_writer = DirectDynamicStateWriter(
                mem_box_store=self.mem_box_store,
                h2d_sem=self.h2d_sem,
            )

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

    @torch.no_grad()
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
            if self.dynamic_state_ablation.max_in_flight_batches is not None:
                max_in_flight_batches = (
                    self.dynamic_state_ablation.max_in_flight_batches
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
            set_max_in_flight_batches = getattr(
                self.h2d_sem, "set_max_in_flight_batches", None
            )
            if callable(set_max_in_flight_batches):
                set_max_in_flight_batches(max_in_flight_batches)
            else:
                for _ in range(initial_slots):
                    self.h2d_sem.release()
            if self.delta_queue is not None:
                self.delta_queue.put("nothing")
        while self.stop_ev.is_set() is False:
            full = self.full_q.get()

            if isinstance(full, tuple):
                meta, ready_event = full
                torch.cuda.current_stream().wait_event(ready_event)
                if meta.split == "train":  # if training
                    # Throughput tracking on train start->start intervals.
                    assert meta.labels is not None
                    self._throughput_epoch_train_batch_count += 1
                    self._throughput_total_train_batch_count += 1
                    batch_events = meta.n_src
                    curr_start_event = torch.cuda.Event(enable_timing=True)
                    curr_start_event.record(torch.cuda.current_stream())
                    curr_start_wall_s = time.perf_counter()
                    curr_start_wall_unix_s = time.time()
                    curr_cpu_total_s: float | None = None
                    if self._cpu_util_enabled and self.env.local_rank == 0:
                        curr_cpu_total_s, self._cpu_pid_list = _sample_total_cpu_time_s(
                            self._cpu_pid_list
                        )

                    target = self._throughput_measure_batches_target
                    can_measure_more = (
                        target == 0 or self._throughput_measured_batches < target
                    )
                    interval_count_so_far = self._throughput_total_train_batch_count - 1
                    if self._throughput_warmup_batches_total is not None:
                        past_warmup = (
                            interval_count_so_far
                            > self._throughput_warmup_batches_total
                        )
                    else:
                        past_warmup = (
                            self._throughput_epoch_train_batch_count - 1
                            > self._throughput_warmup_batches_per_epoch
                        )
                    # Interval k spans [start_{k-1}, start_k), i.e. it corresponds to
                    # the previous batch. Skip either the per-epoch warmup
                    # (legacy behavior) or the one-time global warmup.
                    should_measure_interval = (
                        can_measure_more
                        and self._throughput_prev_start_event is not None
                        and self._throughput_prev_start_wall_s is not None
                        and self._throughput_prev_batch_events is not None
                        and past_warmup
                    )
                    if should_measure_interval:
                        prev_event = self._throughput_prev_start_event
                        prev_wall_s = self._throughput_prev_start_wall_s
                        prev_batch_events = self._throughput_prev_batch_events
                        assert prev_event is not None
                        assert prev_wall_s is not None
                        assert prev_batch_events is not None
                        self._throughput_gpu_pairs.append(
                            (prev_event, curr_start_event)
                        )
                        self._throughput_wall_intervals_s.append(
                            curr_start_wall_s - prev_wall_s
                        )
                        if self._throughput_prev_start_wall_unix_s is not None:
                            self._throughput_interval_start_unix_s.append(
                                self._throughput_prev_start_wall_unix_s
                            )
                            self._throughput_interval_end_unix_s.append(
                                curr_start_wall_unix_s
                            )
                        self._throughput_events += prev_batch_events
                        self._throughput_measured_batches += 1
                        if self._throughput_window_start_unix_s is None:
                            self._throughput_window_start_unix_s = (
                                self._throughput_prev_start_wall_unix_s
                            )
                        self._throughput_window_end_unix_s = curr_start_wall_unix_s
                        if (
                            self._cpu_util_enabled
                            and self.env.local_rank == 0
                            and curr_cpu_total_s is not None
                            and self._cpu_prev_total_s is not None
                        ):
                            wall_interval_s = curr_start_wall_s - prev_wall_s
                            if wall_interval_s > 0:
                                cpu_delta_s = max(
                                    0.0,
                                    curr_cpu_total_s - self._cpu_prev_total_s,
                                )
                                used_cores = cpu_delta_s / wall_interval_s
                                util_pct = used_cores / self._cpu_logical_count * 100.0
                                self._cpu_used_cores.append(used_cores)
                                self._cpu_utilization_pct.append(util_pct)
                                self._cpu_wall_interval_s.append(wall_interval_s)
                    self._throughput_prev_start_event = curr_start_event
                    self._throughput_prev_start_wall_s = curr_start_wall_s
                    self._throughput_prev_start_wall_unix_s = curr_start_wall_unix_s
                    self._throughput_prev_batch_events = batch_events
                    if self._cpu_util_enabled and self.env.local_rank == 0:
                        self._cpu_prev_total_s = curr_cpu_total_s
                    self.train(meta)
                    can_measure_memory = (
                        self._throughput_measure_batches_target == 0
                        or len(self._memory_reserved_mib)
                        < self._throughput_measure_batches_target
                    )
                    if self._throughput_warmup_batches_total is not None:
                        memory_past_warmup = (
                            self._throughput_total_train_batch_count
                            > self._throughput_warmup_batches_total
                        )
                    else:
                        memory_past_warmup = (
                            self._throughput_epoch_train_batch_count
                            > self._throughput_warmup_batches_per_epoch
                        )
                    if can_measure_memory and memory_past_warmup:
                        self._memory_reserved_mib.append(
                            torch.cuda.memory_reserved(self.device) / (1024**2)
                        )
                        self._memory_allocated_mib.append(
                            torch.cuda.memory_allocated(self.device) / (1024**2)
                        )

                elif meta.split in ["val", "test"]:  # if evaluating
                    self.eval(meta)
            if full == "end_of_epoch":
                self._throughput_epoch_train_batch_count = 0
                self._throughput_last_completed_window_start_unix_s = (
                    self._throughput_window_start_unix_s
                )
                self._throughput_last_completed_window_end_unix_s = (
                    self._throughput_window_end_unix_s
                )
                self._throughput_last_completed_measured_batches = (
                    self._throughput_measured_batches
                )
                self._throughput_prev_start_event = None
                self._throughput_prev_start_wall_s = None
                self._throughput_prev_start_wall_unix_s = None
                self._throughput_prev_batch_events = None
                self._throughput_window_start_unix_s = None
                self._throughput_window_end_unix_s = None
                if self._cpu_util_enabled and self.env.local_rank == 0:
                    self._cpu_prev_total_s = None
                result = self.on_train_epoch_end()

                self.result_q.put(result)
                selection_metric = (
                    result.val_ap if self.eval_metric == "ap" else result.val_mrr
                )
                if selection_metric > self.local_best_metric:
                    self.local_best_metric = selection_metric
                    self.save_model(os.path.join(self.save_to, "best.ckpt"))
                # Metric compute() calls above synchronize all worker ranks, so
                # an additional barrier here would be redundant.
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
                wall_elapsed_s = sum(self._throughput_wall_intervals_s)
                gpu_elapsed_s = 0.0
                if self._throughput_gpu_pairs:
                    torch.cuda.synchronize()
                    gpu_elapsed_s = (
                        sum(
                            ev_prev.elapsed_time(ev_curr)
                            for ev_prev, ev_curr in self._throughput_gpu_pairs
                        )
                        / 1000.0
                    )
                wall_throughput = (
                    self._throughput_events / wall_elapsed_s
                    if self._throughput_events > 0 and wall_elapsed_s > 0
                    else 0.0
                )
                gpu_throughput = (
                    self._throughput_events / gpu_elapsed_s
                    if self._throughput_events > 0 and gpu_elapsed_s > 0
                    else 0.0
                )
                if wall_throughput > 0:
                    print(
                        f"TRAIN_THROUGHPUT_PER_GPU rank={self.env.local_rank} "
                        f"events_per_sec_per_gpu={wall_throughput:.1f} "
                        f"events_per_sec_per_gpu_gpu_clock={gpu_throughput:.1f} "
                        f"events={self._throughput_events} "
                        f"elapsed_s={wall_elapsed_s:.3f} "
                        f"gpu_elapsed_s={gpu_elapsed_s:.3f} "
                        "warmup_batches_per_epoch="
                        f"{self._throughput_warmup_batches_per_epoch} "
                        f"warmup_batches_total={self._throughput_warmup_batches_total} "
                        f"measured_batches={self._throughput_measured_batches} "
                        f"measure_target={self._throughput_measure_batches_target}"
                    )
                window_start_unix_s = (
                    self._throughput_window_start_unix_s
                    if self._throughput_window_start_unix_s is not None
                    else self._throughput_last_completed_window_start_unix_s
                )
                window_end_unix_s = (
                    self._throughput_window_end_unix_s
                    if self._throughput_window_end_unix_s is not None
                    else self._throughput_last_completed_window_end_unix_s
                )
                window_measured_batches = (
                    self._throughput_measured_batches
                    if self._throughput_measured_batches > 0
                    else self._throughput_last_completed_measured_batches
                )
                if window_start_unix_s is not None and window_end_unix_s is not None:
                    window_payload = {
                        "system": "frost",
                        "rank": self.env.local_rank,
                        "window_start_unix_s": window_start_unix_s,
                        "window_end_unix_s": window_end_unix_s,
                        "measured_batches": window_measured_batches,
                    }
                    print(
                        "APPENDIX_TRAIN_WINDOW "
                        f"{json.dumps(window_payload, sort_keys=True)}"
                    )
                if (
                    self._throughput_interval_start_unix_s
                    and self._throughput_interval_end_unix_s
                ):
                    interval_payload = {
                        "system": "frost",
                        "rank": self.env.local_rank,
                        "measured_batches": len(self._throughput_interval_start_unix_s),
                        "interval_start_unix_s": self._throughput_interval_start_unix_s,
                        "interval_end_unix_s": self._throughput_interval_end_unix_s,
                    }
                    print(
                        "APPENDIX_TRAIN_INTERVALS "
                        f"{json.dumps(interval_payload, sort_keys=True)}"
                    )
                if self._gpu_util_curve_enabled and self._throughput_wall_intervals_s:
                    gpu_ms_curve = [
                        ev_prev.elapsed_time(ev_curr)
                        for ev_prev, ev_curr in self._throughput_gpu_pairs
                    ]
                    wall_ms_curve = [
                        interval_s * 1000.0
                        for interval_s in self._throughput_wall_intervals_s
                    ]
                    gpu_busy_pct_curve = [
                        (gpu_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0
                        for gpu_ms, wall_ms in zip(gpu_ms_curve, wall_ms_curve)
                    ]
                    gpu_payload: dict[str, int | float | str | list[float]] = {
                        "system": "frost",
                        "rank": self.env.local_rank,
                        "measured_batches": len(gpu_busy_pct_curve),
                        "gpu_ms_curve": [round(v, 4) for v in gpu_ms_curve],
                        "wall_ms_curve": [round(v, 4) for v in wall_ms_curve],
                        "gpu_busy_pct_curve": [round(v, 4) for v in gpu_busy_pct_curve],
                    }
                    if gpu_busy_pct_curve:
                        ordered_gpu_busy = sorted(gpu_busy_pct_curve)
                        n_gpu_busy = len(ordered_gpu_busy)
                        gpu_payload.update(
                            {
                                "gpu_busy_pct_mean": sum(ordered_gpu_busy) / n_gpu_busy,
                                "gpu_busy_pct_p50": ordered_gpu_busy[
                                    int(0.50 * (n_gpu_busy - 1))
                                ],
                                "gpu_busy_pct_p90": ordered_gpu_busy[
                                    int(0.90 * (n_gpu_busy - 1))
                                ],
                                "gpu_busy_pct_max": ordered_gpu_busy[-1],
                            }
                        )
                    print(
                        "APPENDIX_TRAIN_GPU_UTIL_CURVE "
                        f"{json.dumps(gpu_payload, sort_keys=True)}"
                    )
                memory_payload: dict[str, int | float | str | None] = {
                    "system": "frost",
                    "rank": self.env.local_rank,
                    "warmup_batches_per_epoch": self._throughput_warmup_batches_per_epoch,
                    "warmup_batches_total": self._throughput_warmup_batches_total,
                    "measure_batches_target": self._throughput_measure_batches_target,
                    "memory_measured_batches": len(self._memory_reserved_mib),
                }
                if self._memory_reserved_mib:
                    ordered_reserved = sorted(self._memory_reserved_mib)
                    n_reserved = len(ordered_reserved)
                    memory_payload.update(
                        {
                            "reserved_mib_mean": sum(ordered_reserved) / n_reserved,
                            "reserved_mib_p50": ordered_reserved[
                                int(0.50 * (n_reserved - 1))
                            ],
                            "reserved_mib_p90": ordered_reserved[
                                int(0.90 * (n_reserved - 1))
                            ],
                            "reserved_mib_max": ordered_reserved[-1],
                        }
                    )
                if self._memory_allocated_mib:
                    ordered_allocated = sorted(self._memory_allocated_mib)
                    n_allocated = len(ordered_allocated)
                    memory_payload.update(
                        {
                            "allocated_mib_mean": sum(ordered_allocated) / n_allocated,
                            "allocated_mib_p50": ordered_allocated[
                                int(0.50 * (n_allocated - 1))
                            ],
                            "allocated_mib_p90": ordered_allocated[
                                int(0.90 * (n_allocated - 1))
                            ],
                            "allocated_mib_max": ordered_allocated[-1],
                        }
                    )
                print(
                    f"APPENDIX_TRAIN_MEMORY_SUMMARY {json.dumps(memory_payload, sort_keys=True)}"
                )
                if self._memory_reserved_mib or self._memory_allocated_mib:
                    memory_curve_payload: dict[
                        str, int | float | str | list[float] | None
                    ] = {
                        "system": "frost",
                        "rank": self.env.local_rank,
                        "warmup_batches_per_epoch": self._throughput_warmup_batches_per_epoch,
                        "warmup_batches_total": self._throughput_warmup_batches_total,
                        "measure_batches_target": self._throughput_measure_batches_target,
                        "measured_batches": len(self._memory_reserved_mib),
                        "reserved_mib_curve": [
                            round(v, 4) for v in self._memory_reserved_mib
                        ],
                        "allocated_mib_curve": [
                            round(v, 4) for v in self._memory_allocated_mib
                        ],
                    }
                    print(
                        "APPENDIX_TRAIN_MEMORY_CURVE "
                        f"{json.dumps(memory_curve_payload, sort_keys=True)}"
                    )
                if self._cpu_util_enabled and self.env.local_rank == 0:
                    cpu_payload: dict[
                        str, int | float | str | list[int] | list[float] | None
                    ] = {
                        "system": "frost",
                        "rank": self.env.local_rank,
                        "warmup_batches_per_epoch": self._throughput_warmup_batches_per_epoch,
                        "warmup_batches_total": self._throughput_warmup_batches_total,
                        "measure_batches_target": self._throughput_measure_batches_target,
                        "logical_cpu_count": self._cpu_logical_count,
                        "tracked_pid_count": len(self._cpu_pid_list),
                        "tracked_pids": self._cpu_pid_list,
                        "measured_batches": len(self._cpu_used_cores),
                        "used_cores_curve": [round(v, 4) for v in self._cpu_used_cores],
                        "utilization_pct_curve": [
                            round(v, 4) for v in self._cpu_utilization_pct
                        ],
                        "wall_interval_s_curve": [
                            round(v, 6) for v in self._cpu_wall_interval_s
                        ],
                    }
                    if self._cpu_used_cores:
                        ordered_used = sorted(self._cpu_used_cores)
                        n_used = len(ordered_used)
                        cpu_payload.update(
                            {
                                "used_cores_mean": sum(ordered_used) / n_used,
                                "used_cores_p50": ordered_used[
                                    int(0.50 * (n_used - 1))
                                ],
                                "used_cores_p90": ordered_used[
                                    int(0.90 * (n_used - 1))
                                ],
                                "used_cores_max": ordered_used[-1],
                            }
                        )
                    if self._cpu_utilization_pct:
                        ordered_util = sorted(self._cpu_utilization_pct)
                        n_util = len(ordered_util)
                        cpu_payload.update(
                            {
                                "utilization_pct_mean": sum(ordered_util) / n_util,
                                "utilization_pct_p50": ordered_util[
                                    int(0.50 * (n_util - 1))
                                ],
                                "utilization_pct_p90": ordered_util[
                                    int(0.90 * (n_util - 1))
                                ],
                                "utilization_pct_max": ordered_util[-1],
                            }
                        )
                    print(
                        "APPENDIX_TRAIN_CPU_UTIL_CURVE "
                        f"{json.dumps(cpu_payload, sort_keys=True)}"
                    )
                if (
                    self.mem_box_store is not None
                    and self.dynamic_state_ablation.uses_async_dynamic_state_pipeline
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

    # Public model save/load interface
