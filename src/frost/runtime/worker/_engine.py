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
from pathlib import Path
from queue import Queue as Q
from typing import Literal, cast

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency in prod envs
    psutil = None  # type: ignore[assignment]
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)

from frost.ablation import DynamicStateAblation
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
from frost.runtime.worker._direct_writeback import DirectDynamicStateWriter
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
_CLOCK_TICKS_PER_SEC = max(int(os.sysconf("SC_CLK_TCK")), 1)


def _tgb_reciprocal_ranks(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    negative_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return TGB tie-aware reciprocal ranks for fixed-width candidates."""
    if positive_scores.dim() != 1:
        raise ValueError(
            f"positive_scores must be 1D, got shape {tuple(positive_scores.shape)}"
        )
    if negative_scores.dim() != 2:
        raise ValueError(
            f"negative_scores must be 2D, got shape {tuple(negative_scores.shape)}"
        )
    if positive_scores.shape[0] != negative_scores.shape[0]:
        raise ValueError(
            "positive_scores and negative_scores must contain the same number "
            f"of queries, got {positive_scores.shape[0]} and {negative_scores.shape[0]}"
        )
    if negative_valid_mask is not None:
        if negative_valid_mask.shape != negative_scores.shape:
            raise ValueError(
                "negative_valid_mask must match negative_scores, got "
                f"{tuple(negative_valid_mask.shape)} and "
                f"{tuple(negative_scores.shape)}"
            )
        negative_valid_mask = negative_valid_mask.to(
            device=negative_scores.device, dtype=torch.bool
        )

    positive_scores = positive_scores.unsqueeze(1)
    if negative_valid_mask is None:
        negative_valid_mask = torch.ones_like(negative_scores, dtype=torch.bool)
    optimistic_rank = ((negative_scores > positive_scores) & negative_valid_mask).sum(
        dim=1
    )
    pessimistic_rank = ((negative_scores >= positive_scores) & negative_valid_mask).sum(
        dim=1
    )
    rank = 0.5 * (optimistic_rank + pessimistic_rank).float() + 1.0
    return rank.reciprocal()


def _get_non_negative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _get_optional_non_negative_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_memory_worker_queue_size_env(default: int = 1) -> int:
    raw = os.getenv("FROST_MEMORY_WORKER_QUEUE_SIZE")
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 1:
        raise ValueError("FROST_MEMORY_WORKER_QUEUE_SIZE must be >= 1")
    return value


def _discover_sibling_rank_pids() -> list[int]:
    """Return all sibling rank PIDs launched under the same torchrun parent.

    We rely on the launcher parent rather than distributed collectives so the
    benchmark can include both worker ranks and the host process without adding
    any extra synchronization requirements to the host runtime.
    """

    pid = os.getpid()
    if psutil is None:  # pragma: no cover - guarded by benchmark flag
        return [pid]
    try:
        if psutil is not None:
            proc = psutil.Process(pid)
            parent = proc.parent()
            if parent is None:
                return [pid]
            child_pids = sorted(
                {child.pid for child in parent.children(recursive=False)}
            )
            return child_pids or [pid]

        stat_path = Path(f"/proc/{pid}/stat")
        text = stat_path.read_text(encoding="utf-8")
        rparen = text.rfind(")")
        if rparen == -1:
            return [pid]
        tail = text[rparen + 2 :].split()
        parent_pid = int(tail[1])
        if parent_pid <= 1:
            return [pid]
        child_pids: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                sibling_text = (entry / "stat").read_text(encoding="utf-8")
                sibling_rparen = sibling_text.rfind(")")
                if sibling_rparen == -1:
                    continue
                sibling_tail = sibling_text[sibling_rparen + 2 :].split()
                if int(sibling_tail[1]) == parent_pid:
                    child_pids.append(int(entry.name))
            except (OSError, ValueError):
                continue
        return sorted(set(child_pids)) or [pid]
    except (psutil.Error, OSError, ValueError):
        return [pid]


def _sample_total_cpu_time_s(pid_list: list[int]) -> tuple[float, list[int]]:
    """Return summed user+system CPU time over tracked processes."""

    total = 0.0
    live_pids: list[int] = []
    for pid in pid_list:
        try:
            if psutil is not None:
                proc = psutil.Process(pid)
                cpu_times = proc.cpu_times()
                total += float(cpu_times.user) + float(cpu_times.system)
            else:
                text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
                rparen = text.rfind(")")
                if rparen == -1:
                    continue
                tail = text[rparen + 2 :].split()
                utime_ticks = int(tail[11])
                stime_ticks = int(tail[12])
                total += (utime_ticks + stime_ticks) / _CLOCK_TICKS_PER_SEC
        except (
            psutil.NoSuchProcess if psutil is not None else ProcessLookupError,
            psutil.AccessDenied if psutil is not None else PermissionError,
            psutil.ZombieProcess if psutil is not None else ChildProcessError,
            OSError,
            ValueError,
        ):
            continue
        live_pids.append(pid)
    return total, live_pids


def _initial_h2d_pipeline_slots(
    *,
    p2p_world_size: int,
) -> int:
    # Single-worker memory runs have no cross-GPU overlap benefit, and
    # allowing two batches in flight has proven much more timing-sensitive
    # than the serialized one-batch path. Keep that configuration at one
    # batch in flight by default.
    if p2p_world_size == 1:
        return 2
    return 4


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
    direct_state_writer: DirectDynamicStateWriter | None = None

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
                if psutil is not None:
                    logical_cpu_count = int(psutil.cpu_count(logical=True) or 1)
                else:
                    logical_cpu_count = int(os.cpu_count() or 1)
                self._cpu_logical_count = max(logical_cpu_count, 1)
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

    def _init_pipeline(
        self,
        ring_buf: RingBuf,
        nfeat: torch.Tensor | None,
        efeat: torch.Tensor | None,
    ) -> None:
        """Create all queues and background pipeline threads."""
        has_mem = self.mem_box_store is not None
        use_async_dynamic_state_pipeline = (
            has_mem and self.dynamic_state_ablation.uses_async_dynamic_state_pipeline
        )
        memory_q_size = _get_memory_worker_queue_size_env(default=1)
        stateless_q_size = self.no_memory_queue_size if not has_mem else memory_q_size
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
            if use_async_dynamic_state_pipeline:
                self.delta_queue = Q(maxsize=memory_q_size)
            self.h2d_out_q = Q(maxsize=memory_q_size)
            if use_async_dynamic_state_pipeline:
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
                dynamic_state_ablation=self.dynamic_state_ablation,
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
            dynamic_state_ablation=self.dynamic_state_ablation,
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
        self.train_loss_tracker = MeanMetric(process_group=self.nccl).cuda()
        self.val_mrr_tracker = MeanMetric(process_group=self.nccl).cuda()
        self.test_mrr_tracker = MeanMetric(process_group=self.nccl).cuda()
        self.val_mrr_loss_tracker = MeanMetric(process_group=self.nccl).cuda()
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
            if q is not None:
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

    def _write_direct_dynamic_state_request(self, request) -> None:
        assert self.direct_state_writer is not None
        self.direct_state_writer.write(
            request,
            release_h2d_credits=True,
        )

    def _flush_deferred_direct_dynamic_state(
        self,
        meta: Meta,
        *,
        release_h2d_credits: bool,
    ) -> None:
        request = meta.direct_writeback_request
        if request is None:
            return
        assert self.direct_state_writer is not None
        self.direct_state_writer.write(
            request,
            release_h2d_credits=release_h2d_credits,
        )
        meta.direct_writeback_request = None

    def on_train_epoch_end(self) -> Result:
        """Aggregate per-epoch metrics across all ranks and return a Result.

        Computes synchronized AP/AUC metrics by default, or TGB-compatible MRR
        when ``eval_metric="mrr"``, then resets all accumulators.

        Returns:
            Result dataclass containing the aggregated scalar metrics for this epoch.
        """
        avg_loss = self.train_loss_tracker.compute()
        if self.eval_metric == "ap":
            val_results = self.val_metrics.compute()
            test_results = self.test_metrics.compute()
            val_loss = val_results["val_loss"].item()
            val_auc = val_results["val_auc"].item()
            val_ap = val_results["val_ap"].item()
            test_auc = test_results["test_auc"].item()
            test_ap = test_results["test_ap"].item()
            val_mrr = test_mrr = 0.0
            val_auc_batch_mean = self.val_auc_batch_mean.compute().item()
            val_ap_batch_mean = self.val_ap_batch_mean.compute().item()
            test_auc_batch_mean = self.test_auc_batch_mean.compute().item()
            test_ap_batch_mean = self.test_ap_batch_mean.compute().item()
        else:
            val_loss = self.val_mrr_loss_tracker.compute().item()
            val_auc = val_ap = test_auc = test_ap = 0.0
            val_mrr = self.val_mrr_tracker.compute().item()
            test_mrr = self.test_mrr_tracker.compute().item()
            val_auc_batch_mean = val_ap_batch_mean = 0.0
            test_auc_batch_mean = test_ap_batch_mean = 0.0

        self.val_metrics.reset()
        self.test_metrics.reset()
        self.train_loss_tracker.reset()
        self.val_mrr_tracker.reset()
        self.test_mrr_tracker.reset()
        self.val_mrr_loss_tracker.reset()
        self.val_auc_batch_mean.reset()
        self.val_ap_batch_mean.reset()
        self.test_auc_batch_mean.reset()
        self.test_ap_batch_mean.reset()

        res = Result(
            train_loss=avg_loss.item(),
            val_loss=val_loss,
            val_auc=val_auc,
            val_ap=val_ap,
            test_auc=test_auc,
            test_ap=test_ap,
            val_mrr=val_mrr,
            test_mrr=test_mrr,
            val_auc_batch_mean=val_auc_batch_mean,
            val_ap_batch_mean=val_ap_batch_mean,
            test_auc_batch_mean=test_auc_batch_mean,
            test_ap_batch_mean=test_ap_batch_mean,
        )
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
                direct_writeback=self._write_direct_dynamic_state_request,
            )

            logits_flat = logits.squeeze(-1)
            n_pos = meta.n_src
            loss = self.criterion(logits_flat[:n_pos], meta.labels[:n_pos])
            negative_logits = logits_flat[n_pos:]
            negative_labels = meta.labels[n_pos:]
            if meta.neg_valid_mask is not None:
                negative_logits = negative_logits[meta.neg_valid_mask]
                negative_labels = negative_labels[meta.neg_valid_mask]
            loss = loss + self.criterion(negative_logits, negative_labels)
        if (
            self.mem_box_store is not None
            and self.dynamic_state_ablation.uses_async_dynamic_state_pipeline
        ):
            self.p2p_guard.arm()  # Arm guard: wait for P2P before the first AllReduce
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.dynamic_state_ablation.direct_cpu_writeback_after_optimizer_step:
            self._flush_deferred_direct_dynamic_state(meta, release_h2d_credits=True)

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
                direct_writeback=self._write_direct_dynamic_state_request,
            )
        if self.dynamic_state_ablation.direct_cpu_writeback_after_optimizer_step:
            self._flush_deferred_direct_dynamic_state(meta, release_h2d_credits=True)
        if (
            self.mem_box_store is not None
            and self.dynamic_state_ablation.uses_async_dynamic_state_pipeline
        ):
            self.p2p_sem.acquire()  # Ensure no other P2P task is in flight during eval

        y_pred = logits.float().sigmoid().squeeze(-1)
        if self.eval_metric == "ap":
            if meta.n_neg != 1:
                raise ValueError(
                    "eval_metric='ap' currently requires exactly one evaluation "
                    f"negative per positive edge, got n_neg={meta.n_neg}."
                )
            batch_ap = binary_average_precision(
                y_pred, meta.labels.long(), thresholds=None
            )
            batch_auc = binary_auroc(y_pred, meta.labels.long(), thresholds=None)

        if meta.split == "val":
            logits_flat = logits.float().squeeze(-1)
            n_pos = meta.n_src
            loss = self.criterion(logits_flat[:n_pos], meta.labels[:n_pos])
            negative_logits = logits_flat[n_pos:]
            negative_labels = meta.labels[n_pos:]
            if meta.neg_valid_mask is not None:
                negative_logits = negative_logits[meta.neg_valid_mask]
                negative_labels = negative_labels[meta.neg_valid_mask]
            loss = loss + self.criterion(negative_logits, negative_labels)

            if self.eval_metric == "ap":
                self.val_metrics.update(
                    preds=y_pred,
                    target=meta.labels.long(),
                    value=loss,
                )
                self.val_ap_batch_mean.update(batch_ap)
                self.val_auc_batch_mean.update(batch_auc)
            else:
                self.val_mrr_loss_tracker.update(loss, weight=n_pos)
                y_pos = logits_flat[:n_pos].unsqueeze(1)
                y_neg = logits_flat[n_pos:].view(n_pos, meta.n_neg)
                neg_valid_mask = (
                    None
                    if meta.neg_valid_mask is None
                    else meta.neg_valid_mask.view(n_pos, meta.n_neg)
                )
                reciprocal_ranks = _tgb_reciprocal_ranks(
                    y_pos.squeeze(1), y_neg, neg_valid_mask
                )
                self.val_mrr_tracker.update(reciprocal_ranks.mean(), weight=n_pos)

        else:
            assert meta.split == "test"
            if self.eval_metric == "ap":
                self.test_metrics.update(preds=y_pred, target=meta.labels.long())
                self.test_ap_batch_mean.update(batch_ap)
                self.test_auc_batch_mean.update(batch_auc)
            else:
                logits_flat = logits.float().squeeze(-1)
                n_pos = meta.n_src
                y_pos = logits_flat[:n_pos].unsqueeze(1)
                y_neg = logits_flat[n_pos:].view(n_pos, meta.n_neg)
                neg_valid_mask = (
                    None
                    if meta.neg_valid_mask is None
                    else meta.neg_valid_mask.view(n_pos, meta.n_neg)
                )
                reciprocal_ranks = _tgb_reciprocal_ranks(
                    y_pos.squeeze(1), y_neg, neg_valid_mask
                )
                self.test_mrr_tracker.update(reciprocal_ranks.mean(), weight=n_pos)

        self._release_meta_tensors(meta)

    def _release_meta_tensors(self, meta: Meta) -> None:
        """Drop per-batch GPU tensors ASAP to reduce delayed CUDA memory recycling."""

        meta.flat_graph = None
        meta.plan_set = None
        meta.labels = None
        meta.neg_valid_mask = None
        meta.graph = None
        meta.direct_writeback_request = None

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
            assert self.dy_asm is not None
            if self.dynamic_state_ablation.uses_async_dynamic_state_pipeline:
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
                        self._throughput_gpu_pairs.append(
                            (self._throughput_prev_start_event, curr_start_event)
                        )
                        self._throughput_wall_intervals_s.append(
                            curr_start_wall_s - self._throughput_prev_start_wall_s
                        )
                        if self._throughput_prev_start_wall_unix_s is not None:
                            self._throughput_interval_start_unix_s.append(
                                self._throughput_prev_start_wall_unix_s
                            )
                            self._throughput_interval_end_unix_s.append(
                                curr_start_wall_unix_s
                            )
                        self._throughput_events += self._throughput_prev_batch_events
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
                            wall_interval_s = (
                                curr_start_wall_s - self._throughput_prev_start_wall_s
                            )
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
                memory_payload: dict[str, int | float | str] = {
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
                        str, int | float | str | list[int] | list[float]
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
