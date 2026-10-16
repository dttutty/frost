from __future__ import annotations

import threading
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.metrics import tgb_reciprocal_ranks as _tgb_reciprocal_ranks
from frost.models.temporal_link_model import TemporalLinkModel
from frost.models.protocols import MemoryUpdaterProto
from frost.shm.ring_buf import RingBuf
from frost.runtime.worker._batch_convergence import BatchConvergenceRecorder
from frost.runtime.worker._state_coordinator import DynamicStateCoordinator
from frost.runtime.experimental.dynamic_state import TransportMode, WritebackTiming


_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


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


class WorkerTrainerMixin:
    model: TemporalLinkModel | DDP
    optimizer: torch.optim.Optimizer
    state_coordinator: DynamicStateCoordinator

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _init_model(
        self,
        encoder: nn.Module,
        mem_updater: MemoryUpdaterProto | None,
        predictor: nn.Module,
        optimizer: Literal["adam", "sgd"],
        ring_buf: RingBuf,
        compile_model: bool = True,
        compile_mode: str | None = None,
        use_ddp: bool = True,
    ) -> None:
        """Create model, optimizer, and wrap with DDP."""
        model = TemporalLinkModel(
            encoder=encoder,
            mem_updater=mem_updater,
            head=predictor,
            representation=self.execution_plan.representation,
            fanout=self.execution_plan.sampling.fanout,
            root_expansion=ring_buf.root_expansion,
        )

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        model.cuda()
        if use_ddp:
            ddp_model = DDP(
                model, process_group=self.nccl, find_unused_parameters=False
            )
            self.model = ddp_model
            self.p2p_guard = P2PGuardCommHook(self.p2p_sem)
            _guard = self.p2p_guard

            def p2p_guard_comm_hook(state, bucket):
                return _guard(state, bucket)

            ddp_model.register_comm_hook(state=self.nccl, hook=p2p_guard_comm_hook)
        else:
            self.model = model
        if compile_model:
            resolved_compile_mode = compile_mode or "default"
            if resolved_compile_mode not in _COMPILE_MODES:
                raise ValueError(
                    f"Unsupported compile_mode={resolved_compile_mode!r}; "
                    f"expected one of {sorted(_COMPILE_MODES)}"
                )
            compile_mode_arg = (
                None if resolved_compile_mode == "default" else resolved_compile_mode
            )
            # Compile only tensor-heavy submodules; Meta graph assembly remains eager.
            bare = model
            bare.encoder = cast(
                nn.Module, torch.compile(bare.encoder, mode=compile_mode_arg)
            )
            if bare.mem_updater is not None:
                bare.mem_updater = cast(
                    MemoryUpdaterProto,
                    torch.compile(bare.mem_updater, mode=compile_mode_arg),
                )
            bare.head = cast(nn.Module, torch.compile(bare.head, mode=compile_mode_arg))
        self.state_coordinator = DynamicStateCoordinator(
            plan=self.execution_plan,
            dynamic_state_policy=self.dynamic_state_policy,
            mem_task_q=self.mem_task_q,
            box_task_q=self.box_task_q,
            direct_writeback=self._write_direct_dynamic_state_request,
        )
        updater_module = (
            cast(nn.Module, model.mem_updater)
            if model.mem_updater is not None
            else None
        )
        self.state_coordinator.attach(updater_module)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def _init_metrics(self) -> None:
        """Create torchmetrics trackers for training and evaluation."""
        self.batch_convergence = BatchConvergenceRecorder.from_environment(
            system="frost",
            rank=getattr(self.env, "local_rank", 0),
        )
        # A single GPU worker has nothing to aggregate.  In that topology the
        # only other process is the host rank, which is deliberately excluded
        # from ``self.nccl``. TorchMetrics' sync path still issues a barrier
        # for a one-rank NCCL group; that barrier is unnecessary and is not
        # reliable on every supported runtime. Keep metrics local for one
        # worker; multi-GPU runs continue to synchronize over the worker-only
        # process group.
        metric_kwargs = {
            "process_group": self.nccl,
            "sync_on_compute": self.env.n_gpu > 1,
        }
        self.train_loss_tracker = MeanMetric(**metric_kwargs).cuda()
        self.val_mrr_tracker = MeanMetric(**metric_kwargs).cuda()
        self.test_mrr_tracker = MeanMetric(**metric_kwargs).cuda()

        self.val_metrics = MetricCollection(
            {
                # Use exact distributed metrics here. The binned approximation
                # can shift the selected best epoch enough to materially change
                # the reported JODIE parity numbers.
                "val_auc": BinaryAUROC(thresholds=None, **metric_kwargs),
                "val_ap": BinaryAveragePrecision(thresholds=None, **metric_kwargs),
            }
        ).cuda()

        self.test_metrics = MetricCollection(
            {
                "test_auc": BinaryAUROC(thresholds=None, **metric_kwargs),
                "test_ap": BinaryAveragePrecision(thresholds=None, **metric_kwargs),
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
        if getattr(self, "state_sync", None) is not None:
            raise RuntimeError(
                "Per-step writer drain is incompatible with the next-batch CPU snapshot fence"
            )
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
        writer = (
            self.gpu_resident_state_writer
            if self.dynamic_state_policy.transport is TransportMode.GPU_LOCAL
            else self.direct_state_writer
        )
        assert writer is not None
        writer.write(
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
        writer = (
            self.gpu_resident_state_writer
            if self.dynamic_state_policy.transport is TransportMode.GPU_LOCAL
            else self.direct_state_writer
        )
        assert writer is not None
        writer.write(
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
        batch_convergence = getattr(self, "batch_convergence", None)
        if batch_convergence is not None:
            batch_convergence.flush_epoch(getattr(self, "curr_epoch", 0) + 1)
        avg_loss = self.train_loss_tracker.compute()
        if self.train_only:
            val_auc = val_ap = test_auc = test_ap = 0.0
            val_mrr = test_mrr = 0.0
        elif self.eval_metric == "ap":
            val_results = self.val_metrics.compute()
            test_results = self.test_metrics.compute()
            val_auc = val_results["val_auc"].item()
            val_ap = val_results["val_ap"].item()
            test_auc = test_results["test_auc"].item()
            test_ap = test_results["test_ap"].item()
            val_mrr = test_mrr = 0.0
        else:
            val_auc = val_ap = test_auc = test_ap = 0.0
            val_mrr = self.val_mrr_tracker.compute().item()
            test_mrr = self.test_mrr_tracker.compute().item()

        self.val_metrics.reset()
        self.test_metrics.reset()
        self.train_loss_tracker.reset()
        self.val_mrr_tracker.reset()
        self.test_mrr_tracker.reset()

        res = Result(
            train_loss=avg_loss.item(),
            val_auc=val_auc,
            val_ap=val_ap,
            test_auc=test_auc,
            test_ap=test_ap,
            val_mrr=val_mrr,
            test_mrr=test_mrr,
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
        if meta.labels is None:
            raise RuntimeError("Training metadata is missing labels")
        labels = meta.labels
        with torch.amp.autocast("cuda", enabled=self.amp):
            with self.state_coordinator.batch(meta):
                logits = self.model(meta)

            logits_flat = logits.squeeze(-1)
            n_pos = meta.n_src
            loss = F.binary_cross_entropy_with_logits(
                logits_flat[:n_pos], labels[:n_pos]
            )
            negative_logits = logits_flat[n_pos:]
            negative_labels = labels[n_pos:]
            if meta.neg_valid_mask is not None:
                negative_logits = negative_logits[meta.neg_valid_mask]
                negative_labels = negative_labels[meta.neg_valid_mask]
            loss = loss + F.binary_cross_entropy_with_logits(
                negative_logits, negative_labels
            )
        if (
            self.mem_box_store is not None
            and self.dynamic_state_policy.transport is TransportMode.ASYNC_P2P
        ):
            self.p2p_guard.arm()  # Arm guard: wait for P2P before the first AllReduce
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.dynamic_state_policy.writeback is WritebackTiming.AFTER_OPTIMIZER_STEP:
            self._flush_deferred_direct_dynamic_state(meta, release_h2d_credits=True)

        detached_loss = loss.detach().float()
        self.train_loss_tracker.update(detached_loss)
        self.batch_convergence.record(events=meta.n_src, loss=detached_loss)
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
            with self.state_coordinator.batch(meta):
                logits = self.model(meta)
        if self.dynamic_state_policy.writeback is WritebackTiming.AFTER_OPTIMIZER_STEP:
            self._flush_deferred_direct_dynamic_state(meta, release_h2d_credits=True)
        if (
            self.mem_box_store is not None
            and self.dynamic_state_policy.transport is TransportMode.ASYNC_P2P
        ):
            self.p2p_sem.acquire()  # Ensure no other P2P task is in flight during eval

        y_pred = logits.float().sigmoid().squeeze(-1)
        if self.eval_metric == "ap" and meta.n_neg != 1:
            raise ValueError(
                "eval_metric='ap' currently requires exactly one evaluation "
                f"negative per positive edge, got n_neg={meta.n_neg}."
            )

        if meta.split == "val":
            if self.eval_metric == "ap":
                self.val_metrics.update(
                    preds=y_pred,
                    target=meta.labels.long(),
                )
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
                self.val_mrr_tracker.update(reciprocal_ranks.mean(), weight=n_pos)

        else:
            assert meta.split == "test"
            if self.eval_metric == "ap":
                self.test_metrics.update(preds=y_pred, target=meta.labels.long())
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
        meta.direct_writeback_request = None
