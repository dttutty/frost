import json
import logging
import os
import threading
import time
from queue import Queue as Q
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from frost.runtime.host.types import (
    DynamicPlanOutQ,
    LayoutBundle,
    MetaBundle,
    StaticPlanOutQ,
)
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.datatypes.sampled_layout import SampledLayoutView
from frost.models.contracts import RootExpansion
from frost.runtime._thread_base import (
    destroy_process_group,
    join_thread,
    put_shutdown_sentinel,
)
from frost.runtime._early_stopping import is_significant_improvement
from frost.runtime.host import PlanRouter, Scatter
from frost.runtime.host._batch_balance import make_balance_plan
from frost.runtime.host._config import HostOptions, HostResources
from frost.runtime.experimental.dynamic_state import (
    PlannerMode,
    resolve_dynamic_state_policy,
)
from frost.shm.control_plane import SharedControlPlane

logger = logging.getLogger(__name__)

MEMORY_SCATTER_QUEUE_SIZE = 64
MEMORY_MEM_PLAN_OUT_QUEUE_SIZE = 32
MEMORY_MAIL_PLAN_OUT_QUEUE_SIZE = 32
MEMORY_NFEAT_PLAN_OUT_QUEUE_SIZE = 32


def _disable_tqdm_monitor_thread() -> None:
    """Disable tqdm's monitor thread to avoid debugger/threading instability."""
    if getattr(tqdm, "monitor_interval", 0) != 0:
        tqdm.monitor_interval = 0


def _timing_enabled() -> bool:
    return os.environ.get("FROST_HOST_TIMING_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prepare_sampler_inputs(
    roots: torch.Tensor,
    ts: torch.Tensor,
    n_neg: int,
    root_expansion: RootExpansion,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the explicit sampler inputs from the full root layout.

    ``roots`` always keeps the canonical RingBuf root layout
    ``[src | pos_dst | neg_dst×n_neg]``. This helper derives the subset that
    should actually be expanded by the sampler together with the already-aligned
    timestamp tensor.
    """
    root_width = 2 + n_neg
    if roots.shape[0] % root_width != 0:
        raise ValueError(
            f"roots length must be divisible by {root_width}, got {roots.shape[0]}"
        )
    n_src = roots.shape[0] // root_width
    if ts.shape[0] != n_src:
        raise ValueError(f"ts length must equal n_src={n_src}, got {ts.shape[0]}")

    if root_expansion is RootExpansion.ALL_ROOTS:
        return roots, torch.cat([ts, ts, ts.repeat_interleave(n_neg)])

    return roots[: n_src * 2], ts.repeat(2)


def _layout_work_counts(layouts: LayoutBundle) -> dict[str, int]:
    """Count dense slots and semantic rows in one distributed macro-batch."""
    root_rows = 0
    unique_root_rows = 0
    neighbor_slots = 0
    valid_neighbor_rows = 0
    unique_valid_neighbor_rows = 0
    for layout in layouts:
        roots = layout.filter_valid(layout.roots)
        root_rows += int(roots.numel())
        unique_root_rows += int(torch.unique(roots).numel())
        if not layout.fanout:
            continue
        hop = layout.valid_hop(0)
        first_hop_slots = (
            layout.n_root
            if layout.root_expansion is RootExpansion.ALL_ROOTS
            else 2 * layout.n_src
        ) * layout.fanout[0]
        neighbor_slots += first_hop_slots
        valid_neighbor_rows += int(hop.numel())
        unique_valid_neighbor_rows += int(torch.unique(hop).numel())
    return {
        "root_rows": root_rows,
        "unique_root_rows": unique_root_rows,
        "neighbor_slots": neighbor_slots,
        "valid_neighbor_rows": valid_neighbor_rows,
        "unique_valid_neighbor_rows": unique_valid_neighbor_rows,
    }


class Host:
    """Orchestrates the host-side pipeline for distributed Temporal GNN training.

    The Host process drives the full data pipeline: it reads the edge DataFrame,
    splits it into train/val/test, samples neighbors and negative destinations for
    each macro-batch, optionally computes memory/nfeat access plans via
    ``PlanRouter``, and scatters ``Meta`` bundles to Worker processes through the
    ``Scatter`` thread.  Early stopping is evaluated after each epoch.

    Attributes:
        mem_plan_out_q: Output queue for memory DynamicPlan lists, or None.
        mail_plan_out_q: Output queue for mailbox DynamicPlan lists, or None.
        nfeat_plan_out_q: Output queue for node-feature StaticPlan lists, or None.
        scatter_q: Queue of MetaBundles (or sentinels) consumed by Scatter.
        plan_router: Optional PlanRouter managing sub-planner threads.
    """

    mem_plan_out_q: DynamicPlanOutQ | None = None
    mail_plan_out_q: DynamicPlanOutQ | None = None
    nfeat_plan_out_q: StaticPlanOutQ | None = None

    scatter_q: Q[MetaBundle | None | Literal["end_of_epoch"]]
    plan_router: PlanRouter | None = None

    def __init__(
        self,
        resources: HostResources,
        options: HostOptions,
    ) -> None:
        self.neg_sampler_train = resources.train_negative_sampler
        self.neg_sampler_eval = resources.eval_negative_sampler
        self.ngh_sampler = resources.neighborhood_sampler
        self.start_time = options.start_time
        self.has_nfeat = options.has_node_features
        self.has_mem = options.has_dynamic_state
        self.n_gpu = resources.env.n_gpu
        self.batch_size = options.batch_size
        self.eval_batch_size = options.eval_batch_size
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be >= 1.")
        self.analyzer = resources.analyzer
        self.show_tqdm = options.show_progress
        self.time_dtype = options.timestamp_dtype
        self.reset_eval_neg_sampler_each_epoch = options.reset_eval_sampler_each_epoch
        self.share_neg_sampler_across_splits = (
            options.share_negative_sampler_across_splits
        )
        self.normalize_event_time = options.normalize_event_time
        if options.eval_metric not in {"ap", "mrr"}:
            raise ValueError(
                f"Unsupported eval_metric={options.eval_metric!r}; expected 'ap' or 'mrr'."
            )
        self.eval_metric = options.eval_metric
        self.train_only = options.train_only
        self.dynamic_state_policy = (
            options.dynamic_state_policy
            if options.dynamic_state_policy is not None
            else resolve_dynamic_state_policy("baseline")
        )
        if options.no_memory_queue_size < 1:
            raise ValueError("no_memory_queue_size must be >= 1.")
        self.no_memory_queue_size = options.no_memory_queue_size

        self.stop_ev = threading.Event()
        self.mini_batch_idx = 0
        self.macro_batch_idx = 0
        self.curr_epoch = 0
        self.best_metric_epoch: int = -1
        self.best_metric_score = float("-inf")
        self.best_metric_result: Result = Result(
            train_loss=float("inf"),
            val_auc=0.0,
            val_ap=0.0,
            test_auc=0.0,
            test_ap=0.0,
        )
        self.total_epoch = options.epochs
        self.patience = options.patience
        if options.early_stop_tolerance < 0:
            raise ValueError("early_stop_tolerance must be >= 0.")
        self.early_stop_tolerance = options.early_stop_tolerance
        self.patience_counter = 0
        self.last_epoch_result: Result | None = None
        self._shutdown_done = False
        self._timing_enabled = _timing_enabled()
        self._epoch_wall_enabled = os.environ.get(
            "FROST_EPOCH_WALL_BENCH", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._dispatch_s_curve: list[float] = []
        self._scatter_q_put_s_curve: list[float] = []
        self._macro_prepare_s_curve: list[float] = []
        self._macro_total_s_curve: list[float] = []
        self._tensorize_s_curve: list[float] = []
        self._negative_sample_s_curve: list[float] = []
        self._neighborhood_sample_s_curve: list[float] = []
        self._metadata_s_curve: list[float] = []
        self._layout_work_curve: list[dict[str, int]] = []

        self._init_pipeline(
            resources.env,
            resources.gpu_topology,
            resources.control_plane,
            options.carry_over_route,
            options.output_dir,
            options.persist_all_updates_to_cpu,
        )
        self._init_dataset(resources.dataframe)

    def _init_pipeline(
        self,
        env: Env,
        gpu_topo: list,
        control_plane: SharedControlPlane,
        if_carry_over_route: bool,
        output_dir: str | None,
        persist_all_updates_to_cpu: bool,
    ) -> None:
        """Fixed linear pipeline: main thread → [PlanRouter.dispatch() →] scatter_q → Scatter.

        The main thread accumulates n_gpu metas per macro-batch, optionally dispatches to
        sub-planner threads via PlanRouter.dispatch(), then puts MetaBundle into scatter_q.
        No dedicated thread for routing — it is inlined into the main thread.
        """
        enable_mem_mail_planner = (
            self.has_mem
            and self.dynamic_state_policy.planner is not PlannerMode.NONE
        )
        has_planner = self.has_nfeat or enable_mem_mail_planner
        stateless_q_size = self.no_memory_queue_size if not self.has_mem else 2
        memory_scatter_q_size = MEMORY_SCATTER_QUEUE_SIZE
        memory_mem_plan_out_q_size = MEMORY_MEM_PLAN_OUT_QUEUE_SIZE
        memory_mail_plan_out_q_size = MEMORY_MAIL_PLAN_OUT_QUEUE_SIZE
        memory_nfeat_plan_out_q_size = MEMORY_NFEAT_PLAN_OUT_QUEUE_SIZE

        self.result_q: Q[Result] = Q(maxsize=10)
        self.scatter_q = Q(
            maxsize=stateless_q_size if not self.has_mem else memory_scatter_q_size
        )

        if has_planner:
            if enable_mem_mail_planner:
                self.mem_plan_out_q = Q(maxsize=memory_mem_plan_out_q_size)
                self.mail_plan_out_q = Q(maxsize=memory_mail_plan_out_q_size)
            if self.has_nfeat:
                nfeat_q_size = (
                    memory_nfeat_plan_out_q_size
                    if self.has_mem
                    else self.no_memory_queue_size
                )
                self.nfeat_plan_out_q = Q(maxsize=nfeat_q_size)

            self.plan_router = PlanRouter(
                gpu_topo=gpu_topo,
                analyzer=self.analyzer,
                mem_plan_out_q=self.mem_plan_out_q,
                mail_plan_out_q=self.mail_plan_out_q,
                nfeat_plan_out_q=self.nfeat_plan_out_q,
                stop_ev=self.stop_ev,
                if_carry_over_route=if_carry_over_route,
                output_dir=output_dir,
                persist_all_updates_to_cpu=persist_all_updates_to_cpu,
                dynamic_state_policy=self.dynamic_state_policy,
            )

        self.scatter = Scatter(
            n_gpu=env.n_gpu,
            in_q=self.scatter_q,
            mem_plan_out_q=self.mem_plan_out_q,
            mail_plan_out_q=self.mail_plan_out_q,
            nfeat_plan_out_q=self.nfeat_plan_out_q,
            env=env,
            result_q=self.result_q,
            stop_ev=self.stop_ev,
            control_plane=control_plane,
        )

    def _init_dataset(self, df: pd.DataFrame) -> None:
        """Preprocess the edge DataFrame and split into train/val/test subsets.

        Timestamps are normalised to start at zero. The ``default_split`` column is
        used to locate split boundaries (0 = train, 1 = val, >=2 = test).

        Args:
            df: Raw edge DataFrame with columns ``eid``, ``src``, ``dst``,
                ``time``, ``default_split``.
        """
        # preprocess dataframe
        df["src"] = df["src"].astype(np.int64)
        df["dst"] = df["dst"].astype(np.int64)
        if self.normalize_event_time:
            t_min = df["time"].min()
            df["time"] = (df["time"] - t_min).astype(np.int64)
        else:
            df["time"] = df["time"].astype(np.int64)

        train_edge_end = df[df["default_split"].gt(0)].index[0]
        val_edge_end = df[df["default_split"].gt(1)].index[0]
        self.tr_df = df[:train_edge_end]
        self.va_df = df[train_edge_end:val_edge_end]
        self.te_df = df[val_edge_end:]

    def handle_result_q(self) -> None:
        """Block until a Result arrives from the Scatter thread and log epoch metrics.

        The result is cached in ``self.last_epoch_result`` so that
        ``early_stop_check`` can read it without a second queue pop.
        """
        res = self.result_q.get(block=True)

        # Cache result so early_stop_check can read it
        self.last_epoch_result = res
        if self.train_only:
            print(
                f"[Epoch {self.curr_epoch}] Train Loss: {res.train_loss:.4f} "
                "| evaluation skipped"
            )
            return
        selection_score = self._selection_score(res)
        is_improvement = is_significant_improvement(
            selection_score,
            self.best_metric_score,
            self.early_stop_tolerance,
        )
        metric_name = self.eval_metric.upper()
        metric_label = (
            f"Val {metric_name}(New high)"
            if is_improvement
            else f"Val {metric_name}"
        )

        # Host prints the epoch summary; workers stay silent
        if self.eval_metric == "ap":
            print(
                f"[Epoch {self.curr_epoch}] "
                f"Train Loss: {res.train_loss:.4f} | "
                f"{metric_label}: {res.val_ap:.4f} | "
                f"Val AUC: {res.val_auc:.4f} | "
                f"Test AUC: {res.test_auc:.4f} | "
                f"Test AP: {res.test_ap:.4f}"
            )
        else:
            print(
                f"[Epoch {self.curr_epoch}] "
                f"Train Loss: {res.train_loss:.4f} | "
                f"{metric_label}: {res.val_mrr:.4f} | "
                f"Test MRR: {res.test_mrr:.4f}"
            )

    def _selection_score(self, result: Result) -> float:
        return result.val_ap if self.eval_metric == "ap" else result.val_mrr

    def early_stop_check(self, cur_epoch: int) -> bool:
        """
        Check whether early stopping should be triggered and update the best-model record.
        Returns True if training should stop, False to continue.
        """
        res = self.last_epoch_result
        if res is None:
            return False

        is_best = False

        selection_score = self._selection_score(res)
        if is_significant_improvement(
            selection_score,
            self.best_metric_score,
            self.early_stop_tolerance,
        ):
            self.best_metric_epoch = cur_epoch + 1
            self.best_metric_score = selection_score
            self.best_metric_result = res
            is_best = True
        if is_best:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        # Early stopping check
        if self.patience_counter >= self.patience:
            logger.info(
                f"🛑 Early stopping triggered at epoch {cur_epoch + 1}. "
                f"Best {self.eval_metric.upper()} was {self.best_metric_score:.4f} "
                f"at epoch {self.best_metric_epoch}."
            )
            return True

        return False

    def start_all_threads(self) -> None:
        """Start all background threads (currently only the Scatter thread)."""
        self.scatter.start()

    def run(self) -> None:
        """Main training entry point."""
        _disable_tqdm_monitor_thread()
        self.start_all_threads()
        for e in range(self.total_epoch):
            epoch_wall_start = time.perf_counter()
            epoch_wall_start_unix = time.time()
            logger.info("Starting Epoch %s", e)
            self._handle_a_split(self.tr_df, "train", cur_epoch=e)
            if self.train_only:
                self._cleanup_per_epoch()
            else:
                self._handle_a_split(self.va_df, "val", cur_epoch=e)
                self._handle_a_split(self.te_df, "test", cur_epoch=e)
            self.handle_result_q()
            if self._epoch_wall_enabled:
                epoch_wall_end = time.perf_counter()
                print(
                    "APPENDIX_TRAIN_EPOCH_WALL "
                    + json.dumps(
                        {
                            "system": "frost",
                            "epoch": e,
                            "train_events": len(self.tr_df),
                            "wall_time_s": epoch_wall_end - epoch_wall_start,
                            "window_start_unix_s": epoch_wall_start_unix,
                            "window_end_unix_s": time.time(),
                        },
                        sort_keys=True,
                    )
                )
            if not self.train_only and self.early_stop_check(e):
                break

        if self.train_only:
            self.shutdown(graceful=True)
            return

        if self.eval_metric == "ap":
            print(
                f"Best Val AP: {self.best_metric_result.val_ap:.4f} "
                f"@ Epoch {self.best_metric_epoch} | "
                f"Val AUC: {self.best_metric_result.val_auc:.4f} | "
                f"Test AP: {self.best_metric_result.test_ap:.4f} | "
                f"Test AUC: {self.best_metric_result.test_auc:.4f}"
            )
        else:
            print(
                f"Best Val MRR: {self.best_metric_result.val_mrr:.4f} "
                f"@ Epoch {self.best_metric_epoch} | "
                f"Test MRR: {self.best_metric_result.test_mrr:.4f}"
            )
        self.shutdown(graceful=True)
        # wandb.finish()

    def _handle_a_split(
        self, df: pd.DataFrame, split: Literal["train", "val", "test"], cur_epoch: int
    ) -> None:
        """Process all macro-batches for one data split in the current epoch.

        Validates that the split's ``default_split`` values are consistent, applies
        ``make_balance_plan`` to assign edges to GPUs, then iterates over
        macro-batches: builds ``Meta``/``SampledLayoutView`` pairs, optionally dispatches
        to ``PlanRouter``, and enqueues bundles into ``scatter_q``. On the final
        test split, sends the end-of-epoch sentinel.

        Args:
            df: Edge DataFrame for the split (train, val, or test).
            split: String label identifying the split.
            cur_epoch: Zero-based epoch index.
        """
        if "default_split" in df.columns:
            default_split = df["default_split"]
            is_valid_split = (
                (split == "train" and default_split.eq(0).all())
                or (split == "val" and default_split.eq(1).all())
                or (split == "test" and default_split.ge(2).all())
            )
            if not is_valid_split:
                found_rolls = sorted(default_split.dropna().unique().tolist())
                raise ValueError(
                    f"Split contamination detected for split={split}. "
                    f"Found default_split values: {found_rolls}"
                )

        if len(df) == 0:
            return
        timing_enabled = getattr(self, "_timing_enabled", False)

        if split != "train" and not getattr(
            self, "share_neg_sampler_across_splits", False
        ):
            if getattr(self, "reset_eval_neg_sampler_each_epoch", True):
                self.neg_sampler_eval.reset()

        split_debug_env = f"FROST_DEBUG_MAX_BATCHES_{split.upper()}"
        debug_max_batches = int(
            os.environ.get(
                split_debug_env,
                os.environ.get("FROST_DEBUG_MAX_BATCHES_PER_SPLIT", "0"),
            )
        )
        if debug_max_batches < 0:
            raise ValueError(f"{split_debug_env} must be >= 0.")
        if debug_max_batches > 0:
            max_rows = (
                debug_max_batches
                * self.n_gpu
                * (
                    self.batch_size
                    if split == "train"
                    else getattr(self, "eval_batch_size", self.batch_size)
                )
            )
            df = df.iloc[:max_rows]

        # Use make_balance_plan for balanced GPU distribution
        per_gpu_batch_size = (
            self.batch_size
            if split == "train"
            else getattr(self, "eval_batch_size", self.batch_size)
        )
        df_out, step_id, gpu_id = make_balance_plan(
            df, org_batch_size=per_gpu_batch_size, n_gpu=self.n_gpu
        )
        df_out = df_out.assign(step_id=step_id, gpu_id=gpu_id)

        n_macro_of_split = df_out["step_id"].nunique()
        if debug_max_batches > 0:
            n_macro_of_split = min(n_macro_of_split, debug_max_batches)
        step_desc_map = {
            "train": "Train Step",
            "val": "Val Step",
            "test": "Test Step",
        }

        for cur_macro_of_split, step_df in tqdm(
            df_out.groupby("step_id", sort=False),
            total=n_macro_of_split,
            desc=step_desc_map[split],
            disable=not self.show_tqdm,
        ):
            assert isinstance(cur_macro_of_split, int)
            if cur_macro_of_split >= n_macro_of_split:
                break
            macro_start = time.perf_counter()
            macro_batch_idx = getattr(self, "macro_batch_idx", 0)
            is_last_macro_of_split = (
                cur_macro_of_split + 1 == n_macro_of_split
            )
            meta_bundle: MetaBundle = []
            layout_bundle: LayoutBundle = []
            for gpu, edges in step_df.groupby("gpu_id", sort=False):
                assert isinstance(gpu, int)
                meta, layout = self._add_to_td_list(
                    edges,
                    split,
                    self.mini_batch_idx,
                    gpu,
                    macro_batch_idx=macro_batch_idx,
                    epoch_idx=cur_epoch,
                    is_last_macro_of_split=is_last_macro_of_split,
                    is_last_macro_of_epoch=(
                        is_last_macro_of_split
                        and (
                            split == "test"
                            or (split == "train" and getattr(self, "train_only", False))
                        )
                    ),
                )
                meta_bundle.append(meta)
                layout_bundle.append(layout)
                self.mini_batch_idx += 1
            self.macro_batch_idx = macro_batch_idx + 1
            prepare_end = time.perf_counter()
            if timing_enabled and split == "train":
                self._macro_prepare_s_curve.append(prepare_end - macro_start)
                self._layout_work_curve.append(_layout_work_counts(layout_bundle))
            if self.plan_router is not None:
                dispatch_start = time.perf_counter()
                self.plan_router.dispatch(meta_bundle, layout_bundle)
                dispatch_end = time.perf_counter()
                if timing_enabled and split == "train":
                    self._dispatch_s_curve.append(dispatch_end - dispatch_start)
            scatter_put_start = time.perf_counter()
            self.scatter_q.put(meta_bundle)
            scatter_put_end = time.perf_counter()
            if timing_enabled and split == "train":
                self._scatter_q_put_s_curve.append(scatter_put_end - scatter_put_start)
                self._macro_total_s_curve.append(scatter_put_end - macro_start)

        if timing_enabled and split == "train" and self._scatter_q_put_s_curve:
            import json

            payload = {
                "system": "frost",
                "split": split,
                "epoch": cur_epoch,
                "batch_count": len(self._scatter_q_put_s_curve),
                "macro_prepare_s_curve": [
                    round(v, 6) for v in self._macro_prepare_s_curve
                ],
                "macro_total_s_curve": [
                    round(v, 6) for v in self._macro_total_s_curve
                ],
                "tensorize_s_curve": [round(v, 6) for v in self._tensorize_s_curve],
                "negative_sample_s_curve": [
                    round(v, 6) for v in self._negative_sample_s_curve
                ],
                "neighborhood_sample_s_curve": [
                    round(v, 6) for v in self._neighborhood_sample_s_curve
                ],
                "metadata_s_curve": [round(v, 6) for v in self._metadata_s_curve],
                "dispatch_s_curve": [round(v, 6) for v in self._dispatch_s_curve],
                "scatter_q_put_s_curve": [
                    round(v, 6) for v in self._scatter_q_put_s_curve
                ],
                "layout_work_curve": self._layout_work_curve,
            }
            print(
                "APPENDIX_HOST_ENGINE_TIMING_CURVE "
                f"{json.dumps(payload, sort_keys=True)}"
            )

        if split == "test":
            self._cleanup_per_epoch()

    def _cleanup_per_epoch(self) -> None:
        """Signal end-of-epoch to the PlanRouter and Scatter, then increment counters."""
        if self.plan_router is not None:
            self.plan_router.signal_epoch_end()
        self.scatter_q.put("end_of_epoch")

        self.curr_epoch += 1

    def _add_to_td_list(
        self,
        edges: pd.DataFrame,
        split: Literal["train", "val", "test"],
        mini_batch_idx: int,
        gpu_id: int,
        is_last_macro_of_epoch: bool,
        *,
        macro_batch_idx: int = -1,
        epoch_idx: int = 0,
        is_last_macro_of_split: bool = False,
    ) -> tuple[Meta, SampledLayoutView]:
        """Build a Meta descriptor and sampled-layout sidecar for one micro-batch.

        Converts the raw edge rows into tensors, samples negative destinations,
        and runs the neighbor sampler. The resulting ``SampledLayoutView`` stays
        on the host side and is consumed by ``PlanRouter`` to derive mem/mail/
        nfeat planner inputs.

        Args:
            edges: Rows of the edge DataFrame assigned to this GPU for this step.
            split: String label of the current split.
            mini_batch_idx: Global batch index across all splits and epochs.
            gpu_id: Target GPU rank for this micro-batch.
            macro_batch_idx: Global counter shared by all workers in the step.
            epoch_idx: Zero-based epoch index.
            is_last_macro_of_split: Whether the step ends its split.

        Returns:
            A ``(Meta, SampledLayoutView)`` tuple.
        """
        timing_enabled = self._timing_enabled and split == "train"
        stage_start = time.perf_counter()
        # show edges["time"].values for debug
        # logger.info(f"Edge times: {edges['time'].values}")
        src = torch.from_numpy(edges["src"].to_numpy(copy=True)).to(torch.long)
        dst = torch.from_numpy(edges["dst"].to_numpy(copy=True)).to(torch.long)
        ts = torch.from_numpy(edges["time"].to_numpy(copy=True)).to(self.time_dtype)
        eid = torch.from_numpy(edges["eid"].to_numpy(copy=True)).to(torch.long)
        stage_end = time.perf_counter()
        if timing_enabled:
            self._tensorize_s_curve.append(stage_end - stage_start)
        # show ts content for debug

        # TGL parity mode shares a single RNG stream across train/val/test negatives.
        # Otherwise val/test use the dedicated eval sampler.
        stage_start = stage_end
        neg_sampler = (
            self.neg_sampler_train
            if split == "train" or self.share_neg_sampler_across_splits
            else self.neg_sampler_eval
        )
        neg_sampler.set_batch_context(
            batch_src=edges["src"].to_numpy(dtype=np.int64),
            batch_dst=edges["dst"].to_numpy(dtype=np.int64),
            batch_start_time=float(edges["time"].min()),
            batch_end_time=float(edges["time"].max()),
            batch_ts=edges["time"].to_numpy(dtype=np.float64),
        )
        neg_dst = neg_sampler.sample(src, eid)
        neg_valid_mask = neg_sampler.get_last_negative_mask()
        if neg_valid_mask is not None:
            if neg_valid_mask.shape != neg_dst.shape:
                raise ValueError(
                    "Negative validity mask must match sampled negatives: "
                    f"got {tuple(neg_valid_mask.shape)} and {tuple(neg_dst.shape)}."
                )
        stage_end = time.perf_counter()
        if timing_enabled:
            self._negative_sample_s_curve.append(stage_end - stage_start)

        # handle different model types
        stage_start = stage_end
        roots = torch.cat([src, dst, neg_dst])
        sample_nid, sample_ts = _prepare_sampler_inputs(
            roots=roots,
            ts=ts,
            n_neg=neg_sampler.n_neg,
            root_expansion=self.ngh_sampler.root_expansion,
        )
        sampled_nid, len = self.ngh_sampler.sample(
            root_nid=roots,
            root_ts=ts,
            sample_nid=sample_nid,
            sample_ts=sample_ts,
            pos_eid=eid,
            gpu_id=gpu_id,
        )
        stage_end = time.perf_counter()
        if timing_enabled:
            self._neighborhood_sample_s_curve.append(stage_end - stage_start)
        stage_start = stage_end
        fanout = (
            list(self.ngh_sampler.fanout)
            if self.ngh_sampler.fanout is not None
            else None
        )
        layout = SampledLayoutView(
            nid=sampled_nid,
            n_src=src.shape[0],
            n_neg=neg_sampler.n_neg,
            fanout=fanout,
            root_expansion=self.ngh_sampler.root_expansion,
        )

        meta = Meta(
            split=split,
            flat_graph=None,
            n_src=src.shape[0],
            mini_batch_idx=mini_batch_idx,
            macro_batch_idx=macro_batch_idx,
            epoch_idx=epoch_idx,
            plan_set=None,
            buf_len=len,
            n_neg=neg_sampler.n_neg,
            is_last_macro_of_epoch=is_last_macro_of_epoch,
            is_last_macro_of_split=is_last_macro_of_split,
            neg_valid_mask=neg_valid_mask,
        )
        if timing_enabled:
            self._metadata_s_curve.append(time.perf_counter() - stage_start)

        return meta, layout

    def shutdown(self, graceful: bool, join_timeout_s: float = 5.0) -> None:
        """Best-effort host shutdown that works for both normal and aborted runs."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.stop_ev.set()

        if self.plan_router is not None:
            self.plan_router.shutdown(join_timeout_s=join_timeout_s)

        put_shutdown_sentinel(self.scatter_q, None)
        join_thread(self.scatter, join_timeout_s, logger)

        if self.scatter.is_alive():
            logger.warning(
                "Scatter thread is still alive; forcing process-group teardown."
            )
            destroy_process_group(sync=False, logger_=logger)
            join_thread(self.scatter, 1.0, logger)
        else:
            destroy_process_group(sync=graceful, logger_=logger)

        print("Host shutdown complete.")
        end = time.time()
        print("Total training time: {:.2f} seconds".format(end - self.start_time))
        elapsed = end - self.start_time
        logger.info("Elapsed time: %.2f seconds", elapsed)

        # Force-exit if any thread is stuck on a blocking collective op.
        still_alive: list[threading.Thread] = [
            t for t in (self.scatter,) if t is not None and t.is_alive()
        ]
        if self.plan_router is not None:
            still_alive.extend(
                t
                for t in (
                    self.plan_router.mem_planner,
                    self.plan_router.mail_planner,
                    self.plan_router.nfeat_planner,
                )
                if t is not None and t.is_alive()
            )
        if still_alive:
            logger.warning(
                "Host threads still alive after cleanup: %s — forcing os._exit",
                [t.name for t in still_alive],
            )
            import os

            os._exit(1)
