import logging
import threading
import time
from queue import Queue as Q
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from frost.aliases.host import (
    DynamicPlanOutQ,
    LayoutBundle,
    MetaBundle,
    StaticPlanOutQ,
)
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.datatypes.sampled_layout import SampledLayoutView
from frost.protocols.analyzer_proto import AnalyzerProto
from frost.protocols.ngh_sampler_proto import NghSamplerProto
from frost.runtime._thread_base import (
    destroy_process_group,
    join_thread,
    put_shutdown_sentinel,
)
from frost.runtime.host import PlanRouter, Scatter
from frost.runtime.host._batch_balance import make_balance_plan
from frost.neg_sampler import PrecomputedNegSampler

logger = logging.getLogger(__name__)


def _disable_tqdm_monitor_thread() -> None:
    """Disable tqdm's monitor thread to avoid debugger/threading instability."""
    if getattr(tqdm, "monitor_interval", 0) != 0:
        tqdm.monitor_interval = 0


def _prepare_sampler_inputs(
    roots: torch.Tensor,
    ts: torch.Tensor,
    n_neg: int,
    sample_4_neg: bool,
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

    if sample_4_neg:
        return roots, ts.repeat(root_width)

    return roots[: n_src * 2], ts.repeat(2)


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
        has_nfeat: bool,
        has_efeat: bool,
        has_mem: bool,
        mail_same_plan_as_mem: bool,
        df: pd.DataFrame,
        neg_sampler_train: PrecomputedNegSampler,
        neg_sampler_eval: PrecomputedNegSampler,
        ngh_sampler: NghSamplerProto,
        env: Env,
        start_time: float,
        planner: AnalyzerProto,
        gpu_topo: list,
        batch_size: int,
        time_dtype: torch.dtype,
        n_epoch: int,
        patience: int,
        handle_stale: bool = False,
        show_tqdm: bool = False,
        no_memory_queue_size: int = 4,
        output_dir: str | None = None,
    ) -> None:
        self.neg_sampler_train = neg_sampler_train
        self.neg_sampler_eval = neg_sampler_eval
        self.ngh_sampler = ngh_sampler
        self.start_time = start_time
        self.has_nfeat = has_nfeat
        self.has_efeat = has_efeat
        self.has_mem = has_mem
        self.mail_same_plan_as_mem = mail_same_plan_as_mem
        self.n_gpu = env.n_gpu
        self.batch_size = batch_size
        self.analyzer = planner
        self.show_tqdm = show_tqdm
        self.time_dtype = time_dtype
        if no_memory_queue_size < 1:
            raise ValueError("no_memory_queue_size must be >= 1.")
        self.no_memory_queue_size = no_memory_queue_size

        self.stop_ev = threading.Event()
        self.cur_macro_of_epoch = 0
        self.mini_batch_idx = 0
        self.curr_epoch = 0
        self.best_ap_epoch: int = -1
        self.best_ap_result: Result = Result(
            train_loss=float("inf"),
            val_loss=float("inf"),
            val_auc=0.0,
            val_ap=0.0,
            test_auc=0.0,
            test_ap=0.0,
        )
        self.total_epoch = n_epoch
        self.patience = patience
        self.patience_counter = 0
        self.last_epoch_result: Result | None = None
        self._shutdown_done = False

        self._init_pipeline(env, gpu_topo, handle_stale, output_dir)
        self._init_dataset(df)

    def _init_pipeline(
        self,
        env: Env,
        gpu_topo: list,
        handle_stale: bool,
        output_dir: str | None,
    ) -> None:
        """Fixed linear pipeline: main thread → [PlanRouter.dispatch() →] scatter_q → Scatter.

        The main thread accumulates n_gpu metas per macro-batch, optionally dispatches to
        sub-planner threads via PlanRouter.dispatch(), then puts MetaBundle into scatter_q.
        No dedicated thread for routing — it is inlined into the main thread.
        """
        has_planner = self.has_nfeat or self.has_mem
        stateless_q_size = self.no_memory_queue_size if not self.has_mem else 2

        self.result_q: Q[Result] = Q(maxsize=10)
        self.scatter_q = Q(maxsize=stateless_q_size)

        if has_planner:
            if self.has_mem:
                self.mem_plan_out_q = Q(maxsize=1)
                self.mail_plan_out_q = Q(maxsize=1)
            if self.has_nfeat:
                nfeat_q_size = 1 if self.has_mem else self.no_memory_queue_size
                self.nfeat_plan_out_q = Q(maxsize=nfeat_q_size)

            self.plan_router = PlanRouter(
                gpu_topo=gpu_topo,
                analyzer=self.analyzer,
                mem_plan_out_q=self.mem_plan_out_q,
                mail_plan_out_q=self.mail_plan_out_q,
                nfeat_plan_out_q=self.nfeat_plan_out_q,
                stop_ev=self.stop_ev,
                handle_stale=handle_stale,
                output_dir=output_dir,
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
        t_min = df["time"].min()
        df["time"] = (df["time"] - t_min).astype(np.int64)

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
        val_ap_label = (
            "Val AP(New high)" if res.val_ap > self.best_ap_result.val_ap else "Val AP"
        )

        # Host prints the epoch summary; workers stay silent
        print(
            f"[Epoch {self.curr_epoch}] "
            f"Train Loss: {res.train_loss:.4f} | "
            f"{val_ap_label}: {res.val_ap:.4f} | "
            f"Val AUC: {res.val_auc:.4f} | "
            f"Test AUC: {res.test_auc:.4f} | "
            f"Test AP: {res.test_ap:.4f}"
        )

    def early_stop_check(self, cur_epoch: int) -> bool:
        """
        Check whether early stopping should be triggered and update the best-model record.
        Returns True if training should stop, False to continue.
        """
        res = self.last_epoch_result
        if res is None:
            return False

        is_best = False

        # Check AP improvement
        if res.val_ap > self.best_ap_result.val_ap if self.best_ap_result else -1:
            self.best_ap_epoch = cur_epoch + 1
            self.best_ap_result = res
            is_best = True
        if is_best:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        # Early stopping check
        if self.patience_counter >= self.patience:
            logger.info(
                f"🛑 Early stopping triggered at epoch {cur_epoch + 1}. "
                f"Best AP was {self.best_ap_result.val_ap:.4f} at epoch {self.best_ap_epoch}."
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
            logger.info("Starting Epoch %s", e)
            self._handle_a_split(self.tr_df, "train", cur_epoch=e)
            self._handle_a_split(self.va_df, "val", cur_epoch=e)
            self._handle_a_split(self.te_df, "test", cur_epoch=e)
            self.handle_result_q()
            if self.early_stop_check(e):
                break

        # Print final test results for the best-AP checkpoint
        print(
            f"Best Val AP: {self.best_ap_result.val_ap:.4f} @ Epoch {self.best_ap_epoch} | "
            f"Val AUC: {self.best_ap_result.val_auc:.4f} | "
            f"Test AP: {self.best_ap_result.test_ap:.4f} | "
            f"Test AUC: {self.best_ap_result.test_auc:.4f}"
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

        # Use make_balance_plan for balanced GPU distribution
        df_out, step_id, gpu_id = make_balance_plan(
            df, org_batch_size=self.batch_size, n_gpu=self.n_gpu
        )
        df_out = df_out.assign(step_id=step_id, gpu_id=gpu_id)

        n_macro_of_split = df_out["step_id"].nunique()
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
            meta_bundle: MetaBundle = []
            layout_bundle: LayoutBundle = []
            for gpu, edges in step_df.groupby("gpu_id", sort=False):
                assert isinstance(gpu, int)
                meta, layout = self._add_to_td_list(
                    edges,
                    cur_epoch,
                    cur_macro_of_split,
                    n_macro_of_split,
                    split,
                    self.mini_batch_idx,
                    gpu,
                )
                meta_bundle.append(meta)
                layout_bundle.append(layout)
                self.mini_batch_idx += 1
            if self.plan_router is not None:
                self.plan_router.dispatch(meta_bundle, layout_bundle)
            self.scatter_q.put(meta_bundle)
            self.cur_macro_of_epoch += 1

        if split == "test":
            self._cleanup_per_epoch()

    def _cleanup_per_epoch(self) -> None:
        """Signal end-of-epoch to the PlanRouter and Scatter, then increment counters."""
        if self.plan_router is not None:
            self.plan_router.signal_epoch_end()
        self.scatter_q.put("end_of_epoch")

        self.cur_macro_of_epoch = 0
        self.curr_epoch += 1

    def _add_to_td_list(
        self,
        edges: pd.DataFrame,
        cur_epoch: int,
        cur_macro_of_split: int,
        n_macro_of_split: int,
        split: Literal["train", "val", "test"],
        mini_batch_idx: int,
        gpu_id: int,
    ) -> tuple[Meta, SampledLayoutView]:
        """Build a Meta descriptor and sampled-layout sidecar for one micro-batch.

        Converts the raw edge rows into tensors, samples negative destinations,
        and runs the neighbor sampler. The resulting ``SampledLayoutView`` stays
        on the host side and is consumed by ``PlanRouter`` to derive mem/mail/
        nfeat planner inputs.

        Args:
            edges: Rows of the edge DataFrame assigned to this GPU for this step.
            cur_epoch: Zero-based epoch index.
            cur_macro_of_split: Macro-batch index within the current split.
            n_macro_of_split: Total number of macro-batches in the current split.
            split: String label of the current split.
            mini_batch_idx: Global batch index across all splits and epochs.
            gpu_id: Target GPU rank for this micro-batch.

        Returns:
            A ``(Meta, SampledLayoutView)`` tuple.
        """
        # show edges["time"].values for debug
        # logger.info(f"Edge times: {edges['time'].values}")
        src = torch.from_numpy(edges["src"].to_numpy(copy=True)).to(torch.long)
        dst = torch.from_numpy(edges["dst"].to_numpy(copy=True)).to(torch.long)
        ts = torch.from_numpy(edges["time"].to_numpy(copy=True)).to(self.time_dtype)
        eid = torch.from_numpy(edges["eid"].to_numpy(copy=True)).to(torch.long)
        # show ts content for debug

        # pick sampler based on split (val and test share eval sampler)
        neg_sampler = self.neg_sampler_train if split == "train" else self.neg_sampler_eval
        # Historical / inductive samplers need per-batch context to determine
        # which edges are "current" vs "historical".  Feed it if supported.
        if callable(getattr(neg_sampler, "set_batch_context", None)):
            import numpy as np
            neg_sampler.set_batch_context(
                batch_src=edges["src"].to_numpy(dtype=np.int64),
                batch_dst=edges["dst"].to_numpy(dtype=np.int64),
                batch_start_time=float(edges["time"].min()),
                batch_end_time=float(edges["time"].max()),
                batch_ts=edges["time"].to_numpy(dtype=np.float64),
            )
        neg_dst = neg_sampler.sample(src, eid)

        # handle different model types
        roots = torch.cat([src, dst, neg_dst])
        sample_nid, sample_ts = _prepare_sampler_inputs(
            roots=roots,
            ts=ts,
            n_neg=neg_sampler.n_neg,
            sample_4_neg=self.ngh_sampler.sample_4_neg,
        )
        sampled_nid, len = self.ngh_sampler.sample(
            root_nid=roots,
            root_ts=ts,
            sample_nid=sample_nid,
            sample_ts=sample_ts,
            pos_eid=eid,
            gpu_id=gpu_id,
        )
        layout = SampledLayoutView(
            nid=sampled_nid,
            n_src=src.shape[0],
            n_neg=neg_sampler.n_neg,
            fanout=self.ngh_sampler.fanout,
            sample_4_neg=self.ngh_sampler.sample_4_neg,
        )

        meta = Meta(
            cur_epoch=cur_epoch,
            split=split,
            cur_macro_of_split=cur_macro_of_split,
            cur_macro_of_epoch=self.cur_macro_of_epoch,
            n_macro_of_split=n_macro_of_split,
            flat_graph=None,
            n_src=src.shape[0],
            mini_batch_idx=mini_batch_idx,
            plan_set=None,
            buf_len=len,
            n_neg=neg_sampler.n_neg,
            fanout=self.ngh_sampler.fanout,
        )

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
        still_alive = [
            t for t in (self.scatter,) if t is not None and t.is_alive()
        ]
        if self.plan_router is not None:
            still_alive.extend(
                t for t in (
                    self.plan_router.mem_planner,
                    self.plan_router.mail_planner,
                    self.plan_router.nfeat_planner,
                ) if t is not None and t.is_alive()
            )
        if still_alive:
            logger.warning(
                "Host threads still alive after cleanup: %s — forcing os._exit",
                [t.name for t in still_alive],
            )
            import os
            os._exit(1)
