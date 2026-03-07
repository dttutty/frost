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
    AccessBundle,
    DynamicPlanOutQ,
    MetaBundle,
    StaticPlanOutQ,
)
from frost.datatypes.access import Access
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.protocols.analyzer_proto import AnalyzerProto
from frost.protocols.ngh_sampler_proto import NghSamplerProto
from frost.runtime.host import PlanRouter, Scatter
from frost.runtime.host._batch_balance import make_balance_plan
from frost.sampling.neg_sampler import NegLinkSampler

logger = logging.getLogger(__name__)


def _disable_tqdm_monitor_thread() -> None:
    """Disable tqdm's monitor thread to avoid debugger/threading instability."""
    if getattr(tqdm, "monitor_interval", 0) != 0:
        tqdm.monitor_interval = 0


class Host:
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
        neg_sampler: NegLinkSampler,
        ngh_sampler: NghSamplerProto,
        env: Env,
        start_time: float,
        analyzer: AnalyzerProto,
        gpu_topo: list,
        batch_size: int,
        n_epoch: int,
        patience: int,
        handle_stale: bool = False,
    ) -> None:
        self.neg_sampler = neg_sampler
        self.ngh_sampler = ngh_sampler
        self.start_time = start_time
        self.has_nfeat = has_nfeat
        self.has_efeat = has_efeat
        self.has_mem = has_mem
        self.mail_same_plan_as_mem = mail_same_plan_as_mem
        self.n_gpu = env.n_gpu
        self.batch_size = batch_size
        self.analyzer = analyzer

        self.stop_ev = threading.Event()
        self.cur_macro_of_epoch = 0
        self.batch_idx = 0
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

        self._init_pipeline(env, gpu_topo, handle_stale)
        self._init_dataset(df)

    def _init_pipeline(self, env: Env, gpu_topo: list, handle_stale: bool) -> None:
        """Fixed linear pipeline: main thread → [PlanRouter.dispatch() →] scatter_q → Scatter.

        The main thread accumulates n_gpu metas per macro-batch, optionally dispatches to
        sub-planner threads via PlanRouter.dispatch(), then puts MetaBundle into scatter_q.
        No dedicated thread for routing — it is inlined into the main thread.
        """
        has_planner = self.has_nfeat or self.has_mem

        self.result_q: Q[Result] = Q(maxsize=10)
        self.scatter_q = Q(maxsize=2)

        if has_planner:
            if self.has_mem:
                self.mem_plan_out_q = Q(maxsize=1)
                self.mail_plan_out_q = Q(maxsize=1)
            if self.has_nfeat:
                self.nfeat_plan_out_q = Q(maxsize=1)

            self.plan_router = PlanRouter(
                gpu_topo=gpu_topo,
                mem_plan_out_q=self.mem_plan_out_q,
                mail_plan_out_q=self.mail_plan_out_q,
                nfeat_plan_out_q=self.nfeat_plan_out_q,
                stop_ev=self.stop_ev,
                handle_stale=handle_stale,
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
        # preprocess dataframe
        df["src"] = df["src"].astype(np.int64)
        df["dst"] = df["dst"].astype(np.int64)
        t_min = df["time"].min()
        df["time"] = (df["time"] - t_min).astype(np.float32)

        train_edge_end = df[df["ext_roll"].gt(0)].index[0]
        val_edge_end = df[df["ext_roll"].gt(1)].index[0]
        self.tr_df = df[:train_edge_end]
        self.va_df = df[train_edge_end:val_edge_end]
        self.te_df = df[val_edge_end:]

    def handle_result_q(self) -> None:
        res = self.result_q.get(block=True)

        # Cache result so early_stop_check can read it
        self.last_epoch_result = res

        # Host prints the epoch summary; workers stay silent
        print(
            f"[Epoch {self.curr_epoch}] "
            f"Train Loss: {res.train_loss:.4f} | Val Loss: {res.val_loss:.4f} | "
            f"Val AUC: {res.val_auc:.4f} | Val AP: {res.val_ap:.4f} | "
            f"Test AUC: {res.test_auc:.4f} | Test AP: {res.test_ap:.4f}"
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
            self.best_ap_epoch = cur_epoch
            self.best_ap_result = res
            is_best = True
        if is_best:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        # Early stopping check
        if self.patience_counter >= self.patience:
            logger.info(
                f"🛑 Early stopping triggered at epoch {cur_epoch}. "
                f"Best AP was {self.best_ap_result.val_ap:.4f} at epoch {self.best_ap_epoch}."
            )
            return True

        return False

    def start_all_threads(self) -> None:
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
            f"Test AP: {self.best_ap_result.test_ap:.4f} | "
            f"Test AUC: {self.best_ap_result.test_auc:.4f}"
        )
        self._cleanup()
        # wandb.finish()

    def _handle_a_split(
        self, df: pd.DataFrame, split: Literal["train", "val", "test"], cur_epoch: int
    ) -> None:
        if "ext_roll" in df.columns:
            ext_roll = df["ext_roll"]
            is_valid_split = (
                (split == "train" and ext_roll.eq(0).all())
                or (split == "val" and ext_roll.eq(1).all())
                or (split == "test" and ext_roll.ge(2).all())
            )
            if not is_valid_split:
                found_rolls = sorted(ext_roll.dropna().unique().tolist())
                raise ValueError(
                    f"Split contamination detected for split={split}. "
                    f"Found ext_roll values: {found_rolls}"
                )

        df = df.reset_index(names=["eid"])

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
        ):
            assert isinstance(cur_macro_of_split, int)
            meta_bundle: MetaBundle = []
            access_bundle: AccessBundle = []
            for gpu, edges in step_df.groupby("gpu_id", sort=False):
                assert isinstance(gpu, int)
                meta, access = self._add_to_td_list(
                    edges,
                    cur_epoch,
                    cur_macro_of_split,
                    n_macro_of_split,
                    split,
                    self.batch_idx,
                    gpu,
                )
                meta_bundle.append(meta)
                access_bundle.append(access)
                self.batch_idx += 1
            if self.plan_router is not None:
                self.plan_router.dispatch(meta_bundle, access_bundle)
            self.scatter_q.put(meta_bundle)
            self.cur_macro_of_epoch += 1

        if split == "test":
            self._cleanup_per_epoch()

    def _cleanup_per_epoch(self) -> None:
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
        batch_idx: int,
        gpu_id: int,
    ) -> tuple[Meta, Access | None]:
        # show edges["time"].values for debug
        # logger.info(f"Edge times: {edges['time'].values}")
        src = (
            torch.as_tensor(edges["src"].values, dtype=torch.long).clone().contiguous()
        )
        dst = (
            torch.as_tensor(edges["dst"].values, dtype=torch.long).clone().contiguous()
        )
        ts = (
            torch.as_tensor(edges["time"].values, dtype=torch.float32)
            .clone()
            .contiguous()
        )
        eid = (
            torch.as_tensor(edges["eid"].values, dtype=torch.long).clone().contiguous()
        )
        # show ts content for debug

        # sample neg destination nodes
        neg_dst = self.neg_sampler.sample(src)

        # handle different model types
        roots = torch.cat([src, dst, neg_dst])
        uniq_all, len, valid_all = self.ngh_sampler.sample(
            nid=roots, ts=ts, pos_eid=eid, gpu_id=gpu_id
        )

        access = None
        if self.analyzer is not None:
            access = self.analyzer.count_access(
                src,
                dst,
                uniq_all.data,
                fanout=self.ngh_sampler.fanout,
                valid_all=valid_all,
            )

        meta = Meta(
            cur_epoch=cur_epoch,
            split=split,
            cur_macro_of_split=cur_macro_of_split,
            cur_macro_of_epoch=self.cur_macro_of_epoch,
            n_macro_of_split=n_macro_of_split,
            flat_graph=None,
            n_src=src.shape[0],
            batch_idx=batch_idx,
            plan_set=None,
            buf_len=len,
            fanout=self.ngh_sampler.fanout,
        )

        return meta, access

    def _cleanup(self) -> None:
        if self.plan_router is not None:
            self.plan_router.shutdown()
        self.scatter_q.put(None)
        self.scatter.join()
        print("All threads have been joined. Host cleanup complete.")
        end = time.time()
        print("Total training time: {:.2f} seconds".format(end - self.start_time))
        elapsed = end - self.start_time
        logger.info("Elapsed time: %.2f seconds", elapsed)
