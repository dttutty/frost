import logging
import threading
import time
from queue import Queue as Q  # 这是给多线程用的
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from frost.aliases.host import (
    DynamicPlanOutQ,
    MetaBatcherInQ,
    MetaBundleQ,
    StaticPlanOutQ,
)
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.protocols.analyzer_proto import AnalyzerProto
from frost.protocols.ngh_sampler_proto import NghSamplerProto
from frost.runtime.host import MetaBatcher, Planner, Scatter
from frost.runtime.host._df_split import make_balance_plan
from frost.sampling.neg_sampler import NegLinkSampler

logger = logging.getLogger(__name__)


def _disable_tqdm_monitor_thread() -> None:
    """Disable tqdm's monitor thread to avoid debugger/threading instability."""
    if getattr(tqdm, "monitor_interval", 0) != 0:
        tqdm.monitor_interval = 0


class Host:
    meta_in_q: MetaBatcherInQ
    mem_plan_out_q: DynamicPlanOutQ | None = None
    mail_plan_out_q: DynamicPlanOutQ | None = None
    nfeat_plan_out_q: StaticPlanOutQ | None = None

    meta_bundle_q: MetaBundleQ
    scatter_q: MetaBundleQ

    planner: Planner | None = None

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
        gpu_topo: list[int],
        batch_size: int,
        n_epoch: int,
        patience: int,
    ) -> None:
        """
        initialize the host with configuration, graph data, edge dataframe, and environment settings
        """
        self.neg_sampler = neg_sampler
        self.ngh_sampler = ngh_sampler
        self.start_time = start_time
        self.has_nfeat = has_nfeat
        self.has_efeat = has_efeat
        self.has_mem = has_mem
        self.mail_same_plan_as_mem = mail_same_plan_as_mem
        self.n_gpu = env.n_gpu
        self.batch_size = batch_size  # tcfg 会在 run_epoch 中被修改，所以拷贝一份
        self.analyzer = analyzer

        # 1. 初始化通信器
        self.stop_ev = threading.Event()

        self.meta_in_q = Q(maxsize=4 if not self.has_mem else 2)
        self.meta_bundle_q = Q(maxsize=1)
        self.scatter_q = Q(maxsize=2 if not self.has_mem else 1)
        self.cur_macro_of_epoch = 0

        if self.has_mem:
            self.mem_plan_out_q = Q(maxsize=1)
            self.mail_plan_out_q = Q(maxsize=1)
        if self.has_nfeat:
            self.nfeat_plan_out_q = Q(maxsize=1)

        if self.has_nfeat or self.has_mem:
            self.meta_batcher = MetaBatcher(
                n_gpu=env.n_gpu,
                meta_in_q=self.meta_in_q,
                dest_q=self.meta_bundle_q,
                stop_ev=self.stop_ev,
            )

            self.planner = Planner(
                gpu_topo=gpu_topo,
                in_q=self.meta_bundle_q,
                out_q=self.scatter_q,
                mem_plan_out_q=self.mem_plan_out_q,
                mail_plan_out_q=self.mail_plan_out_q,
                nfeat_plan_out_q=self.nfeat_plan_out_q,
                stop_ev=self.stop_ev,
            )
        else:
            self.meta_batcher = MetaBatcher(
                n_gpu=env.n_gpu,
                meta_in_q=self.meta_in_q,
                dest_q=self.scatter_q,
                stop_ev=self.stop_ev,
            )

        self.result_q: Q[Result] = Q(maxsize=10)
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

        self.patience = patience  # 默认 50
        self.patience_counter = 0
        self.last_epoch_result: Result | None = None
        self.batch_idx = 0

        # 3rd module: scatter sender
        # 3. 初始化采样器
        self._init_dataset(df)

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

        # 将结果存下来，供 early_stop_check 使用
        self.last_epoch_result = res

        # 打印日志 (Host 端统一打印，Worker 端就可以静默了)
        print(
            f"[Epoch {self.curr_epoch}] "
            f"Train Loss: {res.train_loss:.4f} | Val Loss: {res.val_loss:.4f} | "
            f"Val AUC: {res.val_auc:.4f} | Val AP: {res.val_ap:.4f} | "
            f"Test AUC: {res.test_auc:.4f} | Test AP: {res.test_ap:.4f}"
        )

    def early_stop_check(self, cur_epoch: int) -> bool:
        """
        检查是否触发早停，并更新最佳模型记录。
        返回: True 表示需要停止训练，False 表示继续。
        """
        res = self.last_epoch_result
        if res is None:
            return False

        # --- 更新最佳指标 (AUC 和 AP 分别记录) ---
        is_best = False

        # 1. 检查 AP (仅记录，不影响早停，除非你以 AP 为准)
        if res.val_ap > self.best_ap_result.val_ap if self.best_ap_result else -1:
            self.best_ap_epoch = cur_epoch
            self.best_ap_result = res
            is_best = True  # 标记为有好结果
        if is_best:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        # --- 早停判断 ---
        if self.patience_counter >= self.patience:
            logger.info(
                f"🛑 Early stopping triggered at epoch {cur_epoch}. "
                f"Best AP was {self.best_ap_result.val_ap:.4f} at epoch {self.best_ap_epoch}."
            )
            return True

        return False

    def start_all_threads(self) -> None:
        """
        启动 Host 相关的所有线程
        """
        if self.has_nfeat or self.has_mem:
            self.meta_batcher.start()
            if self.planner is not None:
                self.planner.start()
        self.scatter.start()

    def run(self) -> None:
        """
        公共的训练入口方法
        """
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

        # 打印最佳AP对应的Test结果
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
            for gpu, edges in step_df.groupby("gpu_id", sort=False):
                assert isinstance(gpu, int)
                self._add_to_td_list(
                    edges,
                    cur_epoch,
                    cur_macro_of_split,
                    n_macro_of_split,
                    split,
                    self.batch_idx,
                    gpu,
                )
                self.batch_idx += 1
            self.cur_macro_of_epoch += 1

        if split == "test":
            self._cleanup_per_epoch()

    def _cleanup_per_epoch(self) -> None:
        self.meta_in_q.put("end_of_epoch")

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
    ) -> None:
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
            cur_macro_of_epoch=self.cur_macro_of_epoch,  # 这里直接用 macro_of_split 来当 macro_of_epoch，因为我们是按照 split 来划分的
            n_macro_of_split=n_macro_of_split,
            flat_graph=None,  # 这个在 Host 端不需要构建，Worker 端会根据采样结果构建
            n_src=src.shape[0],
            batch_idx=batch_idx,
            plan_set=None,
            access=access,
            buf_len=len,
            fanout=self.ngh_sampler.fanout,
        )

        self.meta_in_q.put(meta)

    def _cleanup(self) -> None:
        self.meta_in_q.put(None)
        self.meta_batcher.join()
        if self.planner is not None:
            self.planner.join()
        print("MetaBatcher and Planner threads have been joined.")
        self.scatter.join()
        print("All threads have been joined. Host cleanup complete.")
        end = time.time()
        print("Total training time: {:.2f} seconds".format(end - self.start_time))
        elapsed = end - self.start_time
        logger.info("Elapsed time: %.2f seconds", elapsed)
