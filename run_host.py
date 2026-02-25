import logging
import math
import threading
import time
from queue import Queue as Q  # 这是给多线程用的
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from graph_sampler import SimpleSampler, TemporalGraph, WindowSampler
from tqdm import tqdm

from df_split import make_balance_plan
from gnn.with_memory.apan import APAN
from gnn.with_memory.jodie import JODIE
from gnn.with_memory.tgn import TGN
from host import MetaBatcher, Planner, Scatter
from host.meta_batcher import MetaBundle
from neg_sampler import NegLinkSampler
from schemas.datatypes.cfg import Config, SampleCFG
from schemas.datatypes.env import Env
from schemas.datatypes.misc import EpochSplitIterInfo, ExtFullGraph, Meta, Result
from schemas.datatypes.plan import DyPlan, StPlan
from shm_manager import DoubleBufferedSHM
from tensor_set import TensorSet


logger = logging.getLogger(__name__)


class Host:
    def __init__(
        self,
        cfg: Config,
        graph: ExtFullGraph,
        df: pd.DataFrame,
        env: Env,
        start_time: float,
        gloo: dist.ProcessGroup,
        gnn: torch.nn.Module,
        shm_managers: list[DoubleBufferedSHM],
    ) -> None:
        """
        initialize the host with configuration, graph data, edge dataframe, and environment settings
        """
        self.start_time = start_time
        self.has_nfeat = cfg.t.ndim > 0
        self.has_efeat = cfg.t.edim > 0
        self.has_mem = cfg.m is not None
        self.n_gpu = env.n_gpu
        self.batch_size = cfg.t.batch_size  # tcfg 会在 run_epoch 中被修改，所以拷贝一份
        self.gnn = gnn

        # 1. 初始化通信器
        self.gloo = gloo
        gpu_topo = [0, 1]
        self.stop_ev = threading.Event()
        self.sample_support_node = cfg.s.sample_support_node
        self.sample_support_for_neg_dst = cfg.s.sample_support_for_neg_dst
        self.model_name = cfg.t.model

        self.meta_in_q: Q[Meta | None | Literal["end_of_epoch"]] = Q(maxsize=20)
        self.meta_bundle_q: Q[MetaBundle | None | Literal["end_of_epoch"]] = Q(
            maxsize=10
        )
        self.scatter_q: Q[MetaBundle | None | Literal["end_of_epoch"]] = Q(maxsize=10)
        self.cur_macro_of_epoch = 0

        self.mem_plan_out_q: Q[list[DyPlan] | None] | None = None
        self.mail_plan_out_q: Q[list[DyPlan] | None] | None = None
        self.nfeat_plan_out_q: Q[list[StPlan] | None] | None = None
        self.planner: Planner | None = None
        if self.has_mem:
            self.mem_plan_out_q = Q(maxsize=10)
            self.mail_plan_out_q = Q(maxsize=10)
        if self.has_nfeat:
            self.nfeat_plan_out_q = Q(maxsize=10)

        if self.has_nfeat or self.has_mem:
            self.meta_batcher = MetaBatcher(
                n_gpu=env.n_gpu,
                in_q=self.meta_in_q,
                out_q=self.meta_bundle_q,
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
                in_q=self.meta_in_q,
                out_q=self.scatter_q,
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
            group=gloo,
            stop_ev=self.stop_ev,
            shm_managers=shm_managers,
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

        self.total_epoch = cfg.t.epoch

        self.patience = cfg.t.patience if hasattr(cfg.t, "patience") else 50  # 默认 50
        self.patience_counter = 0
        self.last_epoch_result: Result | None = None

        # 3rd module: scatter sender
        # 3. 初始化采样器
        self._init_sampler(graph, cfg.s)
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
        logger.info(
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

    def _init_sampler(self, graph: ExtFullGraph, s_cfg: SampleCFG) -> None:
        self.neg_sampler = NegLinkSampler(graph)
        self.ngh_sampler: None | SimpleSampler | WindowSampler = None
        # 1. Ensure arrays are strictly 1D, C-contiguous, and use the correct dtypes
        indptr_arr = np.ascontiguousarray(graph.indptr, dtype=np.int64)
        indices_arr = np.ascontiguousarray(graph.indices, dtype=np.int64)
        eid_arr = np.ascontiguousarray(graph.eid, dtype=np.int64)
        ts_arr = np.ascontiguousarray(graph.ts, dtype=np.float32)

        temporal_graph = TemporalGraph(
            indptr=indptr_arr, indices=indices_arr, eid=eid_arr, ts=ts_arr
        )

        if self.sample_support_node:
            if s_cfg.history == 1:
                self.ngh_sampler = SimpleSampler(
                    temporal_graph,
                    s_cfg.num_thread,
                    s_cfg.layer,
                    s_cfg.neighbor,
                    s_cfg.strategy == "recent",
                    s_cfg.prop_time,
                )
            else:
                self.ngh_sampler = WindowSampler(
                    temporal_graph,
                    s_cfg.num_thread,
                    s_cfg.layer,
                    s_cfg.neighbor,
                    s_cfg.strategy == "recent",
                    s_cfg.prop_time,
                    s_cfg.history,
                    float(s_cfg.duration),
                )
            self.sample_history = s_cfg.history

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
        logger.info(
            f"Best Val AP: {self.best_ap_result.val_ap:.4f} @ Epoch {self.best_ap_epoch} | "
            f"Test AP: {self.best_ap_result.test_ap:.4f} | "
            f"Test AUC: {self.best_ap_result.test_auc:.4f}"
        )
        self._cleanup()
        # wandb.finish()

    @staticmethod
    def _plan_gpu_batches(
        num_samples: int,
        target_batch_size: int,
        n_gpu: int,
    ) -> tuple[int, int]:
        """
        this is to ensure iterations number is divisible by number of gpus
        """
        target_batch_size = int(target_batch_size)  # <--- add
        # This is to ensure the total number of training iterations is an integer multiple of the number of GPUs
        # eg: num_samples = 13345, target_batch_size = 1000, n_gpu = 4
        approx_iterations = num_samples // target_batch_size
        # approximate number of iterations, 13
        nbatch_per_gpu = max(approx_iterations // n_gpu, 1)
        # The approximate number of batches that each GPU needs to process, 3
        iteration_total = nbatch_per_gpu * n_gpu  # total batches, 12
        adjusted_batch_size = math.ceil(num_samples / iteration_total)  # 1113
        return iteration_total, adjusted_batch_size

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
            for _, edges in step_df.groupby("gpu_id", sort=False):
                epoch_split_iter_info = EpochSplitIterInfo(
                    cur_epoch=cur_epoch,
                    n_epoch=self.total_epoch,
                    cur_macro_of_split=cast(int, cur_macro_of_split),
                    n_macro_of_split=n_macro_of_split,
                    cur_macro_of_epoch=self.cur_macro_of_epoch,
                    split=split,
                )
                logger.debug(f"epoch_split_iter_info: {epoch_split_iter_info}")
                logger.debug(
                    f"macro_position_of_split: {epoch_split_iter_info.macro_position_of_split}"
                )
                logger.debug(
                    f"is_last_macro_of_epoch: {epoch_split_iter_info.is_last_macro_of_epoch}"
                )
                logger.debug(
                    f"is_last_macro_of_all_epoch: {epoch_split_iter_info.is_last_macro_of_all_epoch}"
                )
                self._add_to_td_list(edges, epoch_split_iter_info)
            self.cur_macro_of_epoch += 1

        if split == "test":
            self._cleanup_per_epoch()

    def _cleanup_per_epoch(self) -> None:
        self.meta_in_q.put("end_of_epoch")
        # if self.ngh_sampler is not None:
        #     self.ngh_sampler.reset()
        self.cur_macro_of_epoch = 0
        self.curr_epoch += 1

    def _add_to_td_list(
        self, edges: pd.DataFrame, epoch_split_iter_info: EpochSplitIterInfo
    ) -> None:
        # show edges["time"].values for debug
        # logger.info(f"Edge times: {edges['time'].values}")
        src = torch.as_tensor(edges["src"].values, dtype=torch.long)
        dst = torch.as_tensor(edges["dst"].values, dtype=torch.long)
        ts = torch.as_tensor(edges["time"].values, dtype=torch.float32)
        # show ts content for debug

        # sample neg destination nodes
        neg_dst = self.neg_sampler.sample(src, dst)

        # handle different model types
        n2r_meta = r2n_meta = state = None

        if isinstance(self.gnn, APAN):
            from gnn.with_memory.apan import (
                analyze_all_n_root as apan_analyze_all_n_root,
                analyze_need as apan_analyze_need,
                sample as apan_sample,
            )

            assert self.ngh_sampler is not None
            r2n_meta = apan_sample(self.ngh_sampler, src, dst, ts)
            all_nodes, root_nodes = apan_analyze_all_n_root(r2n_meta)
            state = apan_analyze_need(all_nodes, root_nodes, neg_dst)
        elif isinstance(self.gnn, JODIE):
            from gnn.with_memory.jodie import analyze_need as jodie_analyze_need

            roots = torch.cat([src, dst, neg_dst])
            root_nodes = TensorSet(roots)
            state = jodie_analyze_need(root_nodes)

        elif isinstance(self.gnn, TGN):
            from gnn.with_memory.tgn import (
                analyze_all_n_root as tgn_analyze_all_n_root,
                analyze_need as tgn_analyze_need,
            )
            from host.sample import sample

            assert self.ngh_sampler is not None

            n2r_meta = sample(self.ngh_sampler, src, dst, neg_dst, ts)
            # todo: print src, dst, neg_dst, ts
            # print(
            #     f"src: {src.shape}, dst: {dst.shape}, neg_dst: {neg_dst.shape}, ts: {ts.shape}"
            # )
            assert len(n2r_meta) == 1, "TGN only supports 1 layer currently"

            all_nodes, root_nodes = tgn_analyze_all_n_root(n2r_meta[0], src.shape[0])
            state = tgn_analyze_need(all_nodes, src, dst)
        else:
            from host.sample import (
                analyze_need as general_analyze_need,
                sample as general_sample,
            )

            assert self.ngh_sampler is not None

            n2r_meta = general_sample(self.ngh_sampler, src, dst, neg_dst, ts)
            state = general_analyze_need(n2r_meta, self.sample_history)

        meta = Meta(
            root_nid=torch.cat([src, dst, neg_dst], dim=0),
            root_ts=ts.repeat(3),
            n2r_meta=n2r_meta,
            pos_edge_id=torch.as_tensor(edges["eid"].values, dtype=torch.long),
            m_state=epoch_split_iter_info,
            r2n_meta=r2n_meta,
            state=state,
            n_src=src.shape[0],
            batch_idx=epoch_split_iter_info.cur_macro_of_epoch,
        )

        self.meta_in_q.put(meta)

    def _cleanup(self) -> None:
        self.meta_in_q.put(None)
        self.meta_batcher.join()
        if self.planner is not None:
            self.planner.join()
        self.scatter.join()
        end = time.time()
        elapsed = end - self.start_time
        logger.info("Elapsed time: %.2f seconds", elapsed)
