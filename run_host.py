import logging
import math
import threading
import time
from queue import Queue as Q  # 这是给多线程用的
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from gnn.with_memory.apan import APAN
from gnn.with_memory.jodie import JODIE
from gnn.with_memory.tgn import TGN
from host import MetaBatcher, Planner, Scatter
from host.meta_batcher import MetaBundle
from neg_sampler import NegLinkSampler
from sampler_core import ParallelSampler
from schemas.datatypes.cfg import Config, SampleCFG
from schemas.datatypes.env import Env
from schemas.datatypes.misc import EpochSplitIterInfo, ExtFullGraph, Meta, Result
from schemas.datatypes.plan import DyPlan, StPlan
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
    ) -> None:
        """
        initialize the host with configuration, graph data, edge dataframe, and environment settings
        """
        logger.debug(
            f"Host: initializing with n_gpu={env.n_gpu}, batch_size={cfg.t.batch_size}"
        )
        self.start_time = start_time
        self.has_nfeat = cfg.t.ndim > 0
        self.has_efeat = cfg.t.edim > 0
        self.has_mem = cfg.m is not None
        logger.debug(
            f"Host: has_nfeat={self.has_nfeat}, has_efeat={self.has_efeat}, has_mem={self.has_mem}"
        )
        self.n_gpu = env.n_gpu
        self.batch_size = cfg.t.batch_size  # tcfg 会在 run_epoch 中被修改，所以拷贝一份
        self.gnn = gnn

        # 1. 初始化通信器
        logger.debug("Host: initializing communication queues")
        self.gloo = gloo
        gpu_topo = [0, 1]
        self.stop_ev = threading.Event()
        self.sample_support_node = cfg.s.sample_support_node
        self.sample_support_for_neg_dst = cfg.s.sample_support_for_neg_dst
        self.model_name = cfg.t.model

        self.meta_in_q: Q[Meta | None | Literal["end_of_epoch"]] = Q(maxsize=10)
        self.meta_bundle_q: Q[MetaBundle | None | Literal["end_of_epoch"]] = Q()
        self.send_q: Q[MetaBundle | None | Literal["end_of_epoch"]] = Q()

        self.mem_plan_out_q: Q[list[DyPlan] | None] | None = None
        self.mail_plan_out_q: Q[list[DyPlan] | None] | None = None
        self.nfeat_plan_out_q: Q[list[StPlan] | None] | None = None
        self.planner: Planner | None = None
        if self.has_mem:
            self.mem_plan_out_q = Q()
            self.mail_plan_out_q = Q()
        if self.has_nfeat:
            self.nfeat_plan_out_q = Q()

        if self.has_nfeat or self.has_mem:
            logger.debug("Host: initializing MetaBatcher and Planner")
            self.meta_batcher = MetaBatcher(
                n_gpu=env.n_gpu,
                in_q=self.meta_in_q,
                out_q=self.meta_bundle_q,
                stop_ev=self.stop_ev,
            )

            self.planner = Planner(
                gpu_topo=gpu_topo,
                in_q=self.meta_bundle_q,
                out_q=self.send_q,
                mem_plan_out_q=self.mem_plan_out_q,
                mail_plan_out_q=self.mail_plan_out_q,
                nfeat_plan_out_q=self.nfeat_plan_out_q,
                stop_ev=self.stop_ev,
            )
        else:
            logger.debug("Host: initializing MetaBatcher only (no Planner)")
            self.meta_batcher = MetaBatcher(
                n_gpu=env.n_gpu,
                in_q=self.meta_in_q,
                out_q=self.send_q,
                stop_ev=self.stop_ev,
            )

        logger.debug("Host: initializing Scatter")
        self.result_q: Q[Result] = Q()
        self.scatter = Scatter(
            n_gpu=env.n_gpu,
            in_q=self.send_q,
            mem_plan_out_q=self.mem_plan_out_q,
            mail_plan_out_q=self.mail_plan_out_q,
            nfeat_plan_out_q=self.nfeat_plan_out_q,
            env=env,
            result_q=self.result_q,
            group=gloo,
            stop_ev=self.stop_ev,
        )
        self.curr_epoch = 0
        self.best_ap_epoch = {
            "val_ap": 0.0,
            "val_auc": 0.0,
            "epoch": -1,
            "test_ap": 0.0,
            "test_auc": 0.0,
        }
        self.best_auc_epoch = {
            "val_ap": 0.0,
            "val_auc": 0.0,
            "epoch": -1,
            "test_ap": 0.0,
            "test_auc": 0.0,
        }
        self.state = {
            "total_epoch": math.inf,
            "done_epoch": 0,
            "total_macro_batch": math.inf,
            "done_macro_batch": 0,
        }

        self.total_epoch = cfg.t.epoch

        self.patience = cfg.t.patience if hasattr(cfg.t, "patience") else 50  # 默认 50
        self.patience_counter = 0
        self.last_epoch_result: Result | None = None

        # 3rd module: scatter sender
        # 3. 初始化采样器
        logger.debug("Host: initializing sampler")
        self._init_sampler(graph, cfg.s)
        logger.debug("Host: initializing dataset")
        self._init_dataset(df)
        logger.debug("Host: initialization complete")

    def _init_dataset(self, df: pd.DataFrame) -> None:
        logger.debug(f"Host: preprocessing dataframe, total edges={len(df)}")
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
        logger.debug(
            f"Host: dataset split - train={len(self.tr_df)}, val={len(self.va_df)}, test={len(self.te_df)}"
        )

    def handle_result_q(self) -> None:
        logger.debug("Host: waiting for result from result_q")
        res = self.result_q.get(block=True)
        logger.debug(f"Host: received result for epoch {self.state['done_epoch']}")

        # 将结果存下来，供 early_stop_check 使用
        self.last_epoch_result = res

        # 打印日志 (Host 端统一打印，Worker 端就可以静默了)
        logger.info(
            f"[Epoch {self.state['done_epoch']}] "
            f"Train Loss: {res.train_loss:.4f} | Val Loss: {res.val_loss:.4f} | "
            f"Val AUC: {res.val_auc:.4f} | Val AP: {res.val_ap:.4f}"
        )

    def early_stop_check(self, cur_epoch: int) -> bool:
        """
        检查是否触发早停，并更新最佳模型记录。
        返回: True 表示需要停止训练，False 表示继续。
        """
        logger.debug(
            f"Host: early_stop_check for epoch {cur_epoch}, patience_counter={self.patience_counter}"
        )
        res = self.last_epoch_result
        if res is None:
            logger.debug("Host: no result yet, skipping early stop check")
            return False

        # --- 更新最佳指标 (AUC 和 AP 分别记录) ---

        # 1. 检查 AUC
        if res.val_auc > self.best_auc_epoch["val_auc"]:
            logger.debug(
                f"Host: new best AUC {res.val_auc:.4f} > {self.best_auc_epoch['val_auc']:.4f}"
            )
            self.best_auc_epoch = {
                "val_ap": res.val_ap,
                "val_auc": res.val_auc,
                "epoch": cur_epoch,
                "test_ap": res.test_ap,
                "test_auc": res.test_auc,
            }
            # 只有当主要指标 (通常是 AUC) 提升时，才重置耐心值
            self.patience_counter = 0
            logger.info(f"🌟 New Best AUC: {res.val_auc:.4f} @ Epoch {cur_epoch}")
            # TODO: 这里可以添加 save_checkpoint()
        else:
            self.patience_counter += 1
            logger.debug(
                f"Host: no AUC improvement, patience_counter={self.patience_counter}/{self.patience}"
            )

        # 2. 检查 AP (仅记录，不影响早停，除非你以 AP 为准)
        if res.val_ap > self.best_ap_epoch["val_ap"]:
            logger.debug(
                f"Host: new best AP {res.val_ap:.4f} > {self.best_ap_epoch['val_ap']:.4f}"
            )
            self.best_ap_epoch = {
                "val_ap": res.val_ap,
                "val_auc": res.val_auc,
                "epoch": cur_epoch,
                "test_ap": res.test_ap,
                "test_auc": res.test_auc,
            }
            logger.info(f"✨ New Best AP: {res.val_ap:.4f} @ Epoch {cur_epoch}")

        # --- 早停判断 ---
        if self.patience_counter >= self.patience:
            logger.warning(
                f"🛑 Early stopping triggered at epoch {cur_epoch}. "
                f"Best AUC was {self.best_auc_epoch['val_auc']:.4f} at epoch {self.best_auc_epoch['epoch']}."
            )
            return True

        return False

    def _init_sampler(self, graph: ExtFullGraph, s_cfg: SampleCFG) -> None:
        logger.debug("Host: initializing negative link sampler")
        self.neg_sampler = NegLinkSampler(graph)
        self.ngh_sampler = None
        if self.sample_support_node:
            logger.debug(
                f"Host: initializing neighbor sampler with {s_cfg.num_thread} threads, {s_cfg.layer} layers"
            )
            self.ngh_sampler = ParallelSampler(
                graph.indptr,
                graph.indices,
                graph.eid,
                graph.ts,
                s_cfg.num_thread,
                1,
                s_cfg.layer,
                s_cfg.neighbor,
                s_cfg.strategy == "recent",
                s_cfg.prop_time,
                s_cfg.history,
                float(s_cfg.duration),
            )
            self.sample_history = s_cfg.history
            logger.debug(
                f"Host: neighbor sampler initialized with history={self.sample_history}"
            )

    def start_all_threads(self) -> None:
        """
        启动 Host 相关的所有线程
        """
        logger.debug("Host: starting all threads")
        if self.has_nfeat or self.has_mem:
            logger.debug("Host: starting MetaBatcher")
            self.meta_batcher.start()
            if self.planner is not None:
                logger.debug("Host: starting Planner")
                self.planner.start()
        logger.debug("Host: starting Scatter")
        self.scatter.start()
        logger.debug("Host: all threads started")

    def start(self) -> None:
        """
        公共的训练入口方法
        """
        logger.debug(f"Host: starting training for {self.total_epoch} epochs")
        for e in range(self.total_epoch):
            logger.debug(f"Host: starting epoch {e}")
            logger.debug(f"Epoch {e}:")
            logger.debug("Host: processing training split")
            self._handle_a_split(self.tr_df, "train", cur_epoch=e)
            logger.debug("Validating...")
            self._handle_a_split(self.va_df, "val", cur_epoch=e)
            logger.debug("Testing...")
            self._handle_a_split(self.te_df, "test", cur_epoch=e)
            logger.debug(f"Host: epoch {e} splits complete, handling results")
            self.handle_result_q()
            if self.early_stop_check(e):
                logger.debug(f"Host: early stopping at epoch {e}")
                break
        logger.debug("Host: training complete, cleaning up")
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
        logger.debug(
            f"Host: planned batches - samples={num_samples}, target_bs={target_batch_size}, "
            f"n_gpu={n_gpu}, iterations={iteration_total}, adjusted_bs={adjusted_batch_size}"
        )
        return iteration_total, adjusted_batch_size

    def _handle_a_split(
        self, df: pd.DataFrame, split: Literal["train", "val", "test"], cur_epoch: int
    ) -> None:
        logger.debug(
            f"Host: handling split={split}, epoch={cur_epoch}, total_edges={len(df)}"
        )
        df = df.reset_index(names=["eid"])

        _, new_batch_size = self._plan_gpu_batches(
            len(df), self.batch_size, n_gpu=self.n_gpu
        )
        batches_assignment = df.index // new_batch_size
        # this will be improved for future random group training
        grouped_df = df.groupby(batches_assignment)
        n_mini = grouped_df.ngroups
        assert n_mini % self.n_gpu == 0
        n_macro_of_split = n_mini // self.n_gpu
        assert n_macro_of_split >= 3, "need at least 3 macro batches"
        logger.debug(
            f"Host: split={split} has {n_mini} mini batches, {n_macro_of_split} macro batches"
        )

        # this is total macro batches for current epoch, including train num_macro_batches, val num_macro_batches and test num_macro_batches
        batch_count = 0
        for _, edges in tqdm(
            grouped_df, total=n_mini, desc=f"Host processing {split} split"
        ):
            batch_count += 1
            cur_macro_of_split = self.curr_mini_of_epoch // self.n_gpu

            epoch_split_iter_info = EpochSplitIterInfo(
                cur_epoch=cur_epoch,
                n_epoch=self.total_epoch,
                cur_macro_of_split=cur_macro_of_split,
                n_macro_of_split=n_macro_of_split,
                cur_macro_of_epoch=self.curr_mini_of_epoch // self.n_gpu,
                split=split,
            )
            logger.debug(
                f"Host: processing batch {batch_count}/{n_mini} for split={split}, "
                f"macro={cur_macro_of_split}/{n_macro_of_split}, edges={len(edges)}"
            )
            self._add_to_td_list(edges, epoch_split_iter_info)
            self.curr_mini_of_epoch += 1

        logger.debug(f"Host: split={split} complete, processed {batch_count} batches")
        if split == "test":
            logger.debug("Host: test split complete, cleaning up epoch")
            self._cleanup_per_epoch()

    def _cleanup_per_epoch(self) -> None:
        logger.debug(f"Host: cleaning up epoch {self.curr_epoch}")
        self.meta_in_q.put("end_of_epoch")
        if self.ngh_sampler is not None:
            logger.debug("Host: resetting neighbor sampler")
            self.ngh_sampler.reset()
        self.curr_mini_of_epoch = 0
        self.curr_epoch += 1
        logger.debug(f"Host: epoch cleanup complete, moving to epoch {self.curr_epoch}")

    def _add_to_td_list(
        self, edges: pd.DataFrame, epoch_split_iter_info: EpochSplitIterInfo
    ) -> None:
        logger.debug(
            f"Host: adding batch to queue, edges={len(edges)}, split={epoch_split_iter_info.split}"
        )
        src = torch.as_tensor(edges["src"].values, dtype=torch.long)
        dst = torch.as_tensor(edges["dst"].values, dtype=torch.long)
        ts = torch.as_tensor(edges["time"].values, dtype=torch.long)

        # sample neg destination nodes
        logger.debug("Host: sampling negative destinations")
        neg_dst = self.neg_sampler.sample(src, dst)
        logger.debug(f"Host: sampled {neg_dst.numel()} negative destinations")

        # handle different model types
        n2r_meta = r2n_meta = state = None
        model_type = type(self.gnn).__name__
        logger.debug(f"Host: handling model type={model_type}")

        if isinstance(self.gnn, APAN):
            logger.debug("Host: sampling for APAN model")
            from gnn.with_memory.apan import (
                analyze_all_n_root as apan_analyze_all_n_root,
                analyze_need as apan_analyze_need,
                sample as apan_sample,
            )

            assert self.ngh_sampler is not None
            r2n_meta = apan_sample(self.ngh_sampler, src, dst, ts)
            all_nodes, root_nodes = apan_analyze_all_n_root(r2n_meta)
            state = apan_analyze_need(all_nodes, root_nodes, neg_dst)
            logger.debug(
                f"Host: APAN analysis complete, all_nodes={all_nodes.numel()}, root_nodes={root_nodes.numel()}"
            )
        elif isinstance(self.gnn, JODIE):
            logger.debug("Host: analyzing for JODIE model")
            from gnn.with_memory.jodie import analyze_need as jodie_analyze_need

            roots = torch.cat([src, dst, neg_dst])
            root_nodes = TensorSet(roots)
            state = jodie_analyze_need(root_nodes)
            logger.debug(
                f"Host: JODIE analysis complete, root_nodes={root_nodes.numel()}"
            )

        elif isinstance(self.gnn, TGN):
            logger.debug("Host: sampling for TGN model")
            from gnn.with_memory.tgn import (
                analyze_all_n_root as tgn_analyze_all_n_root,
                analyze_need as tgn_analyze_need,
            )
            from host.sample import sample

            assert self.ngh_sampler is not None

            n2r_meta = sample(self.ngh_sampler, src, dst, neg_dst, ts)
            assert len(n2r_meta) == 1, "TGN only supports 1 layer currently"

            all_nodes, root_nodes = tgn_analyze_all_n_root(n2r_meta[0])
            state = tgn_analyze_need(all_nodes, src, dst)
            logger.debug(
                f"Host: TGN analysis complete, all_nodes={all_nodes.numel()}, root_nodes={root_nodes.numel()}"
            )
        else:
            logger.debug("Host: sampling for general model")
            from host.sample import (
                analyze_need as general_analyze_need,
                sample as general_sample,
            )

            assert self.ngh_sampler is not None

            n2r_meta = general_sample(self.ngh_sampler, src, dst, neg_dst, ts)
            state = general_analyze_need(n2r_meta, self.sample_history)
            logger.debug(
                f"Host: general model sampling complete, n2r_meta layers={len(n2r_meta) if n2r_meta else 0}"
            )

        meta = Meta(
            root_nid=torch.cat([src, dst, neg_dst], dim=0),
            root_ts=ts.repeat(3),
            n2r_meta=n2r_meta,
            pos_edge_id=torch.as_tensor(edges["eid"].values, dtype=torch.long),
            m_state=epoch_split_iter_info,
            r2n_meta=r2n_meta,
            state=state,
            n_src=src.shape[0],
        )

        logger.debug(
            f"Host: putting meta to queue, root_nid size={meta.root_nid.numel()}"
        )
        self.meta_in_q.put(meta)

    def _cleanup(self) -> None:
        logger.debug("Host: starting cleanup")
        self.meta_in_q.put(None)
        logger.debug("Host: waiting for MetaBatcher to join")
        self.meta_batcher.join()
        if self.planner is not None:
            logger.debug("Host: waiting for Planner to join")
            self.planner.join()
        logger.debug("Host: waiting for Scatter to join")
        self.scatter.join()
        end = time.time()
        elapsed = end - self.start_time
        logger.debug(f"Elapsed time: {elapsed:.2f} seconds")
        logger.debug("Host: cleanup complete")
