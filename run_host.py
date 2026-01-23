import logging
import math
import threading
import time
from queue import Queue  # 这是给多线程用的
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from host import Distributor, MetaBatcher, Planner, Scatter
from host.meta_batcher import MetaBundle
from host.planner.planner import PlanBundle
from sampler import NegLinkSampler
from sampler_core import ParallelSampler
from schemas.alias import MetaBundleWithoutPlan, MetaBundleWithPlan
from schemas.datatypes.cfg import Config, SampleCFG
from schemas.datatypes.env import Env
from schemas.datatypes.misc import (
    EpochSplitIterInfo,
    ExtFullGraph,
    Meta,
    SampleReturn,
)


logger = logging.getLogger(__name__)


class Host:
    meta_bundle_q: Queue[MetaBundle | None]
    meta_in_q: Queue[Meta | None]
    mem_plan_out_q: Queue[PlanBundle | None] | None
    mail_plan_out_q: Queue[PlanBundle | None] | None
    nfeat_plan_out_q: Queue[PlanBundle | None] | None
    send_q: Queue[MetaBundleWithPlan | MetaBundleWithoutPlan | None]

    def __init__(
        self,
        cfg: Config,
        graph: ExtFullGraph,
        df: pd.DataFrame,
        env: Env,
        start_time: float,
        gloo,
        gnn,
    ):
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
        self.comm = Distributor(env, group=self.gloo)
        gpu_topo = [0, 1]
        self.stop_ev = threading.Event()
        self.sample_support_node = cfg.s.sample_support_node
        self.sample_support_for_neg_dst = cfg.s.sample_support_for_neg_dst
        self.model_name = cfg.t.model_name

        self.meta_in_q = Queue[Meta | None]()
        self.meta_bundle_q = Queue[MetaBundle | None]()
        self.send_q = Queue[MetaBundle | None]()

        self.mem_plan_out_q = Queue() if self.has_mem else None
        self.mail_plan_out_q = Queue() if self.has_mem else None
        self.nfeat_plan_out_q = Queue() if self.has_nfeat else None
        if self.has_nfeat or self.has_mem:
            self.meta_batcher = MetaBatcher(
                n_gpus=env.n_gpu,
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

            self.sem = threading.Semaphore()
            self.scatter = Scatter(
                sem=self.sem,
                n_gpus=env.n_gpu,
                comm=self.comm,
                send_q=self.send_q,
                mem_plan_out_q=self.mem_plan_out_q,
                mail_plan_out_q=self.mail_plan_out_q,
                nfeat_plan_out_q=self.nfeat_plan_out_q,
                stop_ev=self.stop_ev,
            )
        else:
            self.meta_batcher = MetaBatcher(
                n_gpus=env.n_gpu,
                in_q=self.meta_in_q,
                out_q=self.send_q,
                stop_ev=self.stop_ev,
            )
            self.planner = None
            self.scatter = Scatter(
                sem=self.sem,
                n_gpus=env.n_gpu,
                comm=self.comm,
                send_q=self.send_q,
                mem_plan_out_q=self.mem_plan_out_q,
                mail_plan_out_q=self.mail_plan_out_q,
                nfeat_plan_out_q=self.nfeat_plan_out_q,
                stop_ev=self.stop_ev,
            )

        # 3rd module: scatter sender
        # 3. 初始化采样器

        self.curr_mini_of_epoch = 0
        self.total_epoch = cfg.t.epoch
        self.deliver_to = cfg.m.deliver_to if cfg.m else None
        self._init_sampler(graph, cfg.s)
        self._init_dataset(df)

    def _init_dataset(self, df: pd.DataFrame):
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

    def _init_sampler(self, graph: ExtFullGraph, s_cfg: SampleCFG):
        self.neg_sampler = NegLinkSampler(graph)
        self.ngh_sampler = None
        if self.sample_support_node:
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

    def run(self):
        """
        公共的训练入口方法
        """
        for e in range(self.total_epoch):
            logger.debug(f"Epoch {e}:")
            self._handle_a_split(self.tr_df, "train", cur_epoch=e)
            logger.debug("Validating...")
            self._handle_a_split(self.va_df, "val", cur_epoch=e)
            logger.debug("Testing...")
            self._handle_a_split(self.te_df, "test", cur_epoch=e)
        self._cleanup()
        # wandb.finish()

    @staticmethod
    def _plan_gpu_batches(
        num_samples, target_batch_size: int, n_gpu: int
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
    ):
        df = df.reset_index(drop=True)

        iteration_total, adjusted_train_batch_size = self._plan_gpu_batches(
            len(df), self.batch_size, n_gpu=self.n_gpu
        )
        batches_assignment = df.index // adjusted_train_batch_size
        # this will be improved for future random group training
        grouped_df = df.groupby(batches_assignment)
        n_mini = grouped_df.ngroups
        assert n_mini % self.n_gpu == 0
        n_macro_of_split = n_mini // self.n_gpu
        assert n_macro_of_split >= 3, "need at least 3 macro batches"

        # this is total macro batches for current epoch, including train num_macro_batches, val num_macro_batches and test num_macro_batches
        for cur_split_mini, (group_key, edges) in enumerate(
            tqdm(
                grouped_df,
                total=n_mini,
                desc=f"Host processing {split} split",
            )
        ):
            cur_macro_of_split = self.curr_mini_of_epoch // self.n_gpu

            epoch_split_iter_info = EpochSplitIterInfo(
                cur_epoch=cur_epoch,
                n_epoch=self.total_epoch,
                cur_macro_of_split=cur_macro_of_split,
                n_macro_of_split=n_macro_of_split,
                cur_macro_of_epoch=self.curr_mini_of_epoch // self.n_gpu,
                n_macro_of_epoch=None,
                split=split,
            )
            self._add_to_td_list(edges, epoch_split_iter_info)
            self.curr_mini_of_epoch += 1

        if split == "test":
            self._cleanup_per_epoch()

    def _cleanup_per_epoch(self):
        self.meta_in_q.put(None)
        if self.ngh_sampler is not None:
            self.ngh_sampler.reset()
        self.curr_mini_of_epoch = 0
        self.curr_epoch += 1

    def _add_to_td_list(
        self, edges: pd.DataFrame, epoch_split_iter_info: EpochSplitIterInfo
    ):
        src = torch.as_tensor(edges.src.values, dtype=torch.long)
        dst = torch.as_tensor(edges.dst.values, dtype=torch.long)
        ts = torch.as_tensor(edges.time.values, dtype=torch.long)

        # sample neg destination nodes
        neg_dst = self.neg_sampler.sample(src, dst)

        # handle different model types
        if self.sample_history > 1:
            sample_ret = self.gnn.sample(self.ngh_sampler, src, dst, neg_dst, ts)
        else:
            sample_ret = self.gnn.sample(
                self.ngh_sampler, src, dst, neg_dst, ts, self.sample_history
            )

        assert isinstance(sample_ret, SampleReturn), type(sample_ret)
        meta = Meta(
            root_nid=torch.cat([src, dst, neg_dst], dim=0),
            root_ts=ts.repeat(3),
            gnn_mfgs_raw=sample_ret.ngh2root_mfgs_raw,
            pos_edge_id=torch.as_tensor(edges.index.values, dtype=torch.long),
            epoch_split_iter_info=epoch_split_iter_info,
            mdb_raw_data=sample_ret.root2ngh_mfg_raw,
            batch_node_state_io=sample_ret.batch_node_state_io,
            n_src=src.shape[0],
        )

        self.meta_in_q.put(meta)

    def _cleanup(self):
        logger.debug("Host: Broadcasting CLEANUP command...")

        dist.barrier()  # 建议指定 group=self.gloo 或者 nccl_pg，通常 gloo 用于控制流
        logger.debug("Host: All processes reached the final barrier.")
        end = time.time()

        logger.debug(f"Elapsed time: {end - self.start_time:.2f} seconds")
        dist.destroy_process_group()
