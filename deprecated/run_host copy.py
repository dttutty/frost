import copy
import math
import time
from multiprocessing import Queue
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from planner import ApanPlanner, sliding_planner
from sampler_core import ParallelSampler
from tqdm import tqdm

from host.comm import Distributor
from host.planner.dynamic_planner import (
    compute_whats_need_by_gpu,
)
from sampler import NegLinkSampler, create_sampler
from schemas import Config, Env, Meta
from utils import analyze_need, extract_raw_data


class Host:
    def __init__(
        self,
        cfg: Config,
        graph: dict,
        df: pd.DataFrame,
        env: Env,
        shared,
        start,
        gloo,
        shared_total_epoch,
    ):
        """
        initialize the host with configuration, graph data, edge dataframe, and environment settings
        """
        self.start_time = start
        self.cfg = cfg
        self.has_nfeat = cfg.t.ndim > 0
        self.has_efeat = cfg.t.edim > 0
        self.has_memory_and_mailbox = cfg.m is not None
        self.env = env
        self.shared_total_epoch = cfg.t.epoch
        self.t_cfg = copy.deepcopy(cfg.t)  # tcfg 会在 run_epoch 中被修改，所以拷贝一份

        # 1. 初始化通信器

        self.gloo = gloo
        self.comm = Distributor(env, group=self.gloo)
        # 2. 拆分数据集
        df["src"] = df["src"].astype(np.int64)
        df["dst"] = df["dst"].astype(np.int64)
        t_min = df["time"].min()
        df["time"] = (df["time"] - t_min).astype(np.float32)
        train_edge_end = df[df["ext_roll"].gt(0)].index[0]
        val_edge_end = df[df["ext_roll"].gt(1)].index[0]
        self.tr_df = df[:train_edge_end]
        self.va_df = df[train_edge_end:val_edge_end]
        self.te_df = df[val_edge_end:]
        gpu_topo = [0, 1]
        self.planner = None
        self.batch_node_state_io_q = Queue()
        self.scatter_q = Queue()
        if self.cfg.m is not None:
            # need planner for memory/mailbox reusing
            if self.cfg.m.deliver_to == "neighbors":
                self.planner = ApanPlanner(
                    gpu_topo,
                    mail_strategy=self.cfg.m.mail_combine,
                )
            else:
                self.planner = sliding_planner(
                    gpu_topo,
                    mail_strategy=self.cfg.m.mail_combine,
                    batch_node_state_io_q=self.batch_node_state_io_q,
                )

        self.scatter = ScatterDistributor(
            n_gpus=self.env.n_gpu,
            group=self.gloo,
            scatter_q=self.scatter_q,
        )

        # 3. 初始化采样器
        self.pos_sampler = None
        if self.cfg.s.sample_support_node:
            logger.debug("Initializing positive sampler...", self.cfg.s.neighbor)
            self.pos_sampler = ParallelSampler(
                graph["indptr"],
                graph["indices"],
                graph["eid"],
                graph["ts"].astype(np.float32),
                self.cfg.s.num_thread,
                1,
                self.cfg.s.layer,
                self.cfg.s.neighbor,
                self.cfg.s.strategy == "recent",
                self.cfg.s.prop_time,
                self.cfg.s.history,
                float(self.cfg.s.duration),
            )

        self.neg_sampler = NegLinkSampler(graph)
        self.e_ns = create_sampler(graph)
        self.curr_minibatch_id = 0
        self.curr_epoch = 0
        self.meta_q: List[Meta] = list()  # 待分发的批次列表
        self.deliver_to = self.cfg.m.deliver_to if self.cfg.m else None
        self.total_epoch = int(shared_total_epoch[0].item())

    def run(self):
        """
        公共的训练入口方法
        """
        for e in range(self.t_cfg.epoch):
            logger.debug(f"Epoch {e}:")
            self._handle_a_split(self.tr_df, "train")
            logger.debug("Validating...")
            self._handle_a_split(self.va_df, "val")
            logger.debug("Testing...")
            self._handle_a_split(self.te_df, "test")

        self._cleanup()
        # wandb.finish()

    @staticmethod
    def plan_gpu_batches(num_samples, target_batch_size, ngpus=1):
        """
        this is to ensure iterations number is divisible by number of gpus
        """
        target_batch_size = int(target_batch_size)  # <--- add
        # This is to ensure the total number of training iterations is an integer multiple of the number of GPUs
        # eg: num_samples = 13345, target_batch_size = 1000, ngpus = 4
        approx_iterations = (
            num_samples // target_batch_size
        )  # approximate number of iterations, 13
        nbatch_per_gpu = max(
            approx_iterations // ngpus, 1
        )  # The approximate number of batches that each GPU needs to process, 3
        iteration_total = nbatch_per_gpu * ngpus  # total batches, 12
        adjusted_batch_size = math.ceil(num_samples / iteration_total)  # 1113
        return iteration_total, adjusted_batch_size

    @staticmethod
    def batches_assign(index, batch_size):
        return index // batch_size

    def _handle_a_split(self, df: pd.DataFrame, split: Literal["train", "val", "test"]):
        # 训练批次规划 (注意这里用 self.tcfg)
        df = df.reset_index(drop=True)

        iteration_total, adjusted_train_batch_size = self.plan_gpu_batches(
            len(df), self.t_cfg.batch_size, ngpus=self.env.n_gpu
        )
        batches_assignment = df.index // adjusted_train_batch_size
        # this will be improved for future random group training
        grouped_df = df.groupby(batches_assignment)
        this_split_n_minibatches = grouped_df.ngroups
        assert this_split_n_minibatches % self.env.n_gpu == 0
        this_split_n_macrobatches = this_split_n_minibatches // self.env.n_gpu
        assert this_split_n_macrobatches >= 3, "need at least 3 macro batches"

        # this is total macro batches for current epoch, including train num_macro_batches, val num_macro_batches and test num_macro_batches
        for this_split_curr_minibatch_id, (group_key, edges) in enumerate(
            tqdm(
                grouped_df,
                total=this_split_n_minibatches,
                desc=f"Host processing {split} split",
            )
        ):
            this_split_curr_macrobatch_bid = (
                this_split_curr_minibatch_id // self.env.n_gpu
            )
            mark = None
            logger.debug(
                "this_split_curr_minibatch_id",
                this_split_curr_minibatch_id,
                "this_split_curr_macrobatch_bid",
                this_split_curr_macrobatch_bid,
                "this_split_n_macrobatches",
                this_split_n_macrobatches,
            )
            if this_split_curr_macrobatch_bid + 1 == this_split_n_macrobatches:
                mark = "last"
            elif this_split_curr_macrobatch_bid == 0:
                mark = "first"
            self._add_to_td_list(
                edges,
                split,
                mark,
                this_split_curr_minibatch_id,
                this_split_curr_macrobatch_bid,
            )
            self.curr_minibatch_id += 1

            # scatter when batches are enough for plan
            # enough for plan: size == n_gpu * 2
            if len(self.meta_q) == self.env.n_gpu * 2:
                if self.planner is not None:
                    self.meta_q = self.planner.plan_next(
                        self.meta_q,
                        is_last_macrobatch=False,
                    )

                meta_2_scatter = self.meta_q[: self.env.n_gpu]
                # only send n_gpus batches
                mini_ids = [td.mini_bid for td in meta_2_scatter]
                split_info = [td.split for td in meta_2_scatter]
                mark_info = [td.mark for td in meta_2_scatter]
                logger.debug(
                    f"Host: Scattering batch {mini_ids} ({split_info}, {mark_info})..."
                )
                assert len(meta_2_scatter) == self.env.n_gpu
                meta_2_scatter = copy.deepcopy(meta_2_scatter)

                self.comm.scatter_send(meta_2_scatter)
                self.meta_q = self.meta_q[self.env.n_gpu :]  # only keep next batches

        if split == "test":
            if self.planner is not None:
                self.plan_lists = self.planner.plan_next(
                    [meta.batch_node_state_io for meta in self.meta_q],
                    is_last_macrobatch=True,
                )
                for i, meta in enumerate(self.meta_q):
                    meta.plan_set = self.plan_lists[i]

            meta_2_scatter = self.meta_q  # only send n_gpus batches

            mini_ids = [td.mini_bid for td in meta_2_scatter]
            split_info = [td.split for td in meta_2_scatter]
            mark_info = [td.mark for td in meta_2_scatter]
            logger.debug(f"Host: Scattering batch {mini_ids} ({split_info}, {mark_info})...")
            meta_2_scatter = copy.deepcopy(meta_2_scatter)

            self.comm.scatter_send(meta_2_scatter)
            self.reset_end_of_epoch_states()

    def reset_end_of_epoch_states(self):
        if self.pos_sampler is not None:
            self.pos_sampler.reset()
        if self.planner is not None:
            self.planner.reset()
        self.meta_q = []
        self.curr_minibatch_id = 0
        self.curr_epoch += 1

    def _add_to_td_list(
        self,
        edges: pd.DataFrame,
        split: Literal["train", "val", "test"],
        mark: Literal["first", "last"] | None,
        curr_mini_bid_of_this_split: int,
        curr_macro_bid_of_this_split: int,
    ):
        assert split is not None
        assert curr_mini_bid_of_this_split is not None
        assert curr_macro_bid_of_this_split is not None
        src_nid = torch.as_tensor(edges.src.values, dtype=torch.long)
        dst_nid = torch.as_tensor(edges.dst.values, dtype=torch.long)
        pos_edge_ts = torch.as_tensor(edges.time.values, dtype=torch.long)

        neg_dst_nid = self.neg_sampler.sample(src_nid, dst_nid)

        pos_edge = torch.stack([src_nid, dst_nid], dim=1)
        pos_edge_id = torch.as_tensor(edges.index.values, dtype=torch.long)

        roots = torch.cat([src_nid, dst_nid, neg_dst_nid])
        ts = torch.cat([pos_edge_ts, pos_edge_ts, pos_edge_ts])

        ts_np = ts.numpy()
        if self.cfg.s.sample_support_node:
            if self.cfg.s.sample_support_for_neg_dst == 0:
                pos_root_np = torch.cat([src_nid, dst_nid]).numpy()
                pos_ts_np = torch.cat([pos_edge_ts, pos_edge_ts]).numpy()
                self.pos_sampler.sample(pos_root_np, pos_ts_np)
            else:
                roots_np = roots.numpy()
                self.pos_sampler.sample(roots_np, ts_np)
            ret = self.pos_sampler.get_ret()
        unique_nodes = None
        if self.cfg.g.arch != "identity":
            blockss = extract_raw_data(ret)
            all_nodes, root_nodes = analyze_need(blockss, self.cfg.s.history)
            root_nodes_for_verify = set(roots.numpy().tolist())
            assert root_nodes == root_nodes_for_verify, (
                "Sampled root nodes do not match the expected root nodes."
            )
        elif self.cfg.g.arch == "identity":
            blockss = None  # for APAN and JODIE
            all_nodes = set(roots.numpy().tolist())
            root_nodes = set(roots.numpy().tolist())
        mdb = None
        if self.cfg.m and self.cfg.m.deliver_to == "neighbors":
            assert self.cfg.s is not None
            assert self.cfg.s.history == 1
            # 只把 history[1] 的反向 block 交给 mailbox, APAN 和 JODIE 专用, only support history=1
            mdb = extract_raw_data(ret)[0]
            from gnn.with_memory import APAN

            APAN.analyze_dynam()

        unique_nodes = compute_whats_need_by_gpu(blockss[0])
        meta = Meta(
            uniq_all=all_nodes,
            uniq_root=root_nodes,
                        gnn_mfgs_raw=gnn_mfgs_raw,=blockss,
            mini_bid=self.curr_minibatch_id,
            macro_bid=self.curr_minibatch_id // self.env.n_gpu,
            pos_edge=pos_edge,
            pos_edge_ts=pos_edge_ts,
            pos_edge_id=pos_edge_id,
            mdb_raw_data=mdb,
            split=split,
            mark=mark,
            curr_mini_bid_of_this_split=curr_mini_bid_of_this_split,
            curr_macro_bid_of_this_split=curr_macro_bid_of_this_split,
            last_epoch=False,
            unique_nodes=unique_nodes,
        )

        if self.curr_epoch + 1 == self.total_epoch:
            meta.last_epoch = True

        self.meta_q.append(meta)

    def _cleanup(self):
        logger.debug("Host: Broadcasting CLEANUP command...")

        # 【新增】: 构造一个由 None 组成的列表，发送给所有 Worker
        # 这会解除 Worker 端 Receiver 线程在 scatter_object_list 上的阻塞
        stop_signal = [None for _ in range(self.env.n_gpu + 1)]

        # 注意：这里需要使用底层的 scatter_object_list，或者确保你的 comm.scatter_send 能处理 None
        # 假设 self.gloo 是你的通信组
        dist.scatter_object_list(
            [None], stop_signal, src=dist.get_rank(), group=self.gloo
        )

        logger.debug("Host: Stop signal sent. Entering final barrier...")
        dist.barrier()  # 建议指定 group=self.gloo 或者 nccl_pg，通常 gloo 用于控制流
        logger.debug("Host: All processes reached the final barrier.")
        end = time.time()

        logger.debug(f"Elapsed time: {end - self.start_time:.2f} seconds")
        dist.destroy_process_group()
