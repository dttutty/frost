import math
import threading
import time
from multiprocessing import Queue
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sampler_core import ParallelSampler
from tqdm import tqdm

from host import Distributor, MetaBatcher, Planner, ScatterSender
from sampler import NegLinkSampler
from schemas import Config, Env, Meta, SampleCFG
from schemas.datatypes import EpochSplitIterInfo, ExtFullGraph
from utils import analyze_blocks, extract_raw_data


class Host:
    def __init__(
        self,
        cfg: Config,
        graph: ExtFullGraph,
        df: pd.DataFrame,
        env: Env,
        start_time: float,
        gloo,
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

        # 1. 初始化通信器

        self.gloo = gloo
        self.comm = Distributor(env, group=self.gloo)
        gpu_topo = [0, 1]
        self.stop_ev = threading.Event()
        self.sample_support_node = cfg.s.sample_support_node
        self.sample_support_for_neg_dst = cfg.s.sample_support_for_neg_dst
        self.model_name = cfg.t.model_name

        self.meta_in_q: Queue[List[Meta]] = Queue()
        self.planner = None
        self.send_q: Queue[List[Meta]] = Queue()
        if self.has_nfeat or self.has_mem:
            self.meta_bundle_q: Queue[List[Meta]] = Queue()
            self.meta_batcher = MetaBatcher(
                n_gpus=env.n_gpu,
                in_q=self.meta_in_q,
                out_q=self.meta_bundle_q,
                stop_ev=self.stop_ev,
            )

            mem_plan_out_q = Queue() if self.has_mem else None
            mail_plan_out_q = Queue() if self.has_mem else None
            nfeat_plan_out_q = Queue() if self.has_nfeat else None
            self.planner = Planner(
                gpu_topo=gpu_topo,
                in_q=self.meta_bundle_q,
                send_q=self.send_q,
                mem_plan_out_q=mem_plan_out_q,
                mail_plan_out_q=mail_plan_out_q,
                nfeat_plan_out_q=nfeat_plan_out_q,
                stop_ev=self.stop_ev,
            )

            self.sem = threading.Semaphore()
            self.scatter = ScatterSender(
                sem=self.sem,
                n_gpus=env.n_gpu,
                comm=self.comm,
                send_q=self.send_q,
                mem_plan_out_q=mem_plan_out_q,
                mail_plan_out_q=mail_plan_out_q,
                nfeat_plan_out_q=nfeat_plan_out_q,
                stop_ev=self.stop_ev,
            )

        else:
            self.meta_bundle_q: Queue[List[Meta]] = Queue()
            self.meta_batcher = MetaBatcher(
                n_gpus=env.n_gpu,
                in_q=self.meta_in_q,
                out_q=self.send_q,
                stop_ev=self.stop_ev,
            )
            self.scatter = ScatterSender(
                sem=self.sem,
                n_gpus=env.n_gpu,
                comm=self.comm,
                send_q=self.send_q,
                mem_plan_out_q=None,
                mail_plan_out_q=None,
                nfeat_plan_out_q=None,
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
            print(f"Epoch {e}:")
            self._handle_a_split(self.tr_df, "train", cur_epoch=e)
            print("Validating...")
            self._handle_a_split(self.va_df, "val", cur_epoch=e)
            print("Testing...")
            self._handle_a_split(self.te_df, "test", cur_epoch=e)
        self._cleanup()
        # wandb.finish()

    @staticmethod
    def _plan_gpu_batches(num_samples, target_batch_size, ngpus=1):
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

    def _handle_a_split(
        self, df: pd.DataFrame, split: Literal["train", "val", "test"], cur_epoch: int
    ):
        df = df.reset_index(drop=True)

        iteration_total, adjusted_train_batch_size = self._plan_gpu_batches(
            len(df), self.batch_size, ngpus=self.n_gpu
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

    def _add_to_td_list(self, edges: pd.DataFrame, epoch_split_iter_info):
        src = torch.as_tensor(edges.src.values, dtype=torch.long)
        dst = torch.as_tensor(edges.dst.values, dtype=torch.long)
        edge_ts = torch.as_tensor(edges.time.values, dtype=torch.long)

        neg_dst = self.neg_sampler.sample(src, dst)

        pos_edge = torch.stack([src, dst], dim=1)
        pos_edge_id = torch.as_tensor(edges.index.values, dtype=torch.long)

        roots = torch.cat([src, dst, neg_dst])
        ts = torch.cat([edge_ts, edge_ts, edge_ts])
        ts_np = ts.numpy()

        # handle different model types
        batch_node_state_io = None
        gnn_mfgs_raw = None
        root2ngh_mfg = None
        if self.model_name == "JODIE":
            # for APAN and JODIE
            all_nodes = root_nodes = set(roots.numpy().tolist())
        elif self.model_name == "APAN":
            from gnn.with_memory.APAN import APAN

            root2ngh_mfg, batch_node_state_io = APAN.sample(
                self.ngh_sampler, src, dst, neg_dst, edge_ts
            )
        else:
            assert self.model_name in ["DyGFormer", "GAT", "TGN", "DySAT"]
            roots_np = roots.numpy()
            self.ngh_sampler.sample(roots_np, ts_np)
            ret = self.ngh_sampler.get_ret()
            gnn_mfgs_raw = extract_raw_data(ret)
            all_nodes, root_nodes = analyze_blocks(gnn_mfgs_raw, self.sample_history)
            root_nodes_for_verify = set(roots.numpy().tolist())
            assert root_nodes == root_nodes_for_verify, (
                "Sampled root nodes do not match the expected root nodes."
            )

        meta = Meta(
            gnn_mfgs_raw=gnn_mfgs_raw,
            all_nodes=all_nodes,
            root_nodes=root_nodes,
            pos_edge=pos_edge,
            pos_edge_ts=edge_ts,
            pos_edge_id=pos_edge_id,
            epoch_split_iter_info=epoch_split_iter_info,
            mdb_raw_data=root2ngh_mfg,
        )

        self.meta_in_q.put((meta, batch_node_state_io))

    def _cleanup(self):
        print("Host: Broadcasting CLEANUP command...")

        # 【新增】: 构造一个由 None 组成的列表，发送给所有 Worker
        # 这会解除 Worker 端 Receiver 线程在 scatter_object_list 上的阻塞
        stop_signal = [None for _ in range(self.n_gpu + 1)]

        # 注意：这里需要使用底层的 scatter_object_list，或者确保你的 comm.scatter_send 能处理 None
        # 假设 self.gloo 是你的通信组
        dist.scatter_object_list(
            [None], stop_signal, src=dist.get_rank(), group=self.gloo
        )

        print("Host: Stop signal sent. Entering final barrier...")
        dist.barrier()  # 建议指定 group=self.gloo 或者 nccl_pg，通常 gloo 用于控制流
        print("Host: All processes reached the final barrier.")
        end = time.time()

        print(f"Elapsed time: {end - self.start_time:.2f} seconds")
        dist.destroy_process_group()
