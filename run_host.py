import copy
import random
import time
from typing import List

import numpy as np
import pandas as pd
import torch.distributed as dist
from sampler_core import ParallelSampler
from tqdm import tqdm

from common.state import WorkerState
from gpu_distance import find_all_gpu_distances
from host.comm import Distributor
from host.scheduler import (
    batches_assign,
    plan_gpu_batches,
)
from memorys import MemoryMailbox
from sampler import NegLinkSampler, create_sampler
from schemas.datatypes import Config, DistEnv, TransferData
from utils import node_to_dgl_blocks, sampler_wrapper, to_dgl_blocks


class HostTrainer:
    def __init__(
        self,
        cfg: Config,
        g: dict,
        df: pd.DataFrame,
        mm: MemoryMailbox | None,
        env: DistEnv,
    ):
        """
        在构造函数中初始化所有组件和数据
        """
        self.cfg = cfg
        self.env = env
        self.mm = mm
        (self.scfg, _, _, self.tcfg) = cfg.s, cfg.m, cfg.g, cfg.t
        self.tcfg = copy.deepcopy(cfg.t)  # tcfg 会在 run_epoch 中被修改，所以拷贝一份

        # 1. 初始化通信器
        self.comm = Distributor()

        # 2. 拆分数据集
        train_edge_end = df[df["ext_roll"].gt(0)].index[0]
        val_edge_end = df[df["ext_roll"].gt(1)].index[0]
        self.tr_df = df[:train_edge_end]
        self.va_df = df[train_edge_end:val_edge_end]
        self.te_df = df[val_edge_end:]

        # 3. 初始化采样器
        self.ps = None
        if not self.scfg.no_sample:
            self.ps = ParallelSampler(
                g["indptr"],
                g["indices"],
                g["eid"],
                g["ts"].astype(np.float32),
                self.scfg.num_thread,
                1,
                self.scfg.layer,
                self.scfg.neighbor,
                self.scfg.strategy == "recent",
                self.scfg.prop_time,
                self.scfg.history,
                float(self.scfg.duration),
            )
        self.e_ns = create_sampler(g)
        self.t_ns = NegLinkSampler(g)

        # 4. 初始化状态变量
        self.best_ap = 0.0
        self.best_e = 0.0
        self.tap = 0.0
        self.tauc = 0.0
        self.m_id = 0  # 消息 ID 作为实例状态

    def train(self):
        """
        公共的训练入口方法
        """
        # wandb.init(
        #     project="my_tgl",
        #     config=self.cfg.dict(),
        # )

        for e in range(self.tcfg.epoch):
            self._run_epoch(e)

        self.tap, self.tauc = self._eval(self.te_df, "test")

        # 更新 wandb summary
        # wandb.summary["test_ap_at_best"] = self.tap
        # wandb.summary["test_auc_at_best"] = self.tauc

        # 训练结束，打印最终结果
        now_ts = time.time()
        local_time = time.localtime(now_ts)
        ms = int((now_ts - int(now_ts)) * 1000)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time) + f".{ms:03d}"
        print("{} Best model at epoch {}.".format(time_str, int(self.best_e)))
        print(
            "\ttest ap:{:4f}  test auc:{:4f}".format(float(self.tap), float(self.tauc))
        )

        # 5. 清理和退出
        self._cleanup()
        # wandb.finish()

    def _run_epoch(self, e: int):
        """
        私有的 epoch 运行逻辑，签名非常干净
        """
        now_ts = time.time()
        local_time = time.localtime(now_ts)
        ms = int((now_ts - int(now_ts)) * 1000)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time) + f".{ms:03d}"
        print(f"{time_str} Epoch {e}:")

        sample_tot = 0.0
        time_tot = 0.0

        if self.ps is not None:
            self.ps.reset()
        if self.mm is not None:
            self.mm.reset()

        # —— Train ——
        t_loss, t_s_time, t_time = self._train(self.tr_df)
        sample_tot += t_s_time
        time_tot += t_time

        # —— Val ——
        print("Validating...")
        # 注意：_eval 内部会使用和更新 self.m_id
        ap, auc = self._eval(self.va_df, "val")

        if ap > self.best_ap:
            print(
                f"New best model found at epoch {e} (val ap {ap:.4f}, auc {auc:.4f}), saving..."
            )
            self.best_e = e
            self.best_ap = ap

            # wandb.summary["best_val_ap"] = self.best_ap
            # wandb.summary["best_val_auc"] = auc
            # wandb.summary["best_epoch"] = self.best_e
            # self.comm.broadcast_state(WorkerState.SAVE_MODEL)
            # self.comm.scatter_send
            # 注意：_eval 内部会使用和更新 self.m_id
            # --- Test ---

        print(f"\ttrain loss:{t_loss:.4f} \tval ap:{ap:4f} \tval auc:{auc:4f}")
        print(f"\ttotal time:{time_tot:.2f}s sample time:{sample_tot:.2f}s")
        # wandb.log(
        #     {
        #         "epoch": e,
        #         "train_loss": t_loss,
        #         "val_ap": ap,
        #         "val_auc": auc,
        #     }
        # )

    def _train(self, df):
        # 训练批次规划 (注意这里用 self.tcfg)
        _, adjusted_train_batch_size = plan_gpu_batches(
            len(df), self.tcfg.batch_size, ngpus=self.env.ngpus
        )

        batches_assignment_0 = batches_assign(df.index, adjusted_train_batch_size)
        # reorder = self.tcfg.reorder if self.tcfg.reorder is not None else 1
        self.group_indexes = [batches_assignment_0]
        # if self.tcfg.reorder is not None:
        #     self.group_indexes.append(
        #         generate_reordered_indexes_1(
        #             reorder,
        #             adjusted_train_batch_size,
        #             len(df),
        #             batches_assignment_0,
        #         )
        #     )
        #     self.group_indexes.extend(
        #         generate_reordered_indexes_2(
        #             reorder,
        #             adjusted_train_batch_size,
        #             batches_assignment_0,
        #         )
        #     )

        sample_tot = time_tot = 0.0
        batch_to_send = []
        batches_assignment = self.group_indexes[
            random.randint(0, len(self.group_indexes) - 1)
        ]
        grouped = df.groupby(batches_assignment)
        for _, edges in tqdm(grouped, total=grouped.ngroups, desc="Training"):
            t0 = time.time()
            sample_tot += self._dispatch_train(edges, self.t_ns, batch_to_send)
            self.m_id += 1
            time_tot += time.time() - t0

        assert len(batch_to_send) == 0, "batch plan出错,不应该还有剩余的batch没有发送"

        print("Training time:", time_tot)
        # self.comm.broadcast_state(WorkerState.REDUCE_LOSS)
        # gathered_loss = self.comm.gather_receive()
        total_loss = 1.0
        # total_loss = float(np.sum(np.array(gathered_loss) * adjusted_train_batch_size))
        return total_loss, sample_tot, time_tot

    def sample_and_pack(self, edges, ns, nbatchs: List[TransferData]):
        roots = np.concatenate(
            [
                edges.src.values,
                edges.dst.values,
                ns.sample(edges.src.values, edges.time.values),
            ]
        ).astype(np.int32)
        ts = np.concatenate(
            [edges.time.values, edges.time.values, edges.time.values]
        ).astype(np.float32)

        if self.ps is not None:
            ret, s_time = sampler_wrapper(self.cfg.s.no_neg, roots, self.ps, ts)

        if self.cfg.g.arch != "identity":
            mfgss = to_dgl_blocks(ret, self.cfg.s.history, save_on_pgu=False)
        else:
            mfgss = node_to_dgl_blocks(roots, ts, cuda=False)

        tran_data = TransferData(
            mfgss=mfgss,
            root=roots,
            ts=ts,
            eid=edges.index.values,  # 如果你的 df 没这列，这里改成 edges.index.values
            block=None,
            mid=self.m_id,
            bid=self.m_id // self.env.ngpus,
        )

        if (self.mm is not None) and (self.cfg.m.deliver_to == "neighbors"):
            # 只把 history[1] 的反向 block 交给 mailbox
            tran_data.block = to_dgl_blocks(
                ret, self.cfg.s.history, reverse=True, save_on_pgu=False
            )[0][0]

        nbatchs.append(tran_data)

        nbatches_len = len(nbatchs)
        if len(nbatchs) >= self.env.ngpus:
            nbatchs[nbatches_len - self.env.ngpus].next_mfgss = mfgss
        return nbatchs, s_time

    def _plan_next(self, batchs):
        gpu_distances = find_all_gpu_distances
        ngpus = len(batchs) / 2
        gpus_groups = [[[0, 1], [2, 3]], [[5, 6], [7, 8]]]
        dim = gpus_groups.dim()
        list_nid_i_all_nodes_set = [] * ngpus
        list_nid_i_need_all_nodes_set = [] * ngpus
        for i, batch in iter(batchs[0:ngpus]):
            n_srcs = len(batch.roots) / 2
            batch.next_all_nodes = _all_node(batchs[ngpus + i].mfgss)
            what_is_need_for_next_big_batch = set(batch.next_all_node)
            i_all_nodes_set = set(batch.all_nodes)
            i_need_all_nodes_set = what_is_need_for_next_big_batch - i_all_nodes_set

        pass

    def _dispatch_train(self, edges, ns, batchs: List[TransferData]) -> float:
        batchs, s_time = self.sample_and_pack(edges, ns, batchs)

        if len(batchs) == self.env.ngpus * 2:
            batchs = _plan_next(batchs)
            batch_to_send = []
            for _ in range(self.env.ngpus):  # 删除3个元素
                batch_to_send.append(batchs.pop())  # 默认删除最后一个
            m_ids = [nbatch.mid for nbatch in batch_to_send]
            print(f"Host: Scattering batch {m_ids} (train) to all workers...")
            # self.comm.broadcast_state(WorkerState.TRAIN)
            # print(f"Host: Scatter batch {self.m_id} ...")
            self.comm.scatter_send(
                WorkerState.TRAIN, batch_to_send
            )  # <-- 真正的一次性 scatter

        return s_time

    @staticmethod
    def safe_mean(data_list: list) -> float:
        """计算列表的平均值，如果列表为空则返回 0.0"""
        valid_data = [x for x in data_list if x is not None and x != 0.0]
        if not valid_data:
            return 0.0
        return sum(valid_data) / len(valid_data)

    def _dispatch_eval(
        self,
        edges,
        ns,
        batchs: List[TransferData],
        split: str,
    ) -> tuple[float, float, float]:
        batchs, s_time = self.sample_and_pack(edges, ns, batchs)
        ap = auc = 0.0
        if len(batchs) == self.env.ngpus:
            m_ids = [nbatch.mid for nbatch in batchs]
            print(f"Host: Scattering batch {m_ids} (eval) to all workers...")
            ws = WorkerState.VAL if split != "val" else WorkerState.TEST
            # self.comm.broadcast_state(WorkerState.EVAL)
            # print(f"Host: Scatter batch {self.m_id} ...")
            self.comm.scatter_send(WorkerState.VAL, batchs)
            # ap_list = self.comm.gather_receive()
            # auc_list = self.comm.gather_receive()
            # ap = self.safe_mean(ap_list)
            # auc = self.safe_mean(auc_list)
            batchs.clear()

        return ap, auc, s_time

    def _eval(self, eval_df: pd.DataFrame, split: str) -> tuple[float, float]:
        """
        This is for valid and test split
        """
        _, adjusted_val_batch_size = plan_gpu_batches(
            len(eval_df), self.tcfg.batch_size, ngpus=self.env.ngpus
        )
        batches_assignment = batches_assign(eval_df.index, adjusted_val_batch_size)

        ap_tot, auc_tot = [], []
        batch_to_send = []
        grouped = eval_df.groupby(batches_assignment)
        for _, edges in tqdm(grouped, total=grouped.ngroups, desc="Evaluating..."):
            ap, auc, s_time = self._dispatch_eval(
                edges, self.e_ns, batch_to_send, split
            )
            self.m_id += 1
            if ap > 0 or auc > 0:
                ap_tot.append(ap)
                auc_tot.append(auc)

        assert len(batch_to_send) == 0, "batch plan出错,不应该还有剩余的batch没有发送"

        ap = self.safe_mean(ap_tot)
        auc = self.safe_mean(auc_tot)
        return ap, auc

    def _cleanup(self):
        """
        私有的清理逻辑
        """
        print("Host: Broadcasting CLEANUP command...")
        # self.comm.broadcast_state(WorkerState.CLEANUP)
        self.comm.scatter_send(WorkerState.CLEANUP)
        print("Host: Entering final barrier...")
        dist.barrier()
        print("Host: All processes reached the final barrier.")
        dist.destroy_process_group()
