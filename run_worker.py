import logging
import math
import os
import queue
import threading
from queue import Queue
from typing import Tuple, Union

import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP

from gnn.base import BaseModel
from schemas.datatypes import (
    BoxTsNid,
    Config,
    CudaPlanSet,
    Delta,
    Env,
    MailTsNid,
    MemTsNid,
    Meta,
    P2PTask,
    Shared,
    SyncTask,
    TensorPlanSet,
    X,
)
from worker.assembler import Assembler
from worker.data_syncer import DataSyncer
from worker.h2d import H2D
from worker.receiver import Receiver


# def tensorfy(self, plan_set: ThreePlan):
#     for plan in [plan_set.mem_plan, plan_set.mail_plan, plan_set.feat_plan]:
#         if plan is not None:
#             plan.rec_cts = torch.tensor(plan.rec_cts, dtype=torch.long)
#             plan.send_cts = torch.tensor(plan.send_cts, dtype=torch.long)


def _setup_logger(local_rank: int) -> logging.Logger:
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(f"worker.rank{local_rank}")
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs

    # Remove old handlers (important in some debug/reload cases)
    if logger.handlers:
        logger.handlers.clear()

    # Only rank0 prints to stdout by default
    if local_rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    h = logging.StreamHandler()
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [rank%(rank)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Inject rank into LogRecord
    class _RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = local_rank
            return True

    h.setFormatter(fmt)
    h.addFilter(_RankFilter())
    logger.addHandler(h)
    return logger


class Worker:
    def __init__(
        self,
        cfg: Config,
        shared: Shared,
        save_to: str,
        nfeat: None | torch.Tensor,
        efeat: None | torch.Tensor,
        env: Env,
        nccl_p2p,
        nccl_ddp,
        nccl_pg,
        gloo,
    ):
        self.cfg = cfg
        self.save_to = save_to
        self.env = env

        self.shared = shared
        self.total_epoch = int(self.shared.total_epoch[0].item())
        self.meta_q = queue.Queue(maxsize=1)
        self.xpart_q: Queue[Tuple[Meta, X, TensorPlanSet]] = queue.Queue(maxsize=1)
        self.delta_q: Queue[Delta | None] = queue.Queue(maxsize=1)

        self.nccl_p2p = nccl_p2p
        self.nccl_ddp = nccl_ddp
        self.mem_p2p_q: Queue[P2PTask] = queue.Queue()
        self.box_p2p_q: Queue[P2PTask] = queue.Queue()
        self.stop_ev = threading.Event()
        self.nfeat = nfeat
        self.efeat = efeat
        self.nccl_pg = nccl_pg
        self.gloo = gloo
        self.curr_epoch = -1
        self.logger = _setup_logger(self.env.local_rank)
        self.h2d_sem = threading.Semaphore(0)
        self.h2d = H2D(
            nfeat=nfeat,
            efeat=efeat,
            shared=shared,
            merge_src=self.cfg.t.merge_src,
            meta_q=self.meta_q,
            xpart_q=self.xpart_q,
            stop_ev=self.stop_ev,
            h2d_sem=self.h2d_sem,
        )

        self.xfull_q = queue.Queue(maxsize=1)
        self.sync_q: Queue[SyncTask] = queue.Queue(maxsize=1)

        self.data_syncer = DataSyncer(
            cfg=cfg,
            env=self.env,
            shared=shared,
            h2d_sem=self.h2d_sem,
            sync_q=self.sync_q,
            stop_ev=self.stop_ev,
            mem_p2p_q=self.mem_p2p_q,
            box_p2p_q=self.box_p2p_q,
        )
        model = BaseModel(cfg, self.sync_q).cuda()

        find_unused_parameters = False
        if cfg.s and cfg.s.history > 1:
            find_unused_parameters = True

        find_unused_parameters = True
        self.model: Union[DDP, BaseModel] = DDP(
            model,
            process_group=self.nccl_ddp,
            find_unused_parameters=find_unused_parameters,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.t.lr)
        self.assembler = Assembler(
            xpart_q=self.xpart_q,
            delta_q=self.delta_q,
            xfull_q=self.xfull_q,
            stop_ev=self.stop_ev,
        )

        # self.receiver_thread = threading.Thread(
        #     target=self._receiver_loop,
        #     name="MetaReceiverThread",
        #     daemon=True,  # 设置为守护线程，主程序退出时它会自动结束
        # )

        self.receiver = Receiver(
            env=self.env,
            gloo=self.gloo,
            meta_q=self.meta_q,
            stop_ev=self.stop_ev,
        )

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

    def reset_ap_auc_count(self):
        self.count_sum = 0
        self.my_rank_preds = []
        self.my_rank_labels = []

    def pad_tensor(
        self, local_tensor: torch.Tensor, max_size: int, fill=-1
    ) -> torch.Tensor:
        if local_tensor.shape[0] < max_size:
            pad_len = max_size - local_tensor.shape[0]
            # Create padding shape: (pad_len, dim1, dim2, ...)
            pad_shape = list(local_tensor.shape)
            pad_shape[0] = pad_len
            padding = torch.full(
                pad_shape, fill, dtype=local_tensor.dtype, device="cuda"
            )
            padded_local = torch.cat([local_tensor, padding], dim=0)
        else:
            padded_local = local_tensor
        return padded_local

    def train(self, meta: Meta, x: X, plan_set: CudaPlanSet):
        if meta.mark == "first":
            self.reset_ap_auc_count()
        n_pos_edges = meta.pos_edge.shape[0]
        labels = torch.cat(
            [
                torch.ones(n_pos_edges, device="cuda"),
                torch.zeros(n_pos_edges, device="cuda"),
            ],
            dim=0,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.logger.debug(
            f"Rank {self.env.local_rank} training microbatch {meta.macro_bid}"
        )

        logits_pos, logits_neg = self.model(meta, x, plan_set, is_last=False)
        self.logger.debug(
            f"Rank {self.env.local_rank} finished forward for microbatch {meta.macro_bid}"
        )

        logits = torch.cat([logits_pos, logits_neg])

        loss = self.criterion(logits.squeeze(-1), labels)

        dist.barrier(group=self.nccl_pg)
        self.handle_p2p_q()
        loss.backward()
        self.optimizer.step()
        # self.receive_next_meta_and_put_in_queue()
        reduced_loss = loss.detach().clone()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM, group=self.nccl_pg)

        if self.env.local_rank == 1:
            self.logger.info(
                f"[Microbatch {meta.macro_bid}, loss {reduced_loss.item()}"
            )

    # def receive_next_meta_and_put_in_queue(self):
    #     next_meta_buf = [None]
    #     dist.scatter_object_list(
    #         next_meta_buf, None, src=self.env.host_rank, group=self.gloo
    #     )
    #     next_meta: Meta = next_meta_buf[0]
    #     assert next_meta.plan_set is not None, (
    #         "plan_set should not be None before putting in queue"
    #     )
    #     self.meta_q.put(copy.deepcopy(next_meta), block=True)

    # def _receiver_loop(self):
    #     """这是在该线程中一直跑的循环"""
    #     while True:
    #         try:
    #             next_meta_buf = [None]

    #             # 这一步是阻塞的，直到 host 发来数据
    #             dist.scatter_object_list(
    #                 next_meta_buf, None, src=self.env.host_rank, group=self.gloo
    #             )

    #             next_meta = next_meta_buf[0]

    #             # 【重要】需要一个退出机制
    #             # 约定：如果 host 发来 None 或者特殊的结束信号，就退出循环
    #             if next_meta is None:
    #                 print(
    #                     f"Rank {self.env.local_rank}: Received stop signal. Exiting receiver thread."
    #                 )
    #                 break

    #             assert next_meta.plan_set is not None, (
    #                 "plan_set should not be None before putting in queue"
    #             )

    #             # 放入队列供主线程消费
    #             self.meta_q.put(copy.deepcopy(next_meta), block=True)

    #         except Exception as e:
    #             print(f"Rank {self.env.local_rank}: Error in receiver thread: {e}")
    #             break

    def _exec_p2p(self, p2p_task):
        nid, data, ts, type = p2p_task.nid, p2p_task.data, p2p_task.ts, p2p_task.type
        recv_nid, send_nid, out_splits, in_splits = (
            nid.recv,
            nid.send,
            nid.out_splits,
            nid.in_splits,
        )
        recv_data, send_x = data.recv, data.send
        recv_ts, send_ts = ts.recv, ts.send

        dist.all_to_all_single(
            recv_nid, send_nid, out_splits, in_splits, group=self.nccl_p2p
        )
        dist.all_to_all_single(
            recv_data, send_x, out_splits, in_splits, group=self.nccl_p2p
        )
        dist.all_to_all_single(
            recv_ts, send_ts, out_splits, in_splits, group=self.nccl_p2p
        )

        if type == "mem":
            return MemTsNid(id=recv_nid, data=recv_data, ts=recv_ts)
        elif type == "box":
            return BoxTsNid(id=recv_nid, data=recv_data, ts=recv_ts)
        elif type == "mail":
            return MailTsNid(id=recv_nid, data=recv_data, ts=recv_ts)

    @torch.no_grad()
    def handle_p2p_q(self):
        mem_delta = None
        box_delta = None

        # 1. 强制先处理 Mem 任务 (Deterministic Order: Mem -> Box)
        # 注意：如果某个 microbatch 不需要 mem p2p，DataSyncer 必须 put(None) 作为哨兵
        p2p_task = self.mem_p2p_q.get(block=True)
        if p2p_task is not None:
            mem_delta = self._exec_p2p(p2p_task)
        else:
            self.delta_q.put(None)

        # 2. 强制后处理 Box 任务
        p2p_task = self.box_p2p_q.get(block=True)
        if p2p_task is not None:
            # Box 任务可能包含 mail -> box 的转换逻辑
            result = self._exec_p2p(p2p_task)
            if p2p_task.type == "mail":
                # 原有逻辑: mail_delta.to_box()
                # 假设 _exec_p2p 返回的是构建好的对象
                box_delta = result.to_box()
            else:
                box_delta = result

        delta = Delta(mem=mem_delta, box=box_delta)
        self.delta_q.put(delta)
        print(f"DataSyncer: rank {self.env.local_rank} put delta into delta_q")

    @torch.no_grad()
    def eval(self, meta: Meta, x: X, plan_set: CudaPlanSet):
        ap = 0.0
        auc = 0.0
        n_pos_edges = meta.pos_edge.shape[0]
        labels = torch.cat(
            [
                torch.ones(n_pos_edges, device="cuda"),
                torch.zeros(n_pos_edges, device="cuda"),
            ],
            dim=0,
        )
        self.model.eval()
        is_last_macrobatch_of_epoch = meta.mark == "last" and meta.split == "test"
        logits_pos, logits_neg = self.model(
            meta, x, plan_set, is_last=is_last_macrobatch_of_epoch
        )

        self.handle_p2p_q()
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        y_pred = logits.sigmoid()
        # if not (meta.split == "test" and meta.mark == "last"):
        #     self.receive_next_meta_and_put_in_queue()
        self.logger.debug(
            f"Worker: rank {self.env.local_rank} received next meta for microbatch {meta.macro_bid}, {meta.split} {meta.mark}"
        )

        self.logger.debug(
            f"Rank {self.env.local_rank} eval val microbatch {meta.split} {meta.mini_bid}"
        )
        if meta.split == "val":
            loss = self.criterion(logits.squeeze(-1), labels)
            reduced_loss = loss.detach().clone()
            # print meta.split

            work = dist.all_reduce(
                reduced_loss,
                op=dist.ReduceOp.SUM,
                group=self.nccl_pg,
            )
            if self.env.local_rank == 0:
                # only rank 0 prints eval loss
                self.logger.info(
                    f"[Microbatch {meta.macro_bid}, loss {reduced_loss.item()}"
                )
        else:
            # torch.cuda.synchronize()
            assert meta.split == "test"
            self.logger.info(
                f"Rank {self.env.local_rank} finished forward for microbatch {meta.macro_bid}"
            )

            if self.env.local_rank == 0:
                # only rank 0 prints eval loss
                self.logger.info(f"[Microbatch {meta.macro_bid}, test done]")
        self.my_rank_preds.append(y_pred)
        self.my_rank_labels.append(labels)
        # print self.my_rank_preds and self.my_rank_labels sizes
        self.logger.debug(
            f"Rank {self.env.local_rank} preds size: {len(self.my_rank_preds)}, labels size: {len(self.my_rank_labels)}"
        )

        if meta.mark == "last":
            my_rank_preds = torch.cat(self.my_rank_preds)
            my_rank_labels = torch.cat(self.my_rank_labels)

            # Only take the valid data length for this specific rank

            local_size = torch.tensor(
                [my_rank_preds.shape[0]], dtype=torch.long, device="cuda"
            )
            all_sizes = [torch.zeros_like(local_size) for _ in range(self.env.n_gpu)]
            self.logger.debug(f"Rank {self.env.local_rank} before gather all_sizes")

            work = dist.all_gather(all_sizes, local_size, group=self.nccl_pg)

            # Convert size tensors to integers list
            size_list = [size.item() for size in all_sizes]
            max_size = int(max(size_list))
            pad_preds = self.pad_tensor(my_rank_preds, max_size, fill=-1)
            pad_labels = self.pad_tensor(my_rank_labels, max_size, fill=-1)

            preds_list = [torch.zeros_like(pad_preds) for _ in range(self.env.n_gpu)]
            labels_list = [torch.zeros_like(pad_labels) for _ in range(self.env.n_gpu)]
            self.logger.debug(f"Rank {self.env.local_rank} before gather preds_list")
            work = dist.all_gather(preds_list, pad_preds, group=self.nccl_pg)
            self.logger.debug(f"Rank {self.env.local_rank} before gather labels_list")

            work = dist.all_gather(labels_list, pad_labels, group=self.nccl_pg)
            self.logger.debug(
                f"Rank {self.env.local_rank} after gather all preds and labels"
            )
            if self.env.local_rank == 1:
                all_rank_preds = []
                all_rank_labels = []
                for i in range(self.env.n_gpu):
                    real_length = size_list[i]
                    all_rank_preds.append(preds_list[i][:real_length])
                    all_rank_labels.append(labels_list[i][:real_length])
                all_labels = torch.cat(all_rank_labels).cpu().numpy()
                all_preds = torch.cat(all_rank_preds).cpu().numpy()
                ap = average_precision_score(all_labels, all_preds)
                auc = roc_auc_score(all_labels, all_preds)
                self.logger.info(
                    f"Evaluation results at epoch end - AP: {ap:.4f}, AUC: {auc:.4f}"
                )

        return ap, auc

    def run(self):
        self.logger.info("Worker: Starting H2D thread...")
        self.receiver.start()
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作

        self.h2d.start()  # 启动 H2D 预取线程

        self.assembler.start()  # 启动 Assembler 组装线程
        self.data_syncer.start()
        self.delta_q.put(None)

        _initialize = False

        while True:
            print("Worker: waiting for x_full...")
            if _initialize is False:
                self.curr_epoch += 1
                # self.receive_next_meta_and_put_in_queue()
                _initialize = True
                # print meta split and mark

            meta, plan_set, sorted_x, assemble_stream = self.xfull_q.get()
            print("Worker: got x_full", meta.macro_bid)
            torch.cuda.current_stream().wait_stream(assemble_stream)
            print("Worker: waited for x_full", meta.macro_bid)
            self.logger.info(
                f"Worker: rank {self.env.local_rank} processing microbatch {meta.macro_bid}, {meta.split} {meta.mark}"
            )

            def check_sparse_table(name, ids, data):
                if ids is None or data is None:
                    return
                assert ids.numel() == data.size(0), (
                    f"{name}: ids rows {ids.numel()} != data rows {data.size(0)}"
                )
                mn = int(ids.min().item())
                assert mn >= 0, f"{name}: negative global id min={mn}"

                # 如果你要求 ids 排序（方便 searchsorted）
                assert torch.all(ids[1:] >= ids[:-1]), f"{name}: ids not sorted"

                # 如果你要求 unique（可选）
                has_dup = torch.any(ids[1:] == ids[:-1])
                assert not has_dup, f"{name}: duplicated ids found"

            if __debug__:
                print("Worker: checking sparse tables", meta.macro_bid)
                if sorted_x.nfeat:
                    check_sparse_table("nfeat", sorted_x.nfeat.id, sorted_x.nfeat.data)
                if sorted_x.mem:
                    check_sparse_table("mem", sorted_x.mem.id, sorted_x.mem.data)
                if sorted_x.box:
                    check_sparse_table("box", sorted_x.box.id, sorted_x.box.data)
                if sorted_x.efeat:
                    check_sparse_table("efeat", sorted_x.efeat.id, sorted_x.efeat.data)

            plan_set_cuda = plan_set.cuda()
            print("plan_set moved to cuda")
            # torch.cuda.synchronize()

            if meta.split == "train":  # if training
                self.train(meta, sorted_x, plan_set_cuda)
            elif meta.split in ["val", "test"]:  # if evaluating
                if meta.mark == "first":
                    self.reset_ap_auc_count()
                self.eval(meta, sorted_x, plan_set_cuda)
            # pint split and mark
            self.logger.debug(
                f"Worker: rank {self.env.local_rank} finished processing microbatch {meta.macro_bid}, {meta.split} {meta.mark}"
            )

            if (
                meta.split == "test" and meta.mark == "last"
            ):  # last microbatch of an epoch
                self.logger.info(
                    f"Worker: Finished epoch {self.curr_epoch} test phase."
                )
                self.shared.reset()
                # print current_epoch
                self.logger.info(f"Worker: Current epoch {self.curr_epoch} completed.")
                _initialize = False
                self.logger.debug("delta set to None after epoch end")

            # print total epochs and current_epoch
            if meta.split == "test" and meta.mark == "last":
                self.logger.info(f"total_epoch {self.total_epoch}")
                self.logger.info(
                    f"Worker: rank {self.env.local_rank} completed microbatch {meta.macro_bid}. Current epoch: {self.curr_epoch}."
                )
            if (
                self.total_epoch == self.curr_epoch + 1
                and meta.split == "test"
                and meta.mark == "last"
            ):  # last microbatch of the final epoch
                torch.cuda.synchronize()
                self.logger.info(
                    "Worker: Training finished. Waiting for receiver to shutdown..."
                )
                self.stop_ev.set()

                # 【新增】等待 Receiver 线程优雅结束
                # 因为 Host 发送了 None，Receiver 里的循环会 break，线程会结束
                if self.receiver.is_alive():
                    self.receiver.join(timeout=30)  # 等待最多30秒
                    if self.receiver.is_alive():
                        self.logger.warning(
                            "Worker: Receiver thread did not exit cleanly!"
                        )

                self.logger.info("Worker: Receiver shutdown. Entering final barrier...")
                dist.barrier()  # 这里的 group 要和 Host 的 barrier group 一致
                break

    # 供外部保存/加载
    def save(self, path):
        if self.env.local_rank == 0:
            torch.save(self.model.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=torch.device("cuda:0"))
        self.model.load_state_dict(state)
