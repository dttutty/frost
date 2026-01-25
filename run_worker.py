import logging
import math
import os
import queue
import threading
from queue import Queue as Q

import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP

from gnn.with_memory.base import MemoryBaseGNN
from gnn.without_memory.base import GNN
from schemas.alias import MetaWEvent
from schemas.datatypes.cfg import MemoryCFG, SampleCFG, TrainCFG
from schemas.datatypes.data_to_sync import BoxTsNid, MailTsNid, MemTsNid
from schemas.datatypes.env import Env
from schemas.datatypes.misc import (
    MemNBox,
    Meta,
    ModelComponents,
    Shared,
)
from schemas.datatypes.plan import TensorPlanSet
from schemas.datatypes.task import P2PTask, SyncTask
from worker.assembler.dy_asm import DyAssembler
from worker.data_syncer.data_syncer import DataSyncer
from worker.h2d import H2D
from worker.receiver import HostWorkerSyncer as Receiver


logger = logging.getLogger(__name__)


# def tensorfy(self, plan_set: ThreePlan):
#     for plan in [plan_set.mem_plan, plan_set.mail_plan, plan_set.feat_plan]:
#         if plan is not None:
#             plan.rec_cts = torch.tensor(plan.rec_cts, dtype=torch.long)
#             plan.send_cts = torch.tensor(plan.send_cts, dtype=torch.long)


class Worker:
    data_syncer: DataSyncer | None

    def __init__(
        self,
        s_cfg: SampleCFG,
        t_cfg: TrainCFG,
        shared: Shared | None,
        save_to: str,
        nfeat: None | torch.Tensor,
        efeat: None | torch.Tensor,
        env: Env,
        nccl: dist.ProcessGroup,
        gloo: dist.ProcessGroup,
        model_components: ModelComponents,
        m_cfg: MemoryCFG | None = None,
    ) -> None:
        self.s_cfg = s_cfg
        self.n_neg = s_cfg.neg_samples
        self.save_to = save_to
        self.env = env

        self.shared = shared
        self.total_epoch = t_cfg.epoch
        self.meta_q: Q[tuple[Meta, torch.cuda.Event] | None] = Q()
        self.stop_ev = threading.Event()
        self.nccl = nccl
        self.curr_epoch = -1
        self.logger: logging.Logger = logger
        self.fetch_factor = 1
        self.model: GNN | MemoryBaseGNN | DDP
        self.xfull_q: Q[MetaWEvent | None] = Q()

        self.h2d_sem: threading.BoundedSemaphore = threading.BoundedSemaphore(0)

        if self.shared is not None:
            assert m_cfg is not None
            self.delta_q: Q[tuple[MemNBox | None, torch.cuda.Event]] = Q()
            self.mem_p2p_q: Q[P2PTask] = Q()
            self.box_p2p_q: Q[P2PTask] = Q()
            self.xpart_q: Q[MetaWEvent | None] = Q()
            self.sync_q: Q[SyncTask] = Q()
            assert shared is not None, "Shared memory must be provided if memory exists"
            destination_q_for_h2d = self.xpart_q

            self.data_syncer = DataSyncer(
                mail_combine=m_cfg.mail_combine,
                merge_src=t_cfg.merge_src,
                n_gpu=env.n_gpu,
                shared=shared,
                h2d_sem=self.h2d_sem,
                sync_q=self.sync_q,
                stop_ev=self.stop_ev,
                mem_p2p_q=self.mem_p2p_q,
                box_p2p_q=self.box_p2p_q,
            )
            self.assembler = DyAssembler(
                in_q=self.xpart_q,
                delta_q=self.delta_q,
                out_q=self.xfull_q,
                stop_ev=self.stop_ev,
            )
            self.data_syncer.start()
            self.assembler.start()  # 启动 Assembler 组装线程
        else:
            destination_q_for_h2d = self.xfull_q
            self.data_syncer = None

        self.h2d = H2D(
            nfeat=nfeat,
            efeat=efeat,
            shared=shared,
            merge_src=t_cfg.merge_src,
            in_q=self.meta_q,
            out_q=destination_q_for_h2d,
            stop_ev=self.stop_ev,
            h2d_sem=self.h2d_sem,
        )

        self.result_q = Q()
        self.cmd_q = Q()

        self.receiver = Receiver(
            env=self.env,
            gloo=gloo,
            cmd_q=self.cmd_q,
            out_q=self.meta_q,
            result_q=self.result_q,
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

        if self.shared is not None:
            assert model_components.memory_updater is not None
            assert model_components.header is not None
            self.gnn = model_components.gnn
            self.model = MemoryBaseGNN(
                gnn=model_components.gnn,
                memory_updater=model_components.memory_updater,
                sync_q=self.sync_q,
                header=model_components.header,
            )
        else:
            assert model_components.header is not None
            self.gnn = model_components.gnn

            self.model = GNN(
                gnn=model_components.gnn,
                header=model_components.header,
            )

        self.model = self.model.cuda()
        find_unused_parameters = False
        if self.s_cfg.history > 1:
            find_unused_parameters = True

        find_unused_parameters = True
        self.model = DDP(
            self.model,
            process_group=self.nccl,
            find_unused_parameters=find_unused_parameters,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=t_cfg.lr)

        self.h2d.start()  # 启动 H2D 预取线程
        self.receiver.start()  # 启动 Receiver 接收线程

    def reset_ap_auc_count(self) -> None:
        self.count_sum = 0
        self.my_rank_preds: list[torch.Tensor] = []
        self.my_rank_labels: list[torch.Tensor] = []

        # [新增] 初始化 Loss 累积器 (必须在 GPU 上，避免 copy)
        self.epoch_loss_sum = torch.tensor(0.0, device="cuda")
        self.epoch_batch_count = torch.tensor(0.0, device="cuda")

        self.val_loss_sum = torch.tensor(0.0, device="cuda")
        self.val_batch_count = torch.tensor(0.0, device="cuda")

    def pad_tensor(
        self, local_tensor: torch.Tensor, max_size: int, fill: int = -1
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

    def train(self, meta: Meta, plan_set: TensorPlanSet | None) -> None:
        meta_info = meta.epoch_split_iter_info
        if meta_info.macro_position_of_split == "first":
            self.reset_ap_auc_count()
        n_pos_edges = meta.pos_edge.shape[0]
        labels = torch.cat(
            [
                torch.ones(n_pos_edges, device="cuda"),
                torch.zeros(n_pos_edges * self.n_neg, device="cuda"),
            ],
            dim=0,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.logger.debug(
            f"Rank {self.env.local_rank} training microbatch {meta_info.cur_macro_of_epoch}"
        )

        logits = self.model(meta)

        self.logger.debug(
            f"Rank {self.env.local_rank} finished forward for microbatch {meta_info.cur_macro_of_epoch}"
        )

        loss = self.criterion(logits.squeeze(-1), labels)
        if self.shared is not None:
            self.handle_p2p_q()

        dist.barrier(group=self.nccl)
        loss.backward()
        self.optimizer.step()

        # self.receive_next_meta_and_put_in_queue()
        self.epoch_loss_sum += loss.detach()
        self.epoch_batch_count += 1.0
        # reduced_loss = loss.detach().clone()
        # dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM, group=self.nccl)

        if meta_info.macro_position_of_split == "last":
            # 此时进行 all_reduce，将所有卡的 Loss 总和加起来
            dist.all_reduce(self.epoch_loss_sum, op=dist.ReduceOp.SUM, group=self.nccl)
            dist.all_reduce(
                self.epoch_batch_count, op=dist.ReduceOp.SUM, group=self.nccl
            )

            if self.env.local_rank == 1:  # 或者 rank 0
                # 只有在这里才调用 .item()，此时 Epoch 已经结束，不影响训练流水线
                avg_loss = (self.epoch_loss_sum / self.epoch_batch_count).item()
                self.logger.info(f"[Epoch End] Training Average Loss: {avg_loss:.6f}")

        # if self.env.l

    def _exec_p2p(self, p2p_task: P2PTask) -> MemTsNid | BoxTsNid | MailTsNid | None:
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
            output=recv_nid,
            input=send_nid,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=self.nccl,
        )
        dist.all_to_all_single(
            output=recv_data,
            input=send_x,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=self.nccl,
        )
        dist.all_to_all_single(
            output=recv_ts,
            input=send_ts,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=self.nccl,
        )

        if type == "mem":
            return MemTsNid(nid=recv_nid, data=recv_data, ts=recv_ts)
        elif type == "box":
            return BoxTsNid(nid=recv_nid, data=recv_data, ts=recv_ts)
        elif type == "mail":
            return MailTsNid(nid=recv_nid, data=recv_data, ts=recv_ts)

    @torch.no_grad()
    def handle_p2p_q(self) -> None:
        delta_mem = None
        delta_box = None

        # 1. 强制先处理 Mem 任务 (Deterministic Order: Mem -> Box)
        # 注意：如果某个 microbatch 不需要 mem p2p，DataSyncer 必须 put(None) 作为哨兵
        p2p_task = self.mem_p2p_q.get(block=True)
        if p2p_task is not None:
            delta_mem = self._exec_p2p(p2p_task)
            assert isinstance(delta_mem, MemTsNid)
            # 2. 强制后处理 Box 任务
            p2p_task = self.box_p2p_q.get(block=True)
            assert p2p_task is not None
            # Box 任务可能包含 mail -> box 的转换逻辑
            result = self._exec_p2p(p2p_task)
            if p2p_task.type == "mail":
                # 原有逻辑: mail_DyX.to_box()
                # 假设 _exec_p2p 返回的是构建好的对象
                assert isinstance(result, MailTsNid)
                delta_box = result.to_box()
            else:
                assert isinstance(result, BoxTsNid)
                delta_box = result

            delta_x = MemNBox(mem=delta_mem, box=delta_box)
        else:
            delta_x = None
        ready_event: torch.cuda.Event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())
        self.delta_q.put((delta_x, ready_event))

        self.logger.debug(
            f"DataSyncer: rank {self.env.local_rank} put DeltaX into DeltaX_q"
        )

    @torch.no_grad()
    def eval(self, meta: Meta, x: X, plan_set: TensorPlanSet | None) -> None:
        ap = 0.0
        auc = 0.0

        if meta.epoch_split_iter_info.macro_position_of_split == "first":
            self.reset_ap_auc_count()

        n_pos_edges = meta.pos_edge.shape[0]
        labels = torch.cat(
            [
                torch.ones(n_pos_edges, device="cuda"),
                torch.zeros(n_pos_edges, device="cuda"),
            ],
            dim=0,
        )
        self.model.eval()

        logits_pos, logits_neg = self.model(meta)
        if self.shared is not None:
            self.handle_p2p_q()
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        y_pred = logits.sigmoid()
        # if not (meta.epoch_split_iter_info.split == "test" and meta.epoch_split_iter_info.position_in_split == "last"):
        #     self.receive_next_meta_and_put_in_queue()
        self.logger.debug(
            f"Worker: rank {self.env.local_rank} received next meta for microbatch {meta.epoch_split_iter_info.cur_macro_of_split}, {meta.epoch_split_iter_info.split} {meta.epoch_split_iter_info.macro_position_of_split}"
        )

        if meta.epoch_split_iter_info.split == "val":
            loss = self.criterion(logits.squeeze(-1), labels)

            self.val_loss_sum += loss.detach()
            self.val_batch_count += 1.0

            # reduced_loss = loss.detach().clone()
            # print meta.split

            # dist.all_reduce(
            #     reduced_loss,
            #     op=dist.ReduceOp.SUM,
            #     group=self.nccl,
            # )
            # if self.env.local_rank == 0:
            #     # only rank 0 prints eval loss
            #     self.logger.info(
            #         f"[Microbatch {meta.macro_bid}, loss {reduced_loss.item()}"
            #     )
        else:
            # torch.cuda.synchronize()
            assert meta.epoch_split_iter_info.split == "test"
            self.logger.info(
                f"Rank {self.env.local_rank} finished forward for microbatch {meta.epoch_split_iter_info.cur_macro_of_split}"
            )

            if self.env.local_rank == 0:
                # only rank 0 prints eval loss
                self.logger.info(
                    f"[Microbatch {meta.epoch_split_iter_info.cur_macro_of_split}, test done]"
                )
        self.my_rank_preds.append(y_pred)
        self.my_rank_labels.append(labels)
        # print self.my_rank_preds and self.my_rank_labels sizes
        self.logger.debug(
            f"Rank {self.env.local_rank} preds size: {len(self.my_rank_preds)}, labels size: {len(self.my_rank_labels)}"
        )

        if meta.epoch_split_iter_info.macro_position_of_split == "last":
            # A. 统一结算 Loss (新增)
            if meta.epoch_split_iter_info.split == "val":
                dist.all_reduce(
                    self.val_loss_sum, op=dist.ReduceOp.SUM, group=self.nccl
                )
                dist.all_reduce(
                    self.val_batch_count, op=dist.ReduceOp.SUM, group=self.nccl
                )

                if self.env.local_rank == 0:  # 只有 Rank 0 打印
                    avg_val_loss = (self.val_loss_sum / self.val_batch_count).item()
                    self.logger.info(f"[Val End] Average Loss: {avg_val_loss:.6f}")

            my_rank_preds = torch.cat(self.my_rank_preds)
            my_rank_labels = torch.cat(self.my_rank_labels)

            # Only take the valid data length for this specific rank

            local_size = torch.tensor(
                [my_rank_preds.shape[0]], dtype=torch.long, device="cuda"
            )
            all_sizes = [torch.zeros_like(local_size) for _ in range(self.env.n_gpu)]
            self.logger.debug(f"Rank {self.env.local_rank} before gather all_sizes")

            dist.all_gather(all_sizes, local_size, group=self.nccl)

            # Convert size tensors to integers list
            size_list = [size.item() for size in all_sizes]
            max_size = int(max(size_list))
            pad_preds = self.pad_tensor(my_rank_preds, max_size, fill=-1)
            pad_labels = self.pad_tensor(my_rank_labels, max_size, fill=-1)

            preds_list = [torch.zeros_like(pad_preds) for _ in range(self.env.n_gpu)]
            labels_list = [torch.zeros_like(pad_labels) for _ in range(self.env.n_gpu)]
            self.logger.debug(f"Rank {self.env.local_rank} before gather preds_list")
            dist.all_gather(preds_list, pad_preds, group=self.nccl)
            self.logger.debug(f"Rank {self.env.local_rank} before gather labels_list")

            dist.all_gather(labels_list, pad_labels, group=self.nccl)
            self.logger.debug(
                f"Rank {self.env.local_rank} after gather all preds and labels"
            )
            if self.env.local_rank == 0:
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

    def run(self) -> None:
        while True:
            # 处理命令（非阻塞/短超时）
            try:
                cmd, payload = self.cmd_q.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd == "SAVE_MODEL":
                self.save_model(os.path)  # 这里在主线程执行

    def run_pipeline(self) -> None:
        self.logger.info("Worker: Starting H2D thread...")

        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        if self.shared is not None:
            self.delta_q.put(None)

        _initialize = False

        while True:
            logger.debug("Worker: waiting for x_full...")
            if _initialize is False:
                self.curr_epoch += 1
                # self.receive_next_meta_and_put_in_queue()
                _initialize = True
                # print meta split and mark
            if self.shared is not None:
                meta, plan_set, sorted_x, assemble_stream = self.xfull_q.get()
                logger.debug(
                    "Worker: got x_full", meta.epoch_split_iter_info.cur_macro_of_split
                )
                torch.cuda.current_stream().wait_stream(assemble_stream)
                logger.debug(
                    "Worker: waited for x_full",
                    meta.epoch_split_iter_info.cur_macro_of_split,
                )
            else:
                meta, sorted_x, plan_set, ready_event = self.xfull_q.get()
                torch.cuda.current_stream().wait_event(ready_event)

                logger.debug(
                    "Worker: got x_full", meta.epoch_split_iter_info.cur_macro_of_split
                )
            self.logger.info(
                f"Worker: rank {self.env.local_rank} processing microbatch {meta.epoch_split_iter_info.cur_macro_of_split}, {meta.epoch_split_iter_info.split} {meta.epoch_split_iter_info.macro_position_of_split}"
            )

            def check_sparse_table(
                name: str, ids: torch.Tensor | None, data: torch.Tensor | None
            ) -> None:
                pass
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

            # if __debug__:
            #     logger.debug("Worker: checking sparse tables", meta.epoch_split_iter_info.cur_macro_of_split)
            #     if sorted_x.mem:
            #         check_sparse_table("mem", sorted_x.mem.id, sorted_x.mem.data)
            #     if sorted_x.box:
            #         check_sparse_table("box", sorted_x.box.id, sorted_x.box.data)
            if plan_set is not None:
                plan_set = plan_set.cuda()
            logger.debug("plan_set moved to cuda")
            # torch.cuda.synchronize()

            if meta.epoch_split_iter_info.split == "train":  # if training
                self.train(meta, sorted_x, plan_set)
            elif meta.epoch_split_iter_info.split in ["val", "test"]:  # if evaluating
                if meta.epoch_split_iter_info.macro_position_of_split == "first":
                    self.reset_ap_auc_count()
                self.eval(meta, sorted_x, plan_set)
            # pint split and mark
            self.logger.debug(
                f"Worker: rank {self.env.local_rank} finished processing microbatch {meta.epoch_split_iter_info.cur_macro_of_split}, {meta.epoch_split_iter_info.split} {meta.epoch_split_iter_info.macro_position_of_split}"
            )

            if (
                meta.epoch_split_iter_info.split == "test"
                and meta.epoch_split_iter_info.macro_position_of_split == "last"
            ):  # last microbatch of an epoch
                self.logger.info(
                    f"Worker: Finished epoch {self.curr_epoch} test phase."
                )
                if self.shared:
                    self.shared.reset()
                # print current_epoch
                self.logger.info(f"Worker: Current epoch {self.curr_epoch} completed.")
                _initialize = False
                self.logger.debug("DyX set to None after epoch end")

            # print total epochs and current_epoch
            if (
                meta.epoch_split_iter_info.split == "test"
                and meta.epoch_split_iter_info.macro_position_of_split == "last"
            ):
                self.logger.info(f"total_epoch {self.total_epoch}")
                self.logger.info(
                    f"Worker: rank {self.env.local_rank} completed microbatch {meta.epoch_split_iter_info.cur_macro_of_split}. Current epoch: {self.curr_epoch}/{self.total_epoch}."
                )
            if (
                self.total_epoch == self.curr_epoch + 1
                and meta.epoch_split_iter_info.split == "test"
                and meta.epoch_split_iter_info.macro_position_of_split == "last"
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
    def save_model(self, path) -> None:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # 关键：保存“裸模型”的权重（非DDP也能加载）
            model = self.model.module if isinstance(self.model, DDP) else self.model
            torch.save(model.state_dict(), path)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def load_model(self, path) -> None:
        # 让所有rank都在文件写完后再开始读（更稳）
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        state = torch.load(path, map_location="cpu")

        # 如果当前是DDP，先拿到裸模型来load（因为你保存的是裸模型state_dict）
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(state, strict=True)

        # 如果每rank只可见1张GPU，这里用0没问题；否则用local_rank
        torch.cuda.set_device(0)
        model = model.to("cuda:0")

        # wrap 成 DDP
        self.model = DDP(model, device_ids=[0])

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def load_memory(self, memory_state) -> None:
        if self.shared is not None:
            self.model.module.load_memory_state(memory_state)
