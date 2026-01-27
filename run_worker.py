import logging
import os
import threading
from queue import Queue as Q
from typing import Literal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from gnn.with_memory.base import MemoryBaseGNN
from gnn.without_memory.base import GNN
from schemas.alias import MetaWEvent
from schemas.datatypes.cfg import MemoryCFG, SampleCFG, TrainCFG
from schemas.datatypes.data_to_sync import BoxTsNid, MemTsNid
from schemas.datatypes.env import Env
from schemas.datatypes.misc import MemNBox, Meta, ModelComponents, Result, Shared
from schemas.datatypes.task import P2PTask, SyncTask
from worker.assembler.dy_asm import DyAssembler
from worker.data_syncer.data_syncer import DataSyncer
from worker.h2d import H2D
from worker.receiver import Receiver as Receiver


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
        components: ModelComponents,
        m_cfg: MemoryCFG | None = None,
    ) -> None:
        self.s_cfg = s_cfg
        self.n_neg = s_cfg.neg_samples
        self.save_to = save_to
        self.env = env

        self.shared = shared
        self.total_epoch = t_cfg.epoch
        self.meta_q: Q[Meta | None | Literal["end_of_epoch"]] = Q()
        self.stop_ev = threading.Event()
        self.nccl = nccl
        self.curr_epoch = 0
        self.logger: logging.Logger = logger
        self.fetch_factor = 1
        self.model: GNN | MemoryBaseGNN | DDP
        self.full_q: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q()

        self.h2d_sem: threading.BoundedSemaphore = threading.BoundedSemaphore(0)
        self.destination_q_for_h2d: Q[MetaWEvent | None | Literal["end_of_epoch"]]

        if self.shared is not None:
            assert m_cfg is not None
            self.delta_q: Q[tuple[MemNBox | None, torch.cuda.Event]] = Q()
            self.mem_p2p_q: Q[P2PTask] = Q()
            self.box_p2p_q: Q[P2PTask] = Q()
            self.xpart_q: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q()
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
                out_q=self.full_q,
                stop_ev=self.stop_ev,
            )
            self.data_syncer.start()
            self.assembler.start()  # 启动 Assembler 组装线程
        else:
            destination_q_for_h2d = self.full_q
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

        self.result_q: Q[Result | None] = Q()

        self.receiver = Receiver(
            env=self.env,
            gloo=gloo,
            out_q=self.meta_q,
            result_q=self.result_q,
            stop_ev=self.stop_ev,
        )

        if self.shared is not None:
            assert components.memory_updater is not None
            assert components.header is not None
            self.gnn = components.gnn
            self.model = MemoryBaseGNN(
                gnn=components.gnn,
                memory_updater=components.memory_updater,
                sync_q=self.sync_q,
                header=components.header,
            )
        else:
            assert components.header is not None
            self.gnn = components.gnn

            self.model = GNN(
                gnn=components.gnn,
                header=components.header,
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
        self.train_loss_tracker = MeanMetric(process_group=self.nccl).cuda()

        # 2. 验证阶段：全都要
        # Loss 用于看过拟合，AUC/AP 用于选模型
        self.val_metrics = MetricCollection(
            {
                "val_loss": MeanMetric(
                    process_group=self.nccl
                ),  # 这里的 Loss 是单纯统计数值
                "val_auc": BinaryAUROC(
                    thresholds=10000, process_group=self.nccl
                ),  # 推荐开分桶加速
                "val_ap": BinaryAveragePrecision(
                    thresholds=10000, process_group=self.nccl
                ),
            }
        ).cuda()

        # 3. 测试阶段：只看指标
        self.test_metrics = MetricCollection(
            {
                "test_auc": BinaryAUROC(thresholds=10000, process_group=self.nccl),
                "test_ap": BinaryAveragePrecision(
                    thresholds=10000, process_group=self.nccl
                ),
            }
        ).cuda()
        self.h2d.start()  # 启动 H2D 预取线程
        self.receiver.start()  # 启动 Receiver 接收线程
        self.local_best_auc = 0.0

    def on_train_epoch_end(self) -> Result | None:
        # 3. 只有在 Epoch 结束时，才做一次全局同步，计算精准的平均 Loss 记入日志
        train_loss = self.train_loss_tracker.compute()

        val_results = self.val_metrics.compute()
        test_results = self.test_metrics.compute()
        if dist.get_rank() == 0:
            print(f"Epoch{self.curr_epoch} Done. ")
            print(f"Train Loss: {train_loss}, Val Loss {val_results['val_loss']}")
            print(f"Val AUC: {val_results['val_auc']}, Val AP: {val_results['val_ap']}")
            print(
                f"Test AUC: {test_results['test_auc']}, Test AP: {test_results['test_ap']}"
            )
        self.val_metrics.reset()
        self.test_metrics.reset()
        self.train_loss_tracker.reset()
        res = Result(
            train_loss=train_loss.item(),
            val_loss=val_results["val_loss"].item(),
            val_auc=val_results["val_auc"].item(),
            val_ap=val_results["val_ap"].item(),
            test_auc=test_results["test_auc"].item(),
            test_ap=test_results["test_ap"].item(),
        )
        if dist.get_rank() == 0:
            return res
        else:
            return None

    def train(self, meta: Meta) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(meta)
        loss = self.criterion(logits.squeeze(-1), meta.labels)
        if self.shared is not None:
            self.handle_p2p_q()
        dist.barrier(group=self.nccl)
        loss.backward()
        self.optimizer.step()
        self.train_loss_tracker.update(loss)

    def _exec_p2p(
        self, p2p_task: P2PTask, type: Literal["mem", "box"]
    ) -> MemTsNid | BoxTsNid:
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

    @torch.no_grad()
    def handle_p2p_q(self) -> None:
        delta_mem = None
        delta_box = None

        # 1. 强制先处理 Mem 任务 (Deterministic Order: Mem -> Box)
        # 注意：如果某个 microbatch 不需要 mem p2p，DataSyncer 必须 put(None) 作为哨兵
        mem_p2p_q = self.mem_p2p_q.get(block=True)
        if mem_p2p_q is None:
            box_p2p_q = self.box_p2p_q.get(block=True)
            assert box_p2p_q is None
            delta_x = None
            return
        if mem_p2p_q == "skip":
            box_p2p_q = self.box_p2p_q.get(block=True)
            assert box_p2p_q == "skip"
            self.delta_q.put("skip")
        else:
            delta_mem = self._exec_p2p(mem_p2p_q, "mem")
            box_p2p_q = self.box_p2p_q.get(block=True)
            assert box_p2p_q is not None
            delta_box = self._exec_p2p(box_p2p_q, "box")
            delta_x = MemNBox(mem=delta_mem, box=delta_box)
            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            self.delta_q.put((delta_x, ready_event))
            self.logger.debug(
                f"DataSyncer: rank {self.env.local_rank} put DeltaX into DeltaX_q"
            )

    @torch.no_grad()
    def eval(self, meta: Meta) -> None:
        assert meta.labels is not None
        self.model.eval()
        logits_pos, logits_neg = self.model(meta)
        if self.shared is not None:
            self.handle_p2p_q()
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        y_pred = logits.sigmoid()

        if meta.m_state.split == "val":
            loss = self.criterion(logits.squeeze(-1), meta.labels)

            self.val_metrics.update(preds=y_pred, target=meta.labels)
            self.val_metrics["val_loss"].update(loss)
        else:
            assert meta.m_state.split == "test"
            self.test_metrics.update(preds=y_pred, target=meta.labels)

    def run(self) -> None:
        self.logger.info("Worker: Starting H2D thread...")
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        if self.shared is not None:
            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            self.delta_q.put((None, ready_event))

        while True:
            full = self.full_q.get()
            if isinstance(full, tuple):
                meta, ready_event = full
                torch.cuda.current_stream().wait_stream(ready_event)
                m_info = meta.m_state
                if m_info.split == "train":  # if training
                    self.train(meta)
                elif m_info.split in ["val", "test"]:  # if evaluating
                    self.eval(meta)
            if full == "end_of_epoch":
                result = self.on_train_epoch_end()
                self.result_q.put(result)
                self.save_model(path="last.ckpt")
                val_auc = result.val_auc if result is not None else 0.0
                if val_auc > self.local_best_auc:
                    self.local_best_auc = val_auc
                    # 只有打破纪录才写磁盘
                    self.save_model(path="best.ckpt")
                self.curr_epoch += 1
                if self.shared:
                    self.shared.reset()
            if full is None:
                torch.cuda.synchronize()
                break

    def cleanup(self) -> None:
        self.h2d.join()
        self.receiver.join()
        if self.data_syncer is not None:
            self.data_syncer.join()
        self.assembler.join()
        dist.barrier()
        dist.destroy_process_group()

    # 供外部保存/加载
    def save_model(self, path: str) -> None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if rank == 0:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # 关键：保存“裸模型”的权重（非DDP也能加载）
            model = self.model.module if isinstance(self.model, DDP) else self.model
            torch.save(model.state_dict(), path)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def load_model(self, path: str) -> None:
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

    def load_memory(self, memory_state: dict) -> None:
        if self.shared is not None:
            self.model.module.load_memory_state(memory_state)
