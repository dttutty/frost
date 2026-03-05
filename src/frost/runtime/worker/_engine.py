import logging
import threading
import time
from queue import Queue as Q
from typing import (
    Literal,
    cast,  # 必须导入这个
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from frost.aliases.syncer import (
    BoxP2PQ,
    BoxTaskQ,
    BoxWrQ,
    MemP2PQ,
    MemTaskQ,
    MemWrQ,
)
from frost.aliases.worker import DeltaQ, FullQ, H2DOutQ, ReceivedQ
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.models._assembled_model import AssembledModel
from frost.protocols.mem_updater_proto import MemoryUpdaterProto
from frost.protocols.postman_proto import PostmanProto
from frost.runtime.worker._dy_asm import DyAssembler
from frost.runtime.worker._h2d import H2D
from frost.runtime.worker._receiver import Receiver as Receiver
from frost.runtime.worker.p2p_syncer._box_handler import BoxTaskHandler
from frost.runtime.worker.p2p_syncer._box_wr import BoxWriter
from frost.runtime.worker.p2p_syncer._mem_handler import MemTaskHandler
from frost.runtime.worker.p2p_syncer._mem_wr import MemWriter
from frost.runtime.worker.p2p_syncer._p2p_q_handler import P2PQHandler
from frost.shm.mem_mgr import MemBoxStore
from frost.shm.ring_buf import RingBuf

logger = logging.getLogger(__name__)


class Worker:
    received_q: ReceivedQ
    delta_q: DeltaQ | None = None
    h2d_out_q: H2DOutQ | None = None
    full_q: FullQ

    mem_task_q: MemTaskQ | None = None
    box_task_q: BoxTaskQ | None = None
    mem_p2p_q: MemP2PQ | None = None
    box_p2p_q: BoxP2PQ | None = None
    mem_wr_q: MemWrQ | None = None
    box_wr_q: BoxWrQ | None = None

    model: AssembledModel | DDP
    result_q: Q[Result]

    def __init__(
        self,
        n_neg: int,
        mem_box_store: MemBoxStore | None,
        save_to: str,
        nfeat: None | torch.Tensor,
        efeat: None | torch.Tensor,
        env: Env,
        nccl_ddp: dist.ProcessGroup,
        nccl_p2p: dist.ProcessGroup,
        mem_updater: MemoryUpdaterProto | None,
        encoder: nn.Module,
        predictor: nn.Module,
        ring_buf: RingBuf,
        postman: PostmanProto | None,
        criterion: torch.nn.Module,
        optimizer: Literal["adam", "sgd"],
        lr: float,
        mail_combine: Literal["last", "mean"] | None,
    ) -> None:
        self.ring_buf = ring_buf
        self.n_neg = n_neg
        self.save_to = save_to
        self.env = env
        self.postman = postman
        self.mem_updater = mem_updater
        self.mem_box_store = mem_box_store
        self.stop_ev = threading.Event()
        self.nccl = nccl_ddp
        self.nccl_p2p = nccl_p2p
        self.curr_epoch = 0
        self.logger: logging.Logger = logger
        self.fetch_factor = 1
        self.full_q = Q(maxsize=2 if mem_box_store is None else 1)
        self.device = torch.device(f"cuda:{env.local_rank}")
        self.comm = cast(torch.cuda.Stream, torch.cuda.Stream(device=self.device))
        self.h2d_sem: threading.Semaphore = threading.Semaphore(0)
        self.mail_combine = mail_combine
        self.lr = lr

        self.train_start2start_ms = MeanMetric(process_group=self.nccl).cuda()
        self._last_train_start_t: float | None = None

        self.p2p_sem = threading.Semaphore(0)
        self.result_q = Q(maxsize=1)

        self.received_q = Q(maxsize=2 if mem_box_store is None else 1)
        self.receiver = Receiver(
            env=self.env,
            received_q=self.received_q,
            result_q=self.result_q,
            stop_ev=self.stop_ev,
            ring_buf=ring_buf,
        )

        if self.mem_box_store is not None:
            assert mem_box_store is not None
            self.delta_q = Q(maxsize=1)

            self.h2d_out_q = Q(maxsize=1)
            destination_q_for_h2d = self.h2d_out_q

            self.dy_asm = DyAssembler(
                mem_box_store=self.mem_box_store,
                h2d_out_q=self.h2d_out_q,
                delta_q=self.delta_q,
                full_q=self.full_q,
                stop_ev=self.stop_ev,
                device=self.device,
                h2d_sem=self.h2d_sem,
            )
            self.init_mem_box_threads()

        else:
            destination_q_for_h2d = self.full_q
            self.mem_task_q = None
            self.box_task_q = None

        self.h2d = H2D(
            ring_buf=ring_buf,
            nfeat=nfeat,
            efeat=efeat,
            device=self.device,
            n_neg=self.n_neg,
            in_q=self.received_q,
            out_q=destination_q_for_h2d,
            stop_ev=self.stop_ev,
        )

        self.model = AssembledModel(
            encoder=encoder,
            mem_updater=mem_updater,
            mem_task_q=self.mem_task_q,
            box_task_q=self.box_task_q,
            header=predictor,
            n_neg=self.n_neg,
            postman=self.postman,
            sample_4_neg=ring_buf.sample_4_neg,
        )

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total, trainable

        # total, trainable = count_parameters(self.model)

        self.model = self.model.cuda()
        find_unused_parameters = False

        self.model = DDP(
            self.model,
            process_group=self.nccl,
            find_unused_parameters=find_unused_parameters,
        )
        self.criterion = criterion
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
        self.local_best_auc = 0.0

    def init_mem_box_threads(self) -> None:
        assert self.mem_updater is not None
        assert self.mem_box_store is not None
        assert self.delta_q is not None
        self.mem_task_q = Q(maxsize=1)
        self.box_task_q = Q(maxsize=1)
        self.mem_p2p_q = Q(maxsize=1)
        self.box_p2p_q = Q(maxsize=1)
        self.mem_wr_q = Q(maxsize=1)
        self.mem_handler = MemTaskHandler(
            h2d_sem=self.h2d_sem,
            mem_task_q=self.mem_task_q,
            mem_p2p_q=self.mem_p2p_q,
            stop_ev=self.stop_ev,
            device=self.device,
            mem_wr_q=self.mem_wr_q,
        )
        self.mem_wr = MemWriter(
            shared_mem=self.mem_box_store.mem,
            shared_mem_ts=self.mem_box_store.mem_ts,
            h2d_sem=self.h2d_sem,
            stop_ev=self.stop_ev,
            mem_wr_q=self.mem_wr_q,
            device=self.device,
        )

        self.box_wr_q = Q(maxsize=1)
        assert self.mail_combine is not None
        self.box_handler = BoxTaskHandler(
            h2d_sem=self.h2d_sem,
            box_task_q=self.box_task_q,
            box_p2p_q=self.box_p2p_q,
            stop_ev=self.stop_ev,
            mail_combine=self.mail_combine,
            device=self.device,
            box_wr_q=self.box_wr_q,
            postman=self.postman,
        )
        self.box_wr = BoxWriter(
            shared_box=self.mem_box_store.box,
            shared_box_ts=self.mem_box_store.box_ts,
            h2d_sem=self.h2d_sem,
            stop_ev=self.stop_ev,
            box_wr_q=self.box_wr_q,
            device=self.device,
        )

        self.p2p_q_handler = P2PQHandler(
            env=self.env,
            mem_p2p_q=self.mem_p2p_q,
            box_p2p_q=self.box_p2p_q,
            nccl_p2p=self.nccl_p2p,
            p2p_sem=self.p2p_sem,
            stop_ev=self.stop_ev,
            delta_q=self.delta_q,
            device=self.device,
        )

    def on_train_epoch_end(self) -> Result:
        avg_s2s = self.train_start2start_ms.compute().item()
        self.train_start2start_ms.reset()
        logger.debug("Average train start->start (ms): %.2f", avg_s2s)
        # 3. 只有在 Epoch 结束时，才做一次全局同步，计算精准的平均 Loss 记入日志

        avg_loss = self.train_loss_tracker.compute()
        val_results = self.val_metrics.compute()
        test_results = self.test_metrics.compute()

        self.val_metrics.reset()
        self.test_metrics.reset()
        self.train_loss_tracker.reset()

        res = Result(
            train_loss=avg_loss.item(),
            val_loss=val_results["val_loss"].item(),
            val_auc=val_results["val_auc"].item(),
            val_ap=val_results["val_ap"].item(),
            test_auc=test_results["test_auc"].item(),
            test_ap=test_results["test_ap"].item(),
        )
        self._last_train_start_t = None
        return res

    def train(self, meta: Meta) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(meta)
        loss = self.criterion(logits.squeeze(-1), meta.labels)

        loss.backward()

        self.optimizer.step()
        if self.mem_box_store is not None:
            self.p2p_sem.acquire()  # 获取信号量，确保当前没有其他 P2P 任务在进行
        self.train_loss_tracker.update(loss.detach())
        self._release_meta_tensors(meta)

    @torch.no_grad()
    def eval(self, meta: Meta) -> None:
        assert meta.labels is not None
        self.model.eval()

        logits = self.model(meta)
        if self.mem_box_store is not None:
            self.p2p_sem.acquire()  # 获取信号量，确保当前没有其他 P2P 任务在进行

        y_pred = logits.sigmoid().squeeze(-1)

        if meta.split == "val":
            loss = self.criterion(logits.squeeze(-1), meta.labels)

            # [Fix] 显式指定哪个参数给哪个指标，或者分开更新
            # MeanMetric 需要 'value'，分类指标需要 'preds' 和 'target'
            self.val_metrics.update(
                preds=y_pred,
                target=meta.labels.long(),
                value=loss,  # 这一行会让 val_loss 找到它需要的参数
            )

            # === [新增] 收集原始数据 (转到 CPU 以节省显存) ===

        else:
            assert meta.split == "test"
            self.test_metrics.update(preds=y_pred, target=meta.labels.long())

        self._release_meta_tensors(meta)

    def _release_meta_tensors(self, meta: Meta) -> None:
        """Drop per-batch GPU tensors ASAP to reduce delayed CUDA memory recycling."""

        meta.flat_graph = None
        meta.plan_set = None
        meta.labels = None
        meta.graph = None
        meta = None

    def start_all_threads(self) -> None:
        self.h2d.start()
        self.receiver.start()
        if self.mem_box_store is not None:
            self.p2p_q_handler.start()
            self.mem_handler.start()
            self.box_handler.start()
            self.box_wr.start()
            self.mem_wr.start()
            self.dy_asm.start()

    def run(self) -> None:
        self.start_all_threads()
        if self.mem_box_store is not None:
            self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
            self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
            self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
            self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
            assert self.delta_q is not None
            self.delta_q.put("nothing")

        def trace_handler(p):
            rank = dist.get_rank() if dist.is_initialized() else 0
            output_filename = f"trace_epoch2_rank_{rank}.json"
            p.export_chrome_trace(output_filename)
            logger.debug("Rank %s: Trace saved to %s", rank, output_filename)

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=None,  # 关键：设为 None，手动控制
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
            with_flops=False,
        )

        # 用于记录第二个 epoch 抓了几个 step
        epoch2_steps_recorded = 0
        # 标记是否已经保存过，防止重复 start/stop
        profiling_done = False

        total_wait_ms = 0.0
        wait_steps = 0
        while self.stop_ev.is_set() is False:
            # torch.cuda.synchronize()  # 确保之前的 CUDA 操作都完成了，准确测量等待时间
            t_wait_start = time.perf_counter()
            full = self.full_q.get()

            t_wait_end = time.perf_counter()
            wait_ms = (t_wait_end - t_wait_start) * 1000.0

            total_wait_ms += wait_ms
            wait_steps += 1

            # if wait_ms > 5.0:  # Skip epoch 0 warmup
            #     logger.debug(
            #         f"[Rank {self.env.local_rank}] Data Starvation! Main loop waited {wait_ms:.2f}ms for full_q"
            #     )

            if isinstance(full, tuple):
                meta, ready_event = full
                torch.cuda.current_stream().wait_event(ready_event)
                if meta.split == "train":  # if training
                    # should_profile = self.curr_epoch == 1 and not profiling_done
                    should_profile = False
                    if should_profile:
                        # 如果是该 Epoch 的前几个 step，确保 profiler 开启
                        if epoch2_steps_recorded == 0:
                            prof.start()

                    with record_function("Model_Train_Step"):
                        now = time.perf_counter()
                        if self._last_train_start_t is not None:
                            interval_ms = (now - self._last_train_start_t) * 1000.0
                            self.train_start2start_ms.update(
                                torch.tensor(interval_ms, device=self.device)
                            )
                        self._last_train_start_t = now
                        self.train(meta)
                    if should_profile:
                        prof.step()  # 标记一步结束
                        epoch2_steps_recorded += 1

                        # 抓取 5 个 batch 后立刻停止并保存
                        if epoch2_steps_recorded >= 100:
                            prof.stop()
                            profiling_done = True

                elif meta.split in ["val", "test"]:  # if evaluating
                    self._last_train_start_t = None
                    self.eval(meta)
            if full == "end_of_epoch":
                avg_wait = total_wait_ms / max(1, wait_steps)
                logger.debug(
                    "[Rank %s] Epoch %s Avg Data Wait: %.2fms",
                    self.env.local_rank,
                    self.curr_epoch,
                    avg_wait,
                )
                # print total_wait_ms
                logger.debug(
                    "[Rank %s] Epoch %s Total Data Wait: %.2fms over %s waits",
                    self.env.local_rank,
                    self.curr_epoch,
                    total_wait_ms,
                    wait_steps,
                )

                total_wait_ms = 0.0
                wait_steps = 0

                result = self.on_train_epoch_end()

                self.result_q.put(result)
                # self.save_model(path="last.ckpt")
                val_auc = result.val_auc if result is not None else 0.0
                if val_auc > self.local_best_auc:
                    self.local_best_auc = val_auc
                    # 只有打破纪录才写磁盘
                    self.save_model(path="best.ckpt")
                dist.barrier(group=self.nccl)
                self.curr_epoch += 1
                if self.mem_box_store:
                    self.mem_box_store.reset()
                torch.cuda.empty_cache()  # 尝试减少内存碎片，帮助稳定训练
            if full is None:
                if self.mem_box_store is not None:
                    assert self.box_task_q is not None
                    assert self.mem_task_q is not None
                    assert self.mem_p2p_q is not None
                    self.box_task_q.put(None)
                    self.mem_task_q.put(None)
                    self.mem_p2p_q.put(None)
                break

    def cleanup(self) -> None:
        self.h2d.join()
        self.receiver.join()
        if self.mem_handler:
            self.mem_handler.join()
        if self.mem_wr:
            self.mem_wr.join()
        if self.box_wr:
            self.box_wr.join()

        if self.box_handler:
            self.box_handler.join()
        if self.p2p_q_handler:
            self.p2p_q_handler.join()
        self.dy_asm.join()
        dist.barrier()
        dist.destroy_process_group()

    # 供外部保存/加载
    def save_model(self, path: str) -> None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if rank == 0:
            # 关键：保存“裸模型”的权重（非DDP也能加载）
            model = self.model.module if isinstance(self.model, DDP) else self.model
            torch.save(model.state_dict(), path)

    def load_model(self, path: str) -> None:
        # 让所有rank都在文件写完后再开始读（更稳）
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.nccl)

        state = torch.load(path, map_location="cpu")

        # 如果当前是DDP，先拿到裸模型来load（因为你保存的是裸模型state_dict）
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(state, strict=True)

        # 如果每rank只可见1张GPU，这里用0没问题；否则用local_rank

        model = model.to(self.device)

        # wrap 成 DDP
        self.model = DDP(model, device_ids=[0])

        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.nccl)

    def load_memory(self, memory_state: dict) -> None:
        if self.mem_box_store is not None:
            self.model.module.load_memory_state(memory_state)
