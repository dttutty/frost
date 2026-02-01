import logging
import threading
from queue import Queue as Q
from typing import Literal  # 必须导入这个

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from buffer import Buffers, init_all_pin_mem
from gnn.with_memory.apan import APAN
from gnn.with_memory.base import MemoryBaseGNN
from gnn.with_memory.jodie import JODIE
from gnn.with_memory.tgn import TGN
from gnn.without_memory.base import GNN
from p2p_syncer.box_handler import BoxHandler
from p2p_syncer.box_writer import BoxWriter
from p2p_syncer.mem_handler import MemHandler
from p2p_syncer.mem_writer import MemWriter
from schemas.alias import MetaWEvent
from schemas.datatypes.cfg import MemoryCFG, SampleCFG, TrainCFG
from schemas.datatypes.env import Env
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.misc import MemNBox, Meta, ModelComponents, Result, Shared
from schemas.datatypes.task import BoxTask, MemTask, P2PTask
from shm_manager import DoubleBufferedSHM
from worker.assembler.dy_asm import DyAssembler
from worker.graph_builder import GraphBuilder
from worker.h2d import H2D
from worker.rebuilder import Rebuilder
from worker.receiver import Receiver as Receiver


logger = logging.getLogger(__name__)


class Worker:
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
        shm_manager: DoubleBufferedSHM,
        m_cfg: MemoryCFG | None = None,
    ) -> None:
        if __debug__:
            logger.debug(f"Worker: initializing on rank {env.local_rank}")
        self.s_cfg = s_cfg
        self.n_neg = s_cfg.neg_samples
        self.save_to = save_to
        self.env = env

        self.shared = shared
        self.total_epoch = t_cfg.epoch
        self.meta_q: Q[Meta | None | Literal["end_of_epoch"]] = Q(maxsize=3)
        self.rebuilder_q: Q[Meta | None | Literal["end_of_epoch"]] = Q(maxsize=4)
        self.graph_builder_q: Q[
            tuple[Meta, torch.cuda.Event] | None | Literal["end_of_epoch"]
        ] = Q(maxsize=4)
        self.stop_ev = threading.Event()
        self.nccl = nccl
        self.curr_epoch = 0
        self.logger: logging.Logger = logger
        self.fetch_factor = 1
        self.model: GNN | MemoryBaseGNN | DDP
        self.full_q: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q(maxsize=1)
        self.device = torch.device(f"cuda:{env.local_rank}")

        self.h2d_sem: threading.Semaphore = threading.Semaphore(0)
        self.destination_q_for_h2d: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q(
            maxsize=1
        )

        if __debug__:
            logger.debug("Worker: initializing buffers")
        self.buffers: Buffers = init_all_pin_mem(
            shared=shared,
            shared_nfeat=nfeat,
            shared_efeat=efeat,
            batch_size=t_cfg.batch_size,
            fanout=s_cfg.neighbor,
            n_neg=self.n_neg,
        )

        if self.shared is not None:
            if __debug__:
                logger.debug("Worker: initializing with shared memory components")
            assert m_cfg is not None
            self.delta_q: Q[tuple[MemNBox, torch.cuda.Event] | Literal["nothing"]] = Q(
                maxsize=2
            )
            self.mem_task_q: Q[tuple[MemTask, torch.cuda.Event] | None] = Q(maxsize=2)
            self.box_task_q: Q[tuple[BoxTask, torch.cuda.Event] | None] = Q(maxsize=2)
            self.mem_p2p_q: Q[
                tuple[P2PTask, torch.cuda.Event] | Literal["nothing_to_p2p"]
            ] = Q(maxsize=2)
            self.box_p2p_q: Q[
                tuple[P2PTask, torch.cuda.Event] | Literal["nothing_to_p2p"]
            ] = Q(maxsize=2)
            self.xpart_q: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q(maxsize=2)
            assert shared is not None, "Shared memory must be provided if memory exists"
            destination_q_for_h2d = self.xpart_q

            if __debug__:
                logger.debug("Worker: starting DyAssembler")
            self.dy_asm = DyAssembler(
                in_q=self.xpart_q,
                delta_q=self.delta_q,
                full_q=self.full_q,
                stop_ev=self.stop_ev,
                device=self.device,
            )
            if __debug__:
                logger.debug("Worker: starting MemHandler")

            self.mem_writer_q: Q[
                tuple[torch.Tensor | None, MemTsNid, torch.cuda.Event]
            ] = Q(maxsize=16)
            self.mem_handler = MemHandler(
                h2d_sem=self.h2d_sem,
                mem_task_q=self.mem_task_q,
                mem_p2p_q=self.mem_p2p_q,
                stop_ev=self.stop_ev,
                device=self.device,
                mem_writer_q=self.mem_writer_q,
            )
            self.mem_writer = MemWriter(
                buf=self.buffers,
                shared_mem=shared.mem,
                shared_mem_ts=shared.mem_ts,
                h2d_sem=self.h2d_sem,
                stop_ev=self.stop_ev,
                mem_writer_q=self.mem_writer_q,
                device=self.device,
            )
            if __debug__:
                logger.debug("Worker: starting BoxHandler")

            self.box_writer_q: Q[
                tuple[torch.Tensor | None, BoxTsNid, torch.cuda.Event]
            ] = Q(maxsize=16)
            self.box_handler = BoxHandler(
                h2d_sem=self.h2d_sem,
                box_task_q=self.box_task_q,
                box_p2p_q=self.box_p2p_q,
                stop_ev=self.stop_ev,
                mail_combine=m_cfg.mail_combine,
                merge_src=t_cfg.merge_src,
                device=self.device,
                box_writer_q=self.box_writer_q,
            )
            self.box_writer = BoxWriter(
                buf=self.buffers,
                shared_box=shared.box,
                shared_box_ts=shared.box_ts,
                h2d_sem=self.h2d_sem,
                stop_ev=self.stop_ev,
                box_writer_q=self.box_writer_q,
                device=self.device,
            )

            if __debug__:
                logger.debug("Worker: all memory handlers started")
        else:
            if __debug__:
                logger.debug("Worker: no shared memory, using direct queue")
            destination_q_for_h2d = self.full_q
            # there is no thread for handling xpart_q, just put to full_q directly

        if __debug__:
            logger.debug("Worker: starting H2D thread")

        self.rebuilder = Rebuilder(
            in_q=self.meta_q,
            out_q=self.rebuilder_q,
            stop_ev=self.stop_ev,
            shm_manager=shm_manager,
        )
        self.graph_builder = GraphBuilder(
            buf=self.buffers,
            efeat=efeat,
            shared=shared,
            in_q=self.rebuilder_q,
            out_q=self.graph_builder_q,
            stop_ev=self.stop_ev,
            history=s_cfg.history,
            device=self.device,
        )
        self.h2d = H2D(
            shm_manager=shm_manager,
            buf=self.buffers,
            nfeat=nfeat,
            efeat=efeat,
            shared=shared,
            device=self.device,
            history=s_cfg.history,
            n_neg=self.n_neg,
            merge_src=t_cfg.merge_src,
            in_q=self.graph_builder_q,
            out_q=destination_q_for_h2d,
            stop_ev=self.stop_ev,
            h2d_sem=self.h2d_sem,
        )

        self.result_q: Q[Result | None] = Q()

        if __debug__:
            logger.debug("Worker: starting Receiver thread")
        self.receiver = Receiver(
            env=self.env,
            gloo=gloo,
            out_q=self.meta_q,
            result_q=self.result_q,
            stop_ev=self.stop_ev,
        )

        if self.shared is not None:
            assert isinstance(components.gnn, TGN | JODIE | APAN)
            assert components.memory_updater is not None
            assert components.header is not None
            self.gnn = components.gnn
            self.model = MemoryBaseGNN(
                gnn=components.gnn,
                memory_updater=components.memory_updater,
                mem_task_q=self.mem_task_q,
                box_task_q=self.box_task_q,
                header=components.header,
            )
        else:
            assert components.header is not None
            self.gnn = components.gnn

            self.model = GNN(
                gnn=components.gnn,
                header=components.header,
            )

        if __debug__:
            logger.debug("Worker: moving model to GPU and wrapping with DDP")
        self.model = self.model.cuda()
        find_unused_parameters = False
        if self.s_cfg.history > 1:
            find_unused_parameters = True
        self.model = DDP(
            self.model,
            process_group=self.nccl,
            find_unused_parameters=find_unused_parameters,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=t_cfg.lr)
        self.train_loss_tracker = MeanMetric(process_group=self.nccl).cuda()

        if __debug__:
            logger.debug("Worker: initializing metrics")
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
        if __debug__:
            logger.debug("Worker: initialization complete")

    def on_train_epoch_end(self) -> Result | None:
        if __debug__:
            logger.debug(f"Worker: epoch {self.curr_epoch} ending, computing metrics")
        # 3. 只有在 Epoch 结束时，才做一次全局同步，计算精准的平均 Loss 记入日志
        train_loss = self.train_loss_tracker.compute()

        val_results = self.val_metrics.compute()
        test_results = self.test_metrics.compute()
        # if dist.get_rank() == 0:
        #     print(f"Epoch{self.curr_epoch} Done. ")
        #     print(f"Train Loss: {train_loss}, Val Loss {val_results['val_loss']}")
        #     print(f"Val AUC: {val_results['val_auc']}, Val AP: {val_results['val_ap']}")
        #     print(
        #         f"Test AUC: {test_results['test_auc']}, Test AP: {test_results['test_ap']}"
        #     )
        if __debug__:
            logger.debug("Worker: resetting metrics")
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
        if __debug__:
            logger.debug(f"Worker: epoch {self.curr_epoch} results computed")
        if dist.get_rank() == 0:
            return res
        else:
            return None

    def train(self, meta: Meta) -> None:
        if __debug__:
            logger.debug("Worker: starting training step")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if __debug__:
            logger.debug("Worker: forward pass")
        logits = self.model(meta)
        loss = self.criterion(logits.squeeze(-1), meta.labels)
        if __debug__:
            logger.debug(f"Worker: loss computed: {loss.item():.4f}")

        if __debug__:
            logger.debug("Worker: waiting at barrier before backward")
        # dist.barrier(group=self.nccl)
        if __debug__:
            logger.debug("Worker: backward pass")

        if self.shared is not None:
            if __debug__:
                logger.debug("Worker: handling P2P queue")
            self.handle_p2p_q()
        loss.backward()

        self.optimizer.step()
        self.train_loss_tracker.update(loss)
        if __debug__:
            logger.debug("Worker: training step complete")

    @torch.no_grad()
    def handle_p2p_q(self) -> None:
        if __debug__:
            logger.debug("Worker: handling P2P communication (Mem + Box)")

        # 1. 获取任务 (并行获取，减少等待间隙)
        # 注意：这里假设 mem 和 box 的 nothing_to_p2p 状态是同步的
        mem_p2p_tuple = self.mem_p2p_q.get(block=True)

        # 快速路径：如果没有 P2P 任务，直接跳过
        if mem_p2p_tuple == "nothing_to_p2p":
            box_p2p_tuple = self.box_p2p_q.get(block=True)
            assert box_p2p_tuple == "nothing_to_p2p"
            if __debug__:
                logger.debug("Worker: skipping P2P, putting skip to delta_q")
            self.delta_q.put("nothing")
            return
        mem_p2p, mem_event = mem_p2p_tuple
        # 此时肯定有 box 任务
        box_p2p_tuple = self.box_p2p_q.get(block=True)
        assert box_p2p_tuple != "nothing_to_p2p"
        box_p2p, box_event = box_p2p_tuple
        torch.cuda.current_stream().wait_event(mem_event)
        torch.cuda.current_stream().wait_event(box_event)

        # 2. 准备 NCCL 通讯列表 (Flatten the Loop)
        # 顺序：Mem_NID -> Mem_Data -> Mem_TS -> Box_NID -> Box_Data -> Box_TS
        # 这样写的好处是 NCCL Kernel 会被密集地压入 CUDA Stream，没有任何 Python 逻辑打断
        p2p_ops = [
            # --- Mem Parts ---
            (mem_p2p.nid2recv, mem_p2p.nid2send, mem_p2p.rec_cts, mem_p2p.send_cts),
            (mem_p2p.data2recv, mem_p2p.data2send, mem_p2p.rec_cts, mem_p2p.send_cts),
            (mem_p2p.ts2recv, mem_p2p.ts2send, mem_p2p.rec_cts, mem_p2p.send_cts),
            # --- Box Parts ---
            (box_p2p.nid2recv, box_p2p.nid2send, box_p2p.rec_cts, box_p2p.send_cts),
            (box_p2p.data2recv, box_p2p.data2send, box_p2p.rec_cts, box_p2p.send_cts),
            (box_p2p.ts2recv, box_p2p.ts2send, box_p2p.rec_cts, box_p2p.send_cts),
        ]

        # 3. 执行所有通信
        if __debug__:
            logger.debug(f"Worker: executing merged P2P batch ({len(p2p_ops)} ops)")

        for i, (recv_t, send_t, rec_cts, send_cts) in enumerate(p2p_ops):
            dist.all_to_all_single(
                output=recv_t,
                input=send_t,
                output_split_sizes=rec_cts,
                input_split_sizes=send_cts,
                group=self.nccl,
                # async_op=True # 如果是较新版本的 PyTorch 且需要重叠计算，可以开启，但这里通常不需要
            )

        # 4. 组装结果
        delta_mem = MemTsNid(
            nid=mem_p2p.nid2recv, data=mem_p2p.data2recv, ts=mem_p2p.ts2recv
        )
        delta_box = BoxTsNid(
            nid=box_p2p.nid2recv, data=box_p2p.data2recv, ts=box_p2p.ts2recv
        )

        delta = MemNBox(mem=delta_mem, box=delta_box)

        # 5. 记录 Event 并发送
        ready_event: torch.cuda.Event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())

        if __debug__:
            logger.debug("Worker: putting merged delta to delta_q")
        self.delta_q.put((delta, ready_event))

    @torch.no_grad()
    def eval(self, meta: Meta) -> None:
        if __debug__:
            logger.debug(f"Worker: evaluating on split={meta.m_state.split}")
        assert meta.labels is not None
        self.model.eval()

        logits = self.model(meta)
        if self.shared is not None:
            self.handle_p2p_q()

        y_pred = logits.sigmoid().squeeze(-1)

        if meta.m_state.split == "val":
            if __debug__:
                logger.debug("Worker: updating validation metrics")
            loss = self.criterion(logits.squeeze(-1), meta.labels)

            # [Fix] 显式指定哪个参数给哪个指标，或者分开更新
            # MeanMetric 需要 'value'，分类指标需要 'preds' 和 'target'
            self.val_metrics.update(
                preds=y_pred,
                target=meta.labels.long(),
                value=loss,  # 这一行会让 val_loss 找到它需要的参数
            )
        else:
            assert meta.m_state.split == "test"
            if __debug__:
                logger.debug("Worker: updating test metrics")
            self.test_metrics.update(preds=y_pred, target=meta.labels.long())

    def start_all_threads(self) -> None:
        if __debug__:
            logger.debug("Worker: starting all threads")
        self.rebuilder.start()
        self.graph_builder.start()
        self.h2d.start()
        self.receiver.start()
        if self.shared is not None:
            self.mem_handler.start()
            self.box_handler.start()
            self.box_writer.start()
            self.mem_writer.start()
            self.dy_asm.start()
        if __debug__:
            logger.debug("Worker: all threads started")

    def run(self) -> None:
        self.start_all_threads()
        if __debug__:
            logger.debug("Worker: starting main run loop")
        if __debug__:
            logger.debug("Worker: Starting H2D thread...")
        if __debug__:
            logger.debug("Worker: releasing semaphores for H2D startup")
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        if self.shared is not None:
            if __debug__:
                logger.debug("Worker: initializing delta_q with None event")
            self.delta_q.put("nothing")

        batch_count = 0
        while self.stop_ev.is_set() is False:
            if __debug__:
                logger.debug("Worker: waiting for data from full_q")
            full = self.full_q.get()

            if isinstance(full, tuple):
                batch_count += 1
                if __debug__:
                    logger.debug(f"Worker: processing batch #{batch_count}")
                meta, ready_event = full
                torch.cuda.current_stream().wait_event(ready_event)
                m_info = meta.m_state
                if __debug__:
                    logger.debug(f"Worker: batch #{batch_count} split={m_info.split}")
                if m_info.split == "train":  # if training
                    self.train(meta)
                elif m_info.split in ["val", "test"]:  # if evaluating
                    self.eval(meta)
                if __debug__:
                    logger.debug(f"Worker: batch #{batch_count} complete")
            if full == "end_of_epoch":
                if __debug__:
                    logger.debug(f"Worker: end of epoch {self.curr_epoch}")
                result = self.on_train_epoch_end()
                self.result_q.put(result)
                if __debug__:
                    logger.debug(
                        f"Worker: saving model checkpoint (epoch {self.curr_epoch})"
                    )
                # self.save_model(path="last.ckpt")
                val_auc = result.val_auc if result is not None else 0.0
                if __debug__:
                    logger.debug(
                        f"DEBUG_RANK_{self.env.local_rank}: current_auc={val_auc}, best_auc={self.local_best_auc}"
                    )
                if val_auc > self.local_best_auc:
                    if __debug__:
                        logger.debug(
                            f"Worker: new best AUC: {val_auc:.4f} > {self.local_best_auc:.4f}"
                        )
                    self.local_best_auc = val_auc
                    # 只有打破纪录才写磁盘
                    self.save_model(path="best.ckpt")
                dist.barrier(group=self.nccl)
                self.curr_epoch += 1
                if self.shared:
                    if __debug__:
                        logger.debug("Worker: resetting shared memory")
                    self.shared.reset()
            if full is None:
                self.box_task_q.put(None)
                self.mem_task_q.put(None)
                if __debug__:
                    logger.debug("Worker: received None, synchronizing and exiting")

                break

        if __debug__:
            logger.debug(f"Worker: run loop complete, processed {batch_count} batches")

    def cleanup(self) -> None:
        if __debug__:
            logger.debug("Worker: cleanup started")
        self.h2d.join()
        if __debug__:
            logger.debug("Worker: H2D thread joined")
        self.receiver.join()
        if __debug__:
            logger.debug("Worker: Receiver thread joined")
        if self.mem_handler:
            self.mem_handler.join()
            if __debug__:
                logger.debug("Worker: MemHandler thread joined")
        if self.mem_writer:
            self.mem_writer.join()
            if __debug__:
                logger.debug("Worker: MemWriter thread joined")
        if self.box_writer:
            self.box_writer.join()
            if __debug__:
                logger.debug("Worker: BoxWriter thread joined")
        if self.rebuilder:
            self.rebuilder.join()
            if __debug__:
                logger.debug("Worker: Rebuilder thread joined")
        if self.graph_builder:
            self.graph_builder.join()
            if __debug__:
                logger.debug("Worker: GraphBuilder thread joined")
        if self.box_handler:
            self.box_handler.join()
            if __debug__:
                logger.debug("Worker: BoxHandler thread joined")
        self.dy_asm.join()
        if __debug__:
            logger.debug("Worker: Assembler thread joined")
        dist.barrier()
        dist.destroy_process_group()
        if __debug__:
            logger.debug("Worker: cleanup complete")

    # 供外部保存/加载
    def save_model(self, path: str) -> None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if __debug__:
            logger.debug(f"Worker: save_model called on rank {rank}, path={path}")

        if rank == 0:
            if __debug__:
                logger.debug(f"Worker: rank 0 saving model to {path}")
            # 关键：保存“裸模型”的权重（非DDP也能加载）
            model = self.model.module if isinstance(self.model, DDP) else self.model
            torch.save(model.state_dict(), path)

    def load_model(self, path: str) -> None:
        if __debug__:
            logger.debug(f"Worker: load_model called, path={path}")
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
        if __debug__:
            logger.debug("Worker: load_memory called")
        if self.shared is not None:
            self.model.module.load_memory_state(memory_state)
            if __debug__:
                logger.debug("Worker: memory state loaded")
