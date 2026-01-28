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

from buffer import Buffers, _init_all_pin_mem
from gnn.with_memory.apan import APAN
from gnn.with_memory.base import MemoryBaseGNN
from gnn.with_memory.jodie import JODIE
from gnn.with_memory.tgn import TGN
from gnn.without_memory.base import GNN
from schemas.alias import MetaWEvent
from schemas.datatypes.cfg import MemoryCFG, SampleCFG, TrainCFG
from schemas.datatypes.env import Env
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.misc import MemNBox, Meta, ModelComponents, Result, Shared
from schemas.datatypes.task import BoxTask, MemTask, P2PTask
from worker.assembler.dy_asm import DyAssembler
from worker.data_syncer.box_handler import BoxHandler
from worker.data_syncer.mem_handler import MemHandler
from worker.h2d import H2D
from worker.receiver import Receiver as Receiver


logger = logging.getLogger(__name__)


# def tensorfy(self, plan_set: ThreePlan):
#     for plan in [plan_set.mem_plan, plan_set.mail_plan, plan_set.feat_plan]:
#         if plan is not None:
#             plan.rec_cts = torch.tensor(plan.rec_cts, dtype=torch.long)
#             plan.send_cts = torch.tensor(plan.send_cts, dtype=torch.long)


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
        m_cfg: MemoryCFG | None = None,
    ) -> None:
        logger.debug(f"Worker: initializing on rank {env.local_rank}")
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

        self.h2d_sem: threading.Semaphore = threading.Semaphore(0)
        self.destination_q_for_h2d: Q[MetaWEvent | None | Literal["end_of_epoch"]]

        logger.debug("Worker: initializing buffers")
        self.buffers: Buffers = _init_all_pin_mem(
            shared=shared,
            shared_nfeat=nfeat,
            shared_efeat=efeat,
            batch_size=t_cfg.batch_size,
            fanout=s_cfg.neighbor,
            n_neg=self.n_neg,
        )

        if self.shared is not None:
            logger.debug("Worker: initializing with shared memory components")
            assert m_cfg is not None
            self.delta_q: Q[tuple[MemNBox, torch.cuda.Event] | Literal["nothing"]] = Q()
            self.mem_task_q: Q[tuple[MemTask, torch.cuda.Event] | None] = Q()
            self.box_task_q: Q[tuple[BoxTask, torch.cuda.Event] | None] = Q()
            self.mem_p2p_q: Q[P2PTask | Literal["nothing_to_p2p"]] = Q()
            self.box_p2p_q: Q[P2PTask | Literal["nothing_to_p2p"]] = Q()
            self.xpart_q: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q()
            assert shared is not None, "Shared memory must be provided if memory exists"
            destination_q_for_h2d = self.xpart_q

            logger.debug("Worker: starting DyAssembler")
            self.assembler = DyAssembler(
                in_q=self.xpart_q,
                delta_q=self.delta_q,
                full_q=self.full_q,
                stop_ev=self.stop_ev,
            )
            logger.debug("Worker: starting MemHandler")
            self.mem_handler = MemHandler(
                buf=self.buffers,
                shared_mem=shared.mem,
                shared_mem_ts=shared.mem_ts,
                mem_task_q=self.mem_task_q,
                mem_p2p_q=self.mem_p2p_q,
                h2d_sem=self.h2d_sem,
                stop_ev=self.stop_ev,
            )
            self.mem_handler.start()
            logger.debug("Worker: starting BoxHandler")
            self.box_handler = BoxHandler(
                buf=self.buffers,
                shared_box=shared.box,
                shared_box_ts=shared.box_ts,
                box_task_q=self.box_task_q,
                box_p2p_q=self.box_p2p_q,
                h2d_sem=self.h2d_sem,
                stop_ev=self.stop_ev,
                mail_combine=m_cfg.mail_combine,
                merge_src=t_cfg.merge_src,
            )
            self.box_handler.start()

            self.assembler.start()  # 启动 Assembler 组装线程
            logger.debug("Worker: all memory handlers started")
        else:
            logger.debug("Worker: no shared memory, using direct queue")
            destination_q_for_h2d = self.full_q
            # there is no thread for handling xpart_q, just put to full_q directly

        logger.debug("Worker: starting H2D thread")
        self.h2d = H2D(
            buf=self.buffers,
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
        logger.debug("Worker: initialization complete")

    def on_train_epoch_end(self) -> Result | None:
        logger.debug(f"Worker: epoch {self.curr_epoch} ending, computing metrics")
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
        logger.debug(f"Worker: epoch {self.curr_epoch} results computed")
        if dist.get_rank() == 0:
            return res
        else:
            return None

    def train(self, meta: Meta) -> None:
        logger.debug("Worker: starting training step")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logger.debug("Worker: forward pass")
        logits = self.model(meta)
        loss = self.criterion(logits.squeeze(-1), meta.labels)
        logger.debug(f"Worker: loss computed: {loss.item():.4f}")
        if self.shared is not None:
            logger.debug("Worker: handling P2P queue")
            self.handle_p2p_q()
        logger.debug("Worker: waiting at barrier before backward")
        dist.barrier(group=self.nccl)
        logger.debug("Worker: backward pass")
        loss.backward()
        self.optimizer.step()
        self.train_loss_tracker.update(loss)
        logger.debug("Worker: training step complete")

    @torch.no_grad()
    def handle_p2p_q(self) -> None:
        logger.debug("Worker: handling P2P communication")

        # only support 1-d mem and 2-d box p2p now
        # if you need to sync mail, convert it to box  first
        def _exec_p2p(
            p2p_task: P2PTask, type: Literal["mem", "box"]
        ) -> MemTsNid | BoxTsNid:
            logger.debug(f"Worker: executing P2P for type={type}")
            iters = [
                (p2p_task.nid2recv, p2p_task.nid2send),
                (p2p_task.data2recv, p2p_task.data2send),
                (p2p_task.ts2recv, p2p_task.ts2send),
            ]
            for idx, (recv_tensor, send_tensor) in enumerate(iters):
                logger.debug(f"Worker: P2P all_to_all iteration {idx}")
                dist.all_to_all_single(
                    output=recv_tensor,
                    input=send_tensor,
                    output_split_sizes=p2p_task.rec_cts,
                    input_split_sizes=p2p_task.send_cts,
                    group=self.nccl,
                )
            cls: type[MemTsNid] | type[BoxTsNid]
            match type:
                case "mem":
                    cls = MemTsNid
                case "box":
                    cls = BoxTsNid
                case _:
                    # Handle unknown type
                    raise ValueError(f"Unknown type: {type}")
            return cls(
                nid=p2p_task.nid2recv, data=p2p_task.data2recv, ts=p2p_task.ts2recv
            )

        delta_mem = None
        delta_box = None

        # 1. 强制先处理 Mem 任务 (Deterministic Order: Mem -> Box)
        # 注意：如果某个 microbatch 不需要 mem p2p，DataSyncer 必须 put(None) 作为哨兵
        logger.debug("Worker: getting mem P2P task")
        mem_p2p = self.mem_p2p_q.get(block=True)

        if mem_p2p == "nothing_to_p2p":
            logger.debug("Worker: skipping mem P2P")
            box_p2p = self.box_p2p_q.get(block=True)
            assert box_p2p == "nothing_to_p2p"
            logger.debug("Worker: skipping box P2P, putting skip to delta_q")
            self.delta_q.put("nothing")
        else:
            logger.debug("Worker: executing mem P2P")
            delta_mem = _exec_p2p(mem_p2p, "mem")
            assert isinstance(delta_mem, MemTsNid)
            logger.debug("Worker: getting box P2P task")
            box_p2p = self.box_p2p_q.get(block=True)
            assert box_p2p != "nothing_to_p2p"
            logger.debug("Worker: executing box P2P")
            delta_box = _exec_p2p(box_p2p, "box")
            assert isinstance(delta_box, BoxTsNid)

            delta = MemNBox(mem=delta_mem, box=delta_box)
            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            logger.debug("Worker: putting delta to delta_q")
            self.delta_q.put((delta, ready_event))

    @torch.no_grad()
    def eval(self, meta: Meta) -> None:
        logger.debug(f"Worker: evaluating on split={meta.m_state.split}")
        assert meta.labels is not None
        self.model.eval()
        logits_pos, logits_neg = self.model(meta)
        if self.shared is not None:
            logger.debug("Worker: handling P2P queue in eval")
            self.handle_p2p_q()
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        y_pred = logits.sigmoid()

        if meta.m_state.split == "val":
            logger.debug("Worker: updating validation metrics")
            loss = self.criterion(logits.squeeze(-1), meta.labels)

            self.val_metrics.update(preds=y_pred, target=meta.labels)
            self.val_metrics["val_loss"].update(loss)
        else:
            assert meta.m_state.split == "test"
            logger.debug("Worker: updating test metrics")
            self.test_metrics.update(preds=y_pred, target=meta.labels)

    def start_all_threads(self) -> None:
        logger.debug("Worker: starting all threads")
        self.h2d.start()
        self.receiver.start()
        if self.shared is not None:
            self.mem_handler.start()
            self.box_handler.start()
            self.assembler.start()
        logger.debug("Worker: all threads started")

    def start(self) -> None:
        self.start_all_threads()
        logger.debug("Worker: starting main run loop")
        self.logger.info("Worker: Starting H2D thread...")
        logger.debug("Worker: releasing semaphores for H2D startup")
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        if self.shared is not None:
            logger.debug("Worker: initializing delta_q with None event")
            self.delta_q.put("nothing")

        batch_count = 0
        while self.stop_ev.is_set() is False:
            logger.debug("Worker: waiting for data from full_q")
            full = self.full_q.get()

            if isinstance(full, tuple):
                batch_count += 1
                logger.debug(f"Worker: processing batch #{batch_count}")
                meta, ready_event = full
                torch.cuda.current_stream().wait_stream(ready_event)
                m_info = meta.m_state
                logger.debug(f"Worker: batch #{batch_count} split={m_info.split}")
                if m_info.split == "train":  # if training
                    self.train(meta)
                elif m_info.split in ["val", "test"]:  # if evaluating
                    self.eval(meta)
                logger.debug(f"Worker: batch #{batch_count} complete")
            if full == "end_of_epoch":
                logger.debug(f"Worker: end of epoch {self.curr_epoch}")
                result = self.on_train_epoch_end()
                self.result_q.put(result)
                logger.debug(
                    f"Worker: saving model checkpoint (epoch {self.curr_epoch})"
                )
                self.save_model(path="last.ckpt")
                val_auc = result.val_auc if result is not None else 0.0
                if val_auc > self.local_best_auc:
                    logger.debug(
                        f"Worker: new best AUC: {val_auc:.4f} > {self.local_best_auc:.4f}"
                    )
                    self.local_best_auc = val_auc
                    # 只有打破纪录才写磁盘
                    self.save_model(path="best.ckpt")
                self.curr_epoch += 1
                if self.shared:
                    logger.debug("Worker: resetting shared memory")
                    self.shared.reset()
            if full is None:
                self.box_task_q.put(None)
                self.mem_task_q.put(None)
                logger.debug("Worker: received None, synchronizing and exiting")
                torch.cuda.synchronize()
                break

        logger.debug(f"Worker: run loop complete, processed {batch_count} batches")

    def cleanup(self) -> None:
        logger.debug("Worker: cleanup started")
        self.h2d.join()
        logger.debug("Worker: H2D thread joined")
        self.receiver.join()
        logger.debug("Worker: Receiver thread joined")
        if self.mem_handler:
            self.mem_handler.join()
            logger.debug("Worker: MemHandler thread joined")
        if self.box_handler:
            self.box_handler.join()
            logger.debug("Worker: BoxHandler thread joined")
        self.assembler.join()
        logger.debug("Worker: Assembler thread joined")
        dist.barrier()
        dist.destroy_process_group()
        logger.debug("Worker: cleanup complete")

    # 供外部保存/加载
    def save_model(self, path: str) -> None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        logger.debug(f"Worker: save_model called on rank {rank}, path={path}")

        if rank == 0:
            logger.debug(f"Worker: rank 0 saving model to {path}")
            # 关键：保存“裸模型”的权重（非DDP也能加载）
            model = self.model.module if isinstance(self.model, DDP) else self.model
            torch.save(model.state_dict(), path)

        if dist.is_available() and dist.is_initialized():
            logger.debug("Worker: waiting at barrier after save")
            dist.barrier()

    def load_model(self, path: str) -> None:
        logger.debug(f"Worker: load_model called, path={path}")
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
        logger.debug("Worker: load_memory called")
        if self.shared is not None:
            self.model.module.load_memory_state(memory_state)
            logger.debug("Worker: memory state loaded")
