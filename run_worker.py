import logging
import threading
import time
from queue import Queue as Q
from typing import Any, Literal, cast  # 必须导入这个

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)
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
from p2p_syncer.p2p_q_handler import P2PQHandler
from schemas.alias import MetaWEvent
from schemas.datatypes.cfg import MemoryCFG, SampleCFG, TrainCFG
from schemas.datatypes.env import Env
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.misc import Meta, ModelComponents, Result, Shared
from schemas.datatypes.task import BoxTask, MemTask, P2PTask
from schemas.protocol.misc import CudaEventLike
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
        nccl_p2p: dist.ProcessGroup,
        gloo: dist.ProcessGroup,
        components: ModelComponents,
        shm_manager: DoubleBufferedSHM,
        m_cfg: MemoryCFG | None = None,
        pin_mem: bool = False,
    ) -> None:
        self.s_cfg = s_cfg
        self.n_neg = s_cfg.neg_samples
        self.save_to = save_to
        self.env = env

        self.shared = shared
        self.total_epoch = t_cfg.epoch
        self.meta_q: Q[Meta | None | Literal["end_of_epoch"]] = Q(maxsize=1)
        self.rebuilder_q: Q[Meta | None | Literal["end_of_epoch"]] = Q(maxsize=1)
        self.graph_builder_q: Q[
            tuple[Meta, CudaEventLike] | None | Literal["end_of_epoch"]
        ] = Q(maxsize=1)
        self.stop_ev = threading.Event()
        self.nccl = nccl
        self.nccl_p2p = nccl_p2p
        self.curr_epoch = 0
        self.logger: logging.Logger = logger
        self.fetch_factor = 1
        self.model: GNN | MemoryBaseGNN | DDP
        self.full_q: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q(maxsize=1)
        self.device = torch.device(f"cuda:{env.local_rank}")
        self.comm = cast(torch.cuda.Stream, torch.cuda.Stream(device=self.device))
        self.h2d_sem: threading.Semaphore = threading.Semaphore(0)
        self.destination_q_for_h2d: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q(
            maxsize=1
        )

        self.train_start2start_ms = MeanMetric(process_group=self.nccl).cuda()
        self._last_train_start_t: float | None = None

        self.pin_mem = pin_mem
        self.buffers: Buffers | None = (
            init_all_pin_mem(
                shared=shared,
                shared_nfeat=nfeat,
                shared_efeat=efeat,
                batch_size=t_cfg.batch_size,
                fanout=s_cfg.neighbor,
                n_neg=self.n_neg,
            )
            if self.pin_mem
            else None
        )
        self.p2p_sem = threading.Semaphore(
            0
        )  # 信号量，确保同一时间只有一个 P2P 任务在进行

        if self.shared is not None:
            assert m_cfg is not None
            self.delta_q: Q[tuple[P2PTask, P2PTask, list[Any]] | Literal["nothing"]] = (
                Q(maxsize=1)
            )
            self.mem_task_q: Q[tuple[MemTask, CudaEventLike] | None] = Q(maxsize=1)
            self.box_task_q: Q[tuple[BoxTask, CudaEventLike] | None] = Q(maxsize=1)
            self.mem_p2p_q: Q[
                tuple[P2PTask, CudaEventLike] | Literal["nothing_to_p2p"] | None
            ] = Q(maxsize=1)
            self.box_p2p_q: Q[
                tuple[P2PTask, CudaEventLike] | Literal["nothing_to_p2p"]
            ] = Q(maxsize=1)
            self.xpart_q: Q[MetaWEvent | None | Literal["end_of_epoch"]] = Q(maxsize=1)
            assert shared is not None, "Shared memory must be provided if memory exists"
            destination_q_for_h2d = self.xpart_q

            self.dy_asm = DyAssembler(
                in_q=self.xpart_q,
                delta_q=self.delta_q,
                full_q=self.full_q,
                stop_ev=self.stop_ev,
                device=self.device,
            )

            self.mem_writer_q: Q[
                tuple[torch.Tensor | None, MemTsNid, CudaEventLike]
            ] = Q(maxsize=1)
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

            self.box_writer_q: Q[
                tuple[torch.Tensor | None, BoxTsNid, CudaEventLike]
            ] = Q(maxsize=1)
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

            self.p2p_q_handler = P2PQHandler(
                buf=self.buffers,
                env=self.env,
                mem_p2p_q=self.mem_p2p_q,
                box_p2p_q=self.box_p2p_q,
                nccl_p2p=self.nccl_p2p,
                p2p_sem=self.p2p_sem,
                stop_ev=self.stop_ev,
                mem_writer_q=self.mem_writer_q,
                delta_q=self.delta_q,
                device=self.device,
            )

        else:
            destination_q_for_h2d = self.full_q
            # there is no thread for handling xpart_q, just put to full_q directly

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

        self.result_q: Q[Result | None] = Q(maxsize=1)

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

            def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                return total, trainable

            # total, trainable = count_parameters(self.model)

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
        self.local_best_auc = 0.0

    def on_train_epoch_end(self) -> Result | None:
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
        if dist.get_rank() == 0:
            return res
        else:
            return None

    def train(self, meta: Meta) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(meta)
        loss = self.criterion(logits.squeeze(-1), meta.labels)

        # logger.debug(f"Rank {dist.get_rank()} - Train Step Loss: {loss.item():.4f}")

        # dist.barrier(group=self.nccl)
        # logger.debug(f"Rank {dist.get_rank()} - Train Step Loss: {loss.item():.4f}")

        # torch.cuda.synchronize()
        if self.shared is not None:
            # t0 = time.perf_counter()
            self.p2p_sem.acquire()  # 获取信号量，确保当前没有其他 P2P 任务在进行
        #     duration = time.perf_counter() - t0
        #     logger.debug(
        #         f"Rank {dist.get_rank()} - Acquired P2P semaphore after waiting {duration * 1000:.2f} ms"
        #     )
        loss.backward()

        self.optimizer.step()
        self.train_loss_tracker.update(loss.detach())

    # @torch.no_grad()
    # def handle_p2p_q(self) -> None:
    #     # 1. 获取任务 (并行获取，减少等待间隙)
    #     # 注意：这里假设 mem 和 box 的 nothing_to_p2p 状态是同步的
    #     t_start = time.perf_counter()
    #     mem_p2p_tuple = self.mem_p2p_q.get(block=True)
    #     t_got_mem = time.perf_counter()
    #     mem_wait_ms = (t_got_mem - t_start) * 1000.0

    #     # 快速路径：如果没有 P2P 任务，直接跳过
    #     if mem_p2p_tuple == "nothing_to_p2p":
    #         box_p2p_tuple = self.box_p2p_q.get(block=True)
    #         assert box_p2p_tuple == "nothing_to_p2p"
    #         self.delta_q.put("nothing")

    #         if mem_wait_ms > 5.0:
    #             logger.debug(
    #                 f"[Rank {self.env.local_rank}] P2P Starvation (Nothing): Waited {mem_wait_ms:.2f}ms for empty signal"
    #             )

    #         return
    #     mem_p2p, mem_event = mem_p2p_tuple
    #     # 此时肯定有 box 任务
    #     box_p2p_tuple = self.box_p2p_q.get(block=True)
    #     t_got_box = time.perf_counter()
    #     box_wait_ms = (t_got_box - t_got_mem) * 1000.0
    #     total_wait = mem_wait_ms + box_wait_ms
    #     if total_wait > 2.0:
    #         logger.debug(
    #             f"[Rank {self.env.local_rank}] P2P Blocked! Total: {total_wait:.2f}ms | Mem: {mem_wait_ms:.2f}ms | Box: {box_wait_ms:.2f}ms"
    #         )
    #     assert box_p2p_tuple != "nothing_to_p2p"
    #     box_p2p, box_event = box_p2p_tuple
    #     torch.cuda.current_stream().wait_event(mem_event)
    #     torch.cuda.current_stream().wait_event(box_event)

    #     # 2. 准备 NCCL 通讯列表 (Flatten the Loop)
    #     # 顺序：Mem_NID -> Mem_Data -> Mem_TS -> Box_NID -> Box_Data -> Box_TS
    #     # 这样写的好处是 NCCL Kernel 会被密集地压入 CUDA Stream，没有任何 Python 逻辑打断
    #     p2p_ops = [
    #         # 格式：(Name, Recv, Send, Rec_Cts, Send_Cts)
    #         (
    #             "Mem_NID",
    #             mem_p2p.nid2recv,
    #             mem_p2p.nid2send,
    #             mem_p2p.rec_cts,
    #             mem_p2p.send_cts,
    #         ),
    #         (
    #             "Mem_Data",
    #             mem_p2p.data2recv,
    #             mem_p2p.data2send,
    #             mem_p2p.rec_cts,
    #             mem_p2p.send_cts,
    #         ),
    #         (
    #             "Mem_TS",
    #             mem_p2p.ts2recv,
    #             mem_p2p.ts2send,
    #             mem_p2p.rec_cts,
    #             mem_p2p.send_cts,
    #         ),
    #         (
    #             "Box_NID",
    #             box_p2p.nid2recv,
    #             box_p2p.nid2send,
    #             box_p2p.rec_cts,
    #             box_p2p.send_cts,
    #         ),
    #         (
    #             "Box_Data",
    #             box_p2p.data2recv,
    #             box_p2p.data2send,
    #             box_p2p.rec_cts,
    #             box_p2p.send_cts,
    #         ),
    #         (
    #             "Box_TS",
    #             box_p2p.ts2recv,
    #             box_p2p.ts2send,
    #             box_p2p.rec_cts,
    #             box_p2p.send_cts,
    #         ),
    #     ]

    #     # if torch.equal(mem_p2p.nid2recv, box_p2p.nid2recv) == False:
    #     #     logger.debug("Error: Mem P2P NIDs and Box P2P NIDs to send do not match.")
    #     #     logger.debug("Mem NIDs to send: %s", mem_p2p.nid2send)
    #     #     logger.debug("Box NIDs to send: %s", box_p2p.nid2send)
    #     #     raise ValueError("Mismatched NIDs in P2P send buffers.")

    #     # 3. 执行所有通信

    #     works = []
    #     for i, (name, recv_t, send_t, rec_cts, send_cts) in enumerate(p2p_ops):
    #         # --- [DEBUG] 强力检查代码开始 ---
    #         total_send = sum(send_cts)
    #         real_shape = send_t.shape[0]
    #         if total_send != real_shape:
    #             import sys

    #             logger.debug(
    #                 f"\n[CRITICAL ERROR] P2P Mismatch on Rank {dist.get_rank()} | Op: {name}"
    #             )
    #             logger.debug(f"  > Sum of send_cts: {total_send}")
    #             logger.debug(f"  > Actual Tensor dim0: {real_shape}")
    #             logger.debug(f"  > Tensor Full Shape: {send_t.shape}")
    #             logger.debug(f"  > Send Cts List: {send_cts}")
    #             sys.stdout.flush()
    #             # 抛出异常让程序停在这里
    #             raise RuntimeError(
    #                 f"Split size mismatch in {name}: planned {total_send} vs actual {real_shape}"
    #             )
    #         # --- [DEBUG] 强力检查代码结束 ---

    #         work = dist.all_to_all_single(
    #             output=recv_t,
    #             input=send_t,
    #             output_split_sizes=rec_cts,
    #             input_split_sizes=send_cts,
    #             group=self.nccl_p2p,
    #             async_op=True,
    #         )
    #         works.append(work)

    #     # # 4. 组装结果
    #     # delta_mem = MemTsNid(
    #     #     nid=mem_p2p.nid2recv, data=mem_p2p.data2recv, ts=mem_p2p.ts2recv
    #     # )
    #     # delta_box = BoxTsNid(
    #     #     nid=box_p2p.nid2recv, data=box_p2p.data2recv, ts=box_p2p.ts2recv
    #     # )

    #     # delta = MemNBox(mem=delta_mem, box=delta_box)

    #     # # 5. 记录 Event 并发送
    #     # ready_event: CudaEventLike = torch.cuda.Event()
    #     # ready_event.record(torch.cuda.current_stream())

    #     self.delta_q.put((mem_p2p, box_p2p, works))

    @torch.no_grad()
    def eval(self, meta: Meta) -> None:
        assert meta.labels is not None
        self.model.eval()

        logits = self.model(meta)
        if self.shared is not None:
            self.p2p_sem.acquire()  # 获取信号量，确保当前没有其他 P2P 任务在进行
        #     # with torch.cuda.stream(self.comm):
        #     #     self.handle_p2p_q()

        y_pred = logits.sigmoid().squeeze(-1)

        if meta.m_state.split == "val":
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
            assert meta.m_state.split == "test"
            self.test_metrics.update(preds=y_pred, target=meta.labels.long())

    def start_all_threads(self) -> None:
        self.rebuilder.start()
        self.graph_builder.start()
        self.h2d.start()
        self.receiver.start()
        if self.shared is not None:
            self.p2p_q_handler.start()
            self.mem_handler.start()
            self.box_handler.start()
            self.box_writer.start()
            self.mem_writer.start()
            self.dy_asm.start()

    def run(self) -> None:
        self.start_all_threads()
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第一个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        self.h2d_sem.release()  # 释放一次信号量，启动第2个 H2D 操作
        if self.shared is not None:
            self.delta_q.put("nothing")

        batch_count = 0

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
                batch_count += 1
                meta, ready_event = full
                torch.cuda.current_stream().wait_event(ready_event)
                m_info = meta.m_state
                if m_info.split == "train":  # if training
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
                        # ----------------------------------------------------------
                        self.train(meta)
                    if should_profile:
                        prof.step()  # 标记一步结束
                        epoch2_steps_recorded += 1

                        # 抓取 5 个 batch 后立刻停止并保存
                        if epoch2_steps_recorded >= 100:
                            prof.stop()
                            profiling_done = True

                elif m_info.split in ["val", "test"]:  # if evaluating
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
                if self.shared:
                    self.shared.reset()
            if full is None:
                self.box_task_q.put(None)
                self.mem_task_q.put(None)
                self.mem_p2p_q.put(None)
                break

    def cleanup(self) -> None:
        self.h2d.join()
        self.receiver.join()
        if self.mem_handler:
            self.mem_handler.join()
        if self.mem_writer:
            self.mem_writer.join()
        if self.box_writer:
            self.box_writer.join()
        if self.rebuilder:
            self.rebuilder.join()
        if self.graph_builder:
            self.graph_builder.join()
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
        if self.shared is not None:
            self.model.module.load_memory_state(memory_state)
