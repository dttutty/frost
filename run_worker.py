# --- 1. 确保在文件顶部有这些导入 ---
import sys
import threading
from typing import Optional

import torch
import torch.distributed as dist

from common.state import WorkerState
from memorys import MemoryMailbox
from schemas.datatypes import DistEnv
from train_dist import Config  # 只保留一次导入
from worker.buffers import alc_feat_buf_no_dedup, alc_feat_buf_w_dedup
from worker.d2h import d2h
from worker.gate import GlobalMicrobatchGate
from worker.h2d import h2d
from worker.trainer import Trainer


class Worker:
    def __init__(
        self,
        cfg: Config,
        nccl_group,
        mm: Optional[MemoryMailbox],
        save_to,
        nfeat,
        efeat,
        env: DistEnv,
        gate: Optional[GlobalMicrobatchGate],
    ):
        """
        构造函数：初始化所有组件和状态。
        """
        # --- 存储配置和环境 ---
        self.cfg = cfg
        self.nccl_group = nccl_group
        self.mm = mm
        self.save_to = save_to
        self.nfeat = nfeat
        self.efeat = efeat
        self.env = env
        self.gate = gate
        self.scfg, self.mcfg, _, self.tcfg = cfg.s, cfg.m, cfg.g, cfg.t
        self.dedup = self.tcfg.dedup
        self.pending_data = None

        # --- 1. 分配 Host 缓冲 ---
        if self.dedup:
            nbuf, ebuf = alc_feat_buf_w_dedup(
                self.scfg, self.tcfg.batch_size, nfeat, efeat
            )
        else:
            nbuf, ebuf = alc_feat_buf_no_dedup(
                self.scfg, self.tcfg.batch_size, nfeat, efeat
            )

        if self.mm is not None:
            self.mm.alc_pin_mem_mailbox_buf(self.dedup, self.scfg, self.tcfg.batch_size)

        # --- 2. 初始化核心组件 ---
        self.stream = torch.cuda.Stream()
        self.comm_lock = threading.Lock()  # 保护 dist 通信

        # H2D 移动器
        self.mover = h2d(
            dedup=self.dedup,
            host_rank=self.env.host_rank,
            stm=self.stream,
            nbuf=nbuf,
            ebuf=ebuf,
            nfeat=self.nfeat,
            efeat=self.efeat,
            mm=self.mm,
            mcfg=self.mcfg,
            merge_src=self.tcfg.merge_src,
            max_inflight=2,
            gate=self.gate,
            comm_lock=self.comm_lock,  # 传入锁
        )

        # D2H 邮件投递器
        self.mail_worker = d2h(
            self.env.local_rank,
            self.mm,
            self.dedup,
            self.gate,
            deliver_to=self.mcfg.deliver_to,
            max_inflight=1,
        )

        # 训练器
        self.trainer = Trainer(
            self.cfg,
            self.nccl_group,
            self.dedup,
            self.efeat,
            self.mm,
            self.mail_worker,
            self.env,
        )

        # --- 3. 初始化状态变量 ---
        self.is_last_train_minibatch_finished = False
        self.state_buf = torch.empty(1, dtype=torch.int32, device="cpu")
        self.first_batch = True
        self.is_running = True  # ???

        print(f"[Rank {self.env.local_rank}] Worker initialized.")

    def run(self):
        """
        运行 Worker 的主循环。
        """
        self.mover.start()  # 启动 H2D 预取线程

        try:
            while self.is_running:
                # 1. 等待 Host 广播命令
                with self.comm_lock:
                    dist.broadcast(self.state_buf, src=self.env.host_rank)
                    cmd = WorkerState(self.state_buf.item())

                print(f"[Rank {self.env.local_rank}] cmd = {cmd.name} ({cmd.value})")

                # 2. 根据命令分发任务
                if cmd is WorkerState.EXIT:
                    print(f"[Rank {self.env.local_rank}] EXIT")
                    self.is_running = False
                elif cmd is WorkerState.TRAIN:
                    self._handle_train()
                elif cmd is WorkerState.EVAL:
                    self._handle_eval()
                elif cmd is WorkerState.SAVE_MODEL:
                    self._handle_save_model()
                elif cmd is WorkerState.LOAD_MODEL:
                    self._handle_load_model()
                elif cmd is WorkerState.REDUCE_LOSS:
                    self._handle_reduce_loss()
                elif cmd is WorkerState.BARRIER:
                    self._handle_barrier()
                elif cmd is WorkerState.CLEANUP:
                    self._handle_cleanup()
                else:
                    pass  # 忽略未知命令

        finally:
            # 确保即使循环异常退出，也尝试清理
            self._handle_cleanup()
            print(f"[Rank {self.env.local_rank}] Run loop finished.")

    # --- 私有方法：处理各种状态 ---

    def _handle_train(self):
        self.mover.submit()  # 提交请求，拉取下一个(N)批次的数据

        if not self.first_batch:
            # 处理上一个(N-1)批次的数据
            prev = self.mover.get_ready()
            assert prev is not None, (
                f"[Rank {self.env.local_rank}] Mover returned None in TRAIN"
            )
            gpudata, done_ev, td = prev

            if td is None:
                # Host 发送了一个 None 作为尾部数据的填充
                print(
                    f"[Rank {self.env.local_rank}] Got None batch in TRAIN, skipping."
                )
            else:
                # 正常的训练批次
                print(f"[Rank {self.env.local_rank}] got TRAIN data, mid={td.mid}")
                self.trainer.train(gpudata, td, done_ev)  # 一个rank可以起来吗?
        else:
            # 第一个批次，仅用于“预热”流水线，不处理
            # (!!!!) 修复：移除了 time.sleep(1) (!!!!)
            print(
                f"[Rank {self.env.local_rank}] First TRAIN batch, warming up pipeline."
            )
            self.first_batch = False

    def _handle_eval(self):
        self.mover.submit()  # 提交请求，拉取下一个(N)批次的数据
        prev = self.mover.get_ready()  # 获取上一个(N-1)批次的数据
        assert prev is not None, (
            f"[Rank {self.env.local_rank}] Mover returned None in EVAL"
        )
        gpudata, done_ev, td = prev

        if td is None:
            # Host 发送了一个 None 作为尾部数据的填充
            # 必须参与 gather，发送 0.0
            print(
                f"[Rank {self.env.local_rank}] Got None batch in EVAL, sending 0.0 metrics."
            )
            dist.gather_object(float(0), None, dst=self.env.host_rank)
            dist.gather_object(float(0), None, dst=self.env.host_rank)
            return

        print(f"[Rank {self.env.local_rank}] got EVAL data, mid={td.mid}")

        if not self.is_last_train_minibatch_finished:
            # 这是 Host 发来的“最后一拍”训练数据
            print(
                f"[Rank {self.env.local_rank}] Processing final TRAIN batch, mid={td.mid}"
            )
            self.trainer.train(gpudata, td, done_ev)
            self.is_last_train_minibatch_finished = True

            # 必须参与 gather，发送 0.0 作为占位
            dist.gather_object(float(0), None, dst=self.env.host_rank)
            dist.gather_object(float(0), None, dst=self.env.host_rank)
        else:
            # 这是真正的评估批次
            ap, auc = self.trainer.eval(
                gpudata, td, done_ev, host_rank=self.env.host_rank
            )
            dist.gather_object(float(ap), None, dst=self.env.host_rank)
            dist.gather_object(float(auc), None, dst=self.env.host_rank)

    def _handle_save_model(self):
        print(f"[Rank {self.env.local_rank}] SAVE_MODEL -> {self.save_to}")
        self.trainer.save(self.save_to)

    def _handle_load_model(self):
        print(f"[Rank {self.env.local_rank}] LOAD_MODEL <- {self.save_to}")
        self.trainer.load(self.save_to)

    def _handle_reduce_loss(self):
        print(f"[Rank {self.env.local_rank}] REDUCE_LOSS")
        dist.gather_object(float(self.trainer.tot_loss), None, dst=self.env.host_rank)
        self.trainer.tot_loss = 0.0

        # (!!!!) 关键 BUG 修复 (!!!!)
        # 训练阶段结束，重置状态，为下一个 epoch 的训练做准备
        self.is_last_train_minibatch_finished = False
        self.first_batch = True

    def _handle_barrier(self):
        print(f"[Rank {self.env.local_rank}] BARRIER")
        dist.barrier()

    def _handle_cleanup(self):
        print(f"[Rank {self.env.local_rank}] CLEANUP")
        try:
            self.mover.stop()
            self.mover.join(timeout=2)
        except Exception as e:
            print(f"[Rank {self.env.local_rank}] Error stopping mover: {e}")
        try:
            if hasattr(self.mail_worker, "stop"):
                self.mail_worker.stop()
            if hasattr(self.mail_worker, "join"):
                self.mail_worker.join(timeout=2)
        except Exception as e:
            print(f"[Rank {self.env.local_rank}] Error stopping mail_worker: {e}")
        torch.cuda.synchronize()


def run_worker(
    cfg: Config,
    nccl_group,
    mm: Optional[MemoryMailbox],
    path_saver,
    nfeat,
    efeat,
    env: DistEnv,
    gate,
):
    """
    初始化并运行 Worker 对象。
    """
    try:
        worker = Worker(cfg, nccl_group, mm, path_saver, nfeat, efeat, env, gate)
        worker.run()
    except Exception as e:
        print(f"[Rank {env.local_rank}] CRITICAL ERROR in run_worker: {e}")
        import traceback

        traceback.print_exc()
    sys.exit(1)  # 非正常退出
