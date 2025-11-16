# --- 1. 确保在文件顶部有这些导入 ---
import math
import queue
import threading
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score

from memorys import MemoryMailbox
from schemas.datatypes import DistEnv
from train_dist import Config  # 只保留一次导入
from worker.buffers import alc_feat_buf_no_dedup, alc_feat_buf_w_dedup
from worker.d2h import d2h
from worker.gate import GlobalMicrobatchGate
from worker.h2d import h2d
from worker.trainer import Trainer


CTRL_QUEUE_MAX = 8


def worker_scatter_receiver(
    env, rx_q: queue.Queue, comm_lock: threading.Lock, stop_event: threading.Event
):
    host_rank = env.host_rank
    while not stop_event.is_set():
        try:
            out = [None]
            with comm_lock:
                dist.scatter_object_list(out, None, src=host_rank)
            msg_list = out[0]
            # 防止关不掉：put 带超时
            rx_q.put(msg_list, timeout=0.5)
        except queue.Full:
            raise RuntimeError(f"[Rank {env.local_rank}] RX queue full")
        except Exception as e:
            # 根据需要打印或决定是否 break
            print(f"[Rank {env.local_rank}] RX thread error: {e}")


class StateWatcher:
    def __init__(self):
        self._state = None
        self._callbacks = []
        self._lock = threading.Lock()

    @property
    def state(self):
        with self._lock:
            return self._state

    @state.setter
    def state(self, value):
        with self._lock:
            old = self._state
            if old == value:
                return
            self._state = value
            callbacks = list(self._callbacks)  # 拷贝避免迭代时被修改
        # 在锁外调用，避免回调里阻塞 setter
        for cb in callbacks:
            cb(value)

    def on_change(self, fn):
        with self._lock:
            self._callbacks.append(fn)


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
        self.w = StateWatcher()
        self._stop_event = threading.Event()

        self.s_comp = torch.cuda.Stream()  # 计算流
        self.s_comm = torch.cuda.Stream()  # 通信流

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

        self.rx_q = queue.Queue(maxsize=2)
        self.ready_q = queue.Queue(maxsize=2)

        self.q_2 = queue.Queue(maxsize=1)
        self.q_2.put(None)

        self.comm_lock = threading.Lock()

        self.rx_thread = threading.Thread(
            target=worker_scatter_receiver,
            args=(self.env, self.rx_q, self.comm_lock, self._stop_event),
            daemon=True,
        )
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
            rx_q=self.rx_q,
            ready_q=self.ready_q,
            max_inflight=2,
            gate=self.gate,
            comm_lock=self.comm_lock,  # 传入锁
        )

        # D2H 邮件投递器
        self.d2h_worker = d2h(
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
            self.d2h_worker,
            self.env,
        )

        # --- 3. 初始化状态变量 ---
        self.state_buf = torch.empty(1, dtype=torch.int32, device="cpu")
        self.state = None

        self.best_eval_ap = 0.0
        self.best_eval_auc = 0.0
        self.best_eval_epoch = -1
        self.best_model_path = save_to / "best_model.pt"
        self.test_ap = 0.0
        self.test_auc = 0.0
        self.current_epoch = 0

        print(f"[Rank {self.env.local_rank}] Worker initialized.")

        # train_total_mbatchs_for_this_rank = inifinity

        self.total_epoch = math.inf
        self.done_epoch = 0

        self.first_eval_done = False

    def compute_ap_auc_ddp(self, y_true_local, y_score_local):
        # 1) 各 rank 上传各自列表
        obj_local = (y_true_local, y_score_local)  # 两个 list[float/int]
        gathered = [None for _ in range(self.env.ngpus)]
        dist.all_gather_object(gathered, obj_local, group=self.nccl_group)

        # 2) 只在 rank0 计算
        if self.env.local_rank == 0:
            y_true_all = np.concatenate([np.asarray(t[0]) for t in gathered], axis=0)
            y_score_all = np.concatenate([np.asarray(t[1]) for t in gathered], axis=0)
            ap = average_precision_score(y_true_all, y_score_all)
            auc = roc_auc_score(y_true_all, y_score_all)
        else:
            ap, auc = 0.0, 0.0

        # 3) 广播结果
        ap_t = torch.tensor(ap, device="cuda", dtype=torch.float32)
        auc_t = torch.tensor(auc, device="cuda", dtype=torch.float32)
        dist.broadcast(ap_t, src=0, group=self.nccl_group)
        dist.broadcast(auc_t, src=0, group=self.nccl_group)
        return ap_t.item(), auc_t.item()

    def train_loop(self):
        total_mbatchs = math.inf
        done_mbatchs = 0
        loss_sum = 0.0
        count_sum = 0
        while total_mbatchs > done_mbatchs:
            prev = self.ready_q.get(block=True)
            prev_handler = self.q_2.get()
            prev = comb(
                prev_handler,
            )
            gpudata, done_ev, td = prev
            if self.total_epoch == math.inf:
                self.total_epoch = td.total_epochs
            if total_mbatchs == math.inf:
                total_mbatchs = td.total_mbatchs_for_this_rank
            loss, count = self.trainer.train(gpudata, td, done_ev)
            loss_sum += float(loss) * int(count)
            count_sum += int(count)
            done_mbatchs += 1
        loss_sum_t = torch.tensor(loss_sum, device="cuda")
        count_sum_t = torch.tensor(count_sum, device="cuda", dtype=torch.long)
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_sum_t, op=dist.ReduceOp.SUM)
        if self.env.local_rank == 0:
            print(
                f"[Epoch {self.done_epoch}] Train loss: {(loss_sum_t / count_sum_t).item()}"
            )
        self.val_loop()

    def gather_all_tensors(self, tensor: torch.Tensor, group) -> torch.Tensor:
        gather_list = [torch.zeros_like(tensor) for _ in range(self.env.ngpus)]
        dist.all_gather(gather_list, tensor, group)
        return torch.cat(gather_list, dim=0)

    def record_this(self, eval_ap, eval_auc, done_epoch):
        if eval_ap > self.best_eval_ap:
            self.best_eval_ap = eval_ap
            self.best_eval_auc = eval_auc
            self.best_eval_epoch = done_epoch
            return True
        else:
            return False

    def val_loop(self):
        # finish training using last memory state
        total_mbatchs = int("inf")
        done_mbatchs = 0
        y_true_local = []
        y_score_local = []
        prev = self.mover.get_ready()
        gpudata, done_ev, td = prev
        total_mbatchs = td.total_mbatchs_for_this_rank
        self.trainer.train(gpudata, td, done_ev)
        while total_mbatchs > done_mbatchs:
            prev = self.mover.get_ready()
            gpudata, done_ev, td = prev
            y_true, y_score = self.trainer.eval(gpudata, td, done_ev)
            y_true_local += y_true
            y_score_local += y_score
            done_mbatchs += 1
        ap, auc = self.compute_ap_auc_ddp(y_true_local, y_score_local)
        self.done_epoch += 1
        if self.env.local_rank == 0:
            is_best = self.record_this(ap, auc, self.done_epoch)
            print(
                f"Epoch {self.done_epoch} VAL AP: {ap}, AUC: {auc}, is_best: {is_best}"
            )
            if is_best:
                self.trainer.save(self.best_model_path)
        dist.barrier(group=self.nccl_group)
        if self.done_epoch < self.total_epoch:
            self.trainer.reset()
            self.train_loop()
        else:
            self.test_loop()

    def test_loop(self):
        total_mbatchs = int("inf")
        done_mbatchs = 0
        y_true_local = []
        y_score_local = []

        while total_mbatchs > done_mbatchs:
            prev = self.mover.get_ready()
            gpudata, done_ev, td = prev
            y_true, y_score = self.trainer.eval(gpudata, td, done_ev)
            y_true_local += y_true
            y_score_local += y_score
            done_mbatchs += 1
        y_true_all = self.gather_all_tensors(y_true_local, self.nccl_group)
        y_score_all = self.gather_all_tensors(y_score_local, self.nccl_group)
        ap = average_precision_score(y_true_all, y_score_all)
        auc = roc_auc_score(y_true_all, y_score_all)
        if self.env.local_rank == 0:
            print(f"TEST AP: {ap}, AUC: {auc}, best epoch: {self.best_eval_epoch}")
        self._handle_cleanup()

    def run(self):
        """
        运行 Worker 的主循环。
        """
        self.mover.start()  # 启动 H2D 预取线程
        self.rx_thread.start()  # 启动接收线程
        self.train_loop()

    def _handle_cleanup(self):
        print(f"[Rank {self.env.local_rank}] CLEANUP")
        self._stop_event.set()
        try:
            if self.rx_thread.is_alive():
                self.rx_thread.join(timeout=2)
        except Exception as e:
            print(f"[Rank {self.env.local_rank}] Error joining rx_thread: {e}")
        try:
            self.mover.stop()
            self.mover.join(timeout=2)
        except Exception as e:
            print(f"[Rank {self.env.local_rank}] Error stopping mover: {e}")
        try:
            if hasattr(self.d2h_worker, "stop"):
                self.d2h_worker.stop()
            if hasattr(self.d2h_worker, "join"):
                self.d2h_worker.join(timeout=2)
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
    worker = Worker(cfg, nccl_group, mm, path_saver, nfeat, efeat, env, gate)
    worker.run()
