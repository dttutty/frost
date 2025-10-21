# mailbox_worker.py
from queue import Empty, Queue
from threading import Event, Thread
from typing import Optional

import numpy as np
import torch

from memorys import MemoryMailbox
from schemas.datatypes import MailTask
from worker.gate import GlobalMicrobatchGate  # 与 h2d 一致的相对导入


class d2h:
    """
    将 GPU 端计算得到的 (nid, memory, ts, root, block, mem_edge_feats) 写回 Host 侧 MemoryMailbox，
    并与 GlobalMicrobatchGate 配合推进 next_mail_pos 与 microbatch 完成标记。
    """

    def __init__(
        self,
        local_rank: int,
        mm: Optional[MemoryMailbox],
        dedup: bool,
        gate: Optional[GlobalMicrobatchGate],
        deliver_to: str = "self",  # "self" 或 "neighbors"
        max_inflight: int = 2,
        cpu_copy: bool = False,  # 若 MemoryMailbox 期待 CPU 张量，可置 True
    ):
        self.local_rank = local_rank
        self.mm: Optional[MemoryMailbox] = mm
        self.dedup = dedup
        self.deliver_to = deliver_to
        self.gate = gate
        self.cpu_copy = cpu_copy

        self._q: Queue[Optional[MailTask]] = Queue(maxsize=max_inflight)
        self._stop_ev = Event()
        self._th = Thread(target=self._loop, name="MailboxWorker", daemon=True)
        self._th.start()

    # ============ public api ============
    def enqueue(
        self, task: MailTask, *, block: bool = True, timeout: Optional[float] = None
    ):
        """提交一条邮件写入任务；队列满时根据 block/timeout 背压。"""
        self._q.put(task, block=block, timeout=timeout)

    def flush(self, timeout: Optional[float] = None):
        """
        阻塞直到队列清空。实现：提交一个空任务 + join 等待。
        """
        evt = Event()
        self.enqueue(
            MailTask(
                bid=None,
                mid=None,
                nid=torch.tensor([], dtype=torch.long),
                memory=torch.tensor([], dtype=torch.float32),
                ts=np.asarray([], dtype=np.float32),
                root=np.asarray([], dtype=np.int32),
                block=None,
                mem_edge_feats=None,
                done=evt,
            )
        )
        # 等到“空任务”完成，且把 join 做了，确保之前所有任务已处理
        evt.wait(timeout=timeout)
        self._q.join()

    def stop(self):
        """优雅停止：放哨兵 + join 线程。"""
        self._stop_ev.set()
        try:
            self._q.put_nowait(None)
        except:
            pass
        self._th.join(timeout=2)

    # ============ worker loop ============
    def _loop(self):
        torch.set_grad_enabled(False)
        while not self._stop_ev.is_set():
            try:
                task = self._q.get(timeout=0.1)
            except Empty:
                continue

            if task is None:
                self._q.task_done()
                break

            try:
                if self.mm is not None:
                    # 可选将写入张量转为 CPU，视你的 MemoryMailbox 实现而定
                    nid = (
                        task.nid.cpu()
                        if (task.nid is not None and self.cpu_copy and task.nid.is_cuda)
                        else task.nid
                    )
                    mem = (
                        task.memory.cpu()
                        if (
                            task.memory is not None
                            and self.cpu_copy
                            and task.memory.is_cuda
                        )
                        else task.memory
                    )
                    rts = task.ts  # numpy/CPU
                    root = task.root
                    block = task.block
                    mef = (
                        task.mem_edge_feats.cpu()
                        if (
                            task.mem_edge_feats is not None
                            and self.cpu_copy
                            and task.mem_edge_feats.is_cuda
                        )
                        else task.mem_edge_feats
                    )

                    # 1) 先写 mailbox（mail + memory）
                    #    注意：根据你的 MemoryMailbox 依赖关系，若需要先写 memory 再写 mail，可对调顺序
                    self.mm.write_mail_to_mailbox(nid, mem, root, rts, mef, block)
                    self.mm.write_node_memory(self.dedup, nid, mem, root, rts)

                    # 2) 若按邻居投递，需要配合 gate 推进 next_mail_pos
                    if self.deliver_to == "neighbors" and self.gate is not None:
                        if task.bid is not None:
                            self.gate.mark_mm_updated(
                                task.bid
                            )  # 标记该 batch 中有 microbatch 完成了 mailbox 写入
                            self.gate.wait_can_update_next_mail_pos(task.bid)
                        # 仅让 rank0 推进写指针，避免并发写
                        if self.local_rank == 0:
                            self.mm.update_next_mail_pos()

            finally:
                # 3) 通知 microbatch 完成（无论成功或异常）
                if self.gate is not None and task.mid is not None:
                    self.gate.mark_finished(task.mid)

                if task.done is not None:
                    try:
                        task.done.set()
                    except Exception:
                        pass

                # 完成队列项
                self._q.task_done()
