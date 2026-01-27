import logging
import threading
from queue import Queue as Q
from typing import TypeVar

import torch

from schemas.datatypes.data_to_sync import BoxTsNid, MailTsNid, MemTsNid
from schemas.datatypes.task import MemTask, P2PTask
from thread_base_with_stream import StreamThreadBase

from .utils.data_ops import merge_with_priority
from .utils.p2p import P2PHandler


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


@torch.no_grad()
class MemHandler(StreamThreadBase):
    def __init__(
        self,
        shared_mem: torch.Tensor,
        mem_task_q: Q[MemTask | None],
        mem_p2p_q: Q[P2PTask | None],
        shared_mem_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        p2p_handler: P2PHandler,
    ) -> None:
        super().__init__(name="mem_handler")
        self.shared_mem = shared_mem
        self.shared_mem_ts = shared_mem_ts
        self.h2d_sem = h2d_sem
        self.mem_task_q = mem_task_q
        self.mem_p2p_q = mem_p2p_q
        self.stop_ev = stop_ev
        self.p2p_handler = p2p_handler

    def write_back_mem(
        self, writeback: torch.Tensor | None, mem_ts_nid: MemTsNid
    ) -> None:
        # 1. 如果没有需要写回的东西，直接释放信号量（为了不卡死流程）然后返回
        if mem_ts_nid is None or writeback is None:
            # 这里的 release 需要根据你的业务逻辑决定：
            # 如果没做写回操作，是否还需要释放资源给下一轮？通常是需要的。
            self.h2d_sem.release()
            return

        # 2. 正常逻辑
        mask = torch.isin(mem_ts_nid.nid, writeback)

        # 优化：直接切片，不需要创建中间对象 MemTsNid，减少 Python overhead
        # 注意：这里假设 .cpu() 之后数据量不大
        valid_nid = mem_ts_nid.nid[mask].cpu().long()
        valid_data = mem_ts_nid.data[mask].cpu()
        valid_ts = mem_ts_nid.ts[mask].cpu()

        # 3. 写回 CPU 共享内存 (这里的 .cpu() 保证了同步)
        self.shared_mem[valid_nid] = valid_data
        self.shared_mem_ts[valid_nid] = valid_ts

        # 4. 释放信号量
        self.h2d_sem.release()

    def logic(self) -> None:
        while self.stop_ev.is_set() is False:
            mem_task = self.mem_task_q.get(block=True)
            if mem_task is None:
                self.mem_p2p_q.put(None)
                break
            mem_plan = mem_task.mem_plan
            root_new_mem = mem_task.root_new_mem.keep_latest()
            mem_ts_nid = merge_with_priority(root_new_mem, mem_task.old_mem_ts_nid)
            if mem_task.is_last is not None:
                self.p2p_handler.p2p(
                    mem_ts_nid,
                    mem_plan.p2p,
                    mem_plan.rec_cts,
                    mem_plan.send_cts,
                    "latest",
                    target_q=self.mem_p2p_q,
                )
            else:
                self.p2p_handler.skip("mem")
                # skip p2p(), directly put None to p2p_q as sentinel

            self.write_back_mem(mem_plan.writeback, mem_ts_nid)
