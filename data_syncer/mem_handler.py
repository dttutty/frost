import logging
import threading
from queue import Queue
from typing import TypeVar

import torch

from schemas.datatypes.data_to_sync import BoxTsNid, MailTsNid, MemTsNid
from schemas.datatypes.task import (
    MemTask,
    P2PTask,
)
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
        mem_task_q: Queue[MemTask],
        mem_p2p_q: Queue[P2PTask | None],
        shared_mem_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        p2p_handler: P2PHandler,
    ):
        super().__init__(name="mem_handler")
        self.shared_mem = shared_mem
        self.shared_mem_ts = shared_mem_ts
        self.h2d_sem = h2d_sem
        self.mem_task_q: Queue[MemTask] = mem_task_q
        self.mem_p2p_q: Queue[P2PTask | None] = mem_p2p_q
        self.stop_ev = stop_ev
        self.p2p_handler = p2p_handler

    def write_back_mem(self, writeback, mem_ts_nid: MemTsNid):
        # write back mem
        if mem_ts_nid is not None and writeback is not None:
            mask = torch.isin(mem_ts_nid.nid, writeback)
            write_back_mem = MemTsNid(
                nid=mem_ts_nid.nid[mask],
                data=mem_ts_nid.data[mask],
                ts=mem_ts_nid.ts[mask],
            )
        id = write_back_mem.nid.cpu().long()
        self.shared_mem[id] = write_back_mem.data.cpu()
        self.shared_mem_ts[id] = write_back_mem.ts.cpu()

    def logic(self):
        mem_task: MemTask = self.mem_task_q.get(block=True)

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
            )
        else:
            self.mem_p2p_q.put(None)
            # skip p2p(), directly put None to p2p_q as sentinel

        self.write_back_mem(mem_plan.writeback, mem_ts_nid)
        self.h2d_sem.release()
