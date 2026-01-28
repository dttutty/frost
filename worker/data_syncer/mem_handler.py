import logging
import threading
from queue import Queue as Q
from typing import TypeVar

import torch

from buffer import Buffers
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.task import MemTask, P2PTask
from thread_base_with_stream import StreamThreadBase

from .utils.data_ops import make_p2p_task, merge_with_priority


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


@torch.no_grad()
class MemHandler(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers,
        shared_mem: torch.Tensor,
        in_q: Q[MemTask | None],
        out_q: Q[P2PTask | None],
        shared_mem_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="mem_handler", stop_ev=stop_ev)
        self.buf = buf
        self.shared_mem = shared_mem
        self.shared_mem_ts = shared_mem_ts
        self.h2d_sem = h2d_sem
        self.mem_task_q = in_q
        self.mem_p2p_q = out_q
        self.stop_ev = stop_ev

    def write_back_mem(
        self, writeback: torch.Tensor | None, mem_ts_nid: MemTsNid
    ) -> None:
        if mem_ts_nid is None or writeback is None:
            self.h2d_sem.release()
            return

        mask = torch.isin(mem_ts_nid.nid, writeback)

        valid_nid = mem_ts_nid.nid[mask].cpu().long()
        valid_data = mem_ts_nid.data[mask].cpu()
        valid_ts = mem_ts_nid.ts[mask].cpu()

        self.shared_mem[valid_nid] = valid_data
        self.shared_mem_ts[valid_nid] = valid_ts

        self.h2d_sem.release()

    def logic(self) -> None:
        while self.stop_ev.is_set() is False:
            mem_task = self.mem_task_q.get(block=True)
            if mem_task is None:
                self.mem_p2p_q.put(None)
                break
            mem_plan = mem_task.mem_plan
            root_new_mem = mem_task.root_new_mem.keep_last()
            mem_ts_nid = merge_with_priority(root_new_mem, mem_task.old_mem_ts_nid)
            if mem_task.is_last is not None:
                p2p_task = make_p2p_task(
                    data_td_nid=mem_ts_nid,
                    plan=mem_plan,
                    strategy="last",
                )
                self.mem_p2p_q.put(p2p_task)
            else:
                self.mem_p2p_q.put("skip")
                # skip p2p(), directly put None to p2p_q as sentinel

            self.write_back_mem(mem_plan.writeback, mem_ts_nid)
