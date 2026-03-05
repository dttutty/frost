import logging
import threading
from typing import TypeVar, cast

import torch

from frost.aliases.syncer import MemP2PQ, MemTaskQ, MemWrQ
from frost.datatypes.mem_mail_box._data_ops import keep_last
from frost.datatypes.mem_mail_box.box import BoxTsNid
from frost.datatypes.mem_mail_box.mail import MailTsNid
from frost.datatypes.mem_mail_box.mem import MemTsNid
from frost.protocols.cuda_event_proto import CudaEvent
from frost.runtime._thread_base import StreamThreadBase

from .utils.p2p_task_maker import make_p2p_task

AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


class MemTaskHandler(StreamThreadBase):
    def __init__(
        self,
        mem_task_q: MemTaskQ,
        mem_p2p_q: MemP2PQ,
        stop_ev: threading.Event,
        device: torch.device,
        mem_wr_q: MemWrQ,
        h2d_sem: threading.Semaphore,
    ) -> None:
        super().__init__(name="mem_handler", stop_ev=stop_ev)
        self.device = device
        self.mem_task_q = mem_task_q
        self.mem_wr_q = mem_wr_q
        self.mem_p2p_q = mem_p2p_q
        self.stop_ev = stop_ev
        self.h2d_sem = h2d_sem

    @torch.no_grad()
    def logic(self) -> None:
        while self.stop_ev.is_set() is False:
            mem_task_tuple = self.mem_task_q.get(block=True)
            if mem_task_tuple is None:
                self.mem_wr_q.put(None)  # Sentinel for MemWriter
                break

            mem_task, ready_event = mem_task_tuple
            torch.cuda.current_stream().wait_event(ready_event)
            mem_plan = mem_task.mem_plan
            root_new_mem = keep_last(mem_task.root_new_mem)

            # skip stale memory for speed
            # mem_ts_nid = merge_with_priority(root_new_mem, mem_task.old_mem_ts_nid)
            mem_ts_nid = root_new_mem
            batch_idx = mem_task.batch_idx

            if mem_task.is_last is False:
                p2p_task = make_p2p_task(
                    data_td_nid=mem_ts_nid,
                    plan=mem_plan,
                    device=self.device,
                    batch_idx=batch_idx,
                )

                ready_event_1: CudaEvent = cast(CudaEvent, torch.cuda.Event())
                ready_event_1.record(torch.cuda.current_stream())
                self.mem_p2p_q.put((p2p_task, ready_event_1))
            else:
                self.mem_p2p_q.put("nothing_to_p2p")

            ready_event_2: CudaEvent = cast(CudaEvent, torch.cuda.Event())
            ready_event_2.record(torch.cuda.current_stream())
            if mem_task.mem_plan.writeback is not None:
                to_writeback = mem_task.mem_plan.writeback.data
                self.mem_wr_q.put((to_writeback, batch_idx, mem_ts_nid, ready_event_2))
            else:
                self.mem_wr_q.put((None, batch_idx, mem_ts_nid, ready_event_2))
