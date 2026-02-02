import logging
import threading
from queue import Queue as Q
from typing import Literal, TypeVar

import torch

from schemas.datatypes.mem_mail_box._data_ops import merge_with_priority
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.task import MemTask, P2PTask
from thread_base_with_stream import StreamThreadBase

from .utils.p2p_task_maker import make_p2p_task


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


class MemHandler(StreamThreadBase):
    def __init__(
        self,
        mem_task_q: Q[tuple[MemTask, torch.cuda.Event] | None],
        mem_p2p_q: Q[tuple[P2PTask, torch.cuda.Event] | Literal["nothing_to_p2p"]],
        stop_ev: threading.Event,
        device: torch.device,
        mem_writer_q: Q[tuple[torch.Tensor | None, MemTsNid, torch.cuda.Event] | None],
        h2d_sem: threading.Semaphore,
    ) -> None:
        super().__init__(name="mem_handler", stop_ev=stop_ev)
        if __debug__:
            logger.debug("MemHandler: Initializing")
        self.device = device
        self.mem_task_q = mem_task_q
        self.mem_writer_q = mem_writer_q
        self.mem_p2p_q = mem_p2p_q
        self.stop_ev = stop_ev
        self.h2d_sem = h2d_sem

        if __debug__:
            logger.debug("MemHandler: Initialized")

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0
        if __debug__:
            logger.debug("MemHandler: Starting logic loop")

        while self.stop_ev.is_set() is False:
            mem_task_tuple = self.mem_task_q.get(block=True)

            if mem_task_tuple is None:
                self.mem_writer_q.put(None)  # Sentinel for MemWriter
                break

            batch_count += 1
            mem_task, ready_event = mem_task_tuple

            torch.cuda.current_stream().wait_event(ready_event)

            mem_plan = mem_task.mem_plan

            root_new_mem = mem_task.root_new_mem.keep_last()

            mem_ts_nid = merge_with_priority(root_new_mem, mem_task.old_mem_ts_nid)
            if __debug__:
                logger.debug(
                    f"MemHandler: Batch #{batch_count}, after merge, mem_ts_nid size={mem_ts_nid.nid.numel()}"
                )

            if mem_task.is_last is False:
                p2p_task = make_p2p_task(
                    data_td_nid=mem_ts_nid,
                    plan=mem_plan,
                    strategy="last",
                    device=self.device,
                    meta=mem_task.meta,
                )

                ready_event: torch.cuda.Event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream())
                self.mem_p2p_q.put((p2p_task, ready_event))
            else:
                self.mem_p2p_q.put("nothing_to_p2p")
                # skip p2p(), directly put None to p2p_q as sentinel

            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            if mem_task.mem_plan.writeback is not None:
                self.mem_writer_q.put(
                    (mem_task.mem_plan.writeback.data, mem_ts_nid, ready_event)
                )
            else:
                self.mem_writer_q.put((None, mem_ts_nid, ready_event))
