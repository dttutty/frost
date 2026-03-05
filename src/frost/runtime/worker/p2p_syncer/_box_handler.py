import logging
import threading
from typing import Literal, TypeVar, cast

import torch

from frost.aliases.syncer import BoxP2PQ, BoxTaskQ, BoxWrQ
from frost.datatypes.mem_mail_box._data_ops import (
    keep_last,
    keep_mean,
    mail_to_box,
)
from frost.datatypes.mem_mail_box.box import BoxTsNid
from frost.datatypes.mem_mail_box.mail import MailTsNid
from frost.datatypes.mem_mail_box.mem import MemTsNid
from frost.datatypes.tasks import BoxTask, P2PTask
from frost.protocols.cuda_event_proto import CudaEvent
from frost.runtime._thread_base import StreamThreadBase
from frost.runtime.worker.postman import Postman

from .utils.interaction_ops import generate_interaction_mails
from .utils.p2p_task_maker import make_p2p_task

AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")


logger = logging.getLogger(__name__)


class BoxTaskHandler(StreamThreadBase):
    def __init__(
        self,
        box_task_q: BoxTaskQ,
        box_p2p_q: BoxP2PQ,
        stop_ev: threading.Event,
        mail_combine: Literal["last", "mean"],
        device: torch.device,
        box_wr_q: BoxWrQ,
        h2d_sem: threading.Semaphore,
        postman: Postman | None,
    ) -> None:
        super().__init__(name="box_handler", stop_ev=stop_ev)

        self.device = device
        self.box_task_q = box_task_q
        self.box_p2p_q = box_p2p_q
        self.stop_ev = stop_ev
        self.mail_combine = mail_combine
        self.postman = postman
        self.box_wr_q = box_wr_q
        self.h2d_sem = h2d_sem

    def handle_small_box(
        self,
        new_mail: MailTsNid,
        mail_combine: Literal["last", "mean"],
        box_task: BoxTask,
    ) -> tuple[P2PTask, BoxTsNid]:
        box_plan = box_task.box_plan

        if mail_combine == "last":
            new_mail = keep_last(new_mail)
        else:
            assert mail_combine == "mean"
            new_mail = keep_mean(new_mail)

        # disable stale state p2p for speed
        # box = box_task.old_box
        # old_mail = box_to_mail(box)
        # new_mail = new_mail.merge_with_priority(old_mail)

        box_ts_nid = mail_to_box(new_mail)

        # box_ts_nid.data[:] = box_ts_nid.nid.view(-1, 1, 1).to(box_ts_nid.data.dtype)
        p2p_task = make_p2p_task(
            data_td_nid=box_ts_nid,
            plan=box_plan,
            device=self.device,
            batch_idx=box_task.batch_idx,
        )

        return p2p_task, box_ts_nid

    def handle_large_box(
        self,
        new_mail: MailTsNid,
        mail_combine: Literal["last", "mean"],
        box_task: BoxTask,
    ) -> tuple[P2PTask, BoxTsNid]:
        assert self.postman is not None, "Postman is required for handling large boxes"
        graph = box_task.graph
        # disable stale state p2p for speed
        box = box_task.old_box
        box_plan = box_task.box_plan

        delivered_mails = self.postman.deliver(graph, mail_data=new_mail.data)
        new_mail = delivered_mails

        if mail_combine == "last":
            new_mail = keep_last(new_mail)
            box_ts_nid = box.merge_1_mail(new_mail)
        elif mail_combine == "mean":
            new_mail = keep_mean(new_mail)
            box_ts_nid = box.merge_1_mail(new_mail)

        p2p_task = make_p2p_task(
            data_td_nid=box_ts_nid,
            plan=box_plan,
            device=self.device,
            batch_idx=box_task.batch_idx,
        )
        return p2p_task, box_ts_nid

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0

        while not self.stop_ev.is_set():
            box_task_tuple = self.box_task_q.get(block=True)

            if box_task_tuple is None:
                self.box_wr_q.put(None)  # Sentinel for BoxWriter
                break

            batch_count += 1
            box_task, ready_event = box_task_tuple

            torch.cuda.current_stream().wait_event(ready_event)

            new_mail = generate_interaction_mails(
                box_task.root_new_mem,
                box_task.pos_efeat,
            )

            if self.postman is None:
                # r2n_mdt是用来发送mail到别的box的，如果没有,说明不用发送

                p2p_task, box_ts_nid = self.handle_small_box(
                    new_mail, self.mail_combine, box_task
                )
                if box_task.is_last is True:
                    # 最后一个macro-batch不需要p2p了，直接写回就好
                    p2p_task = None
            else:
                p2p_task, box_ts_nid = self.handle_large_box(
                    new_mail, self.mail_combine, box_task
                )
                if box_task.is_last is True:
                    # 最后一个macro-batch不需要p2p了，直接写回就好
                    p2p_task = None

            if box_task.is_last is False:
                ready_event_1 = cast(CudaEvent, torch.cuda.Event())
                ready_event_1.record(torch.cuda.current_stream())
                # torch.cuda.current_stream().synchronize()
                self.box_p2p_q.put((p2p_task, ready_event_1))
            else:
                assert p2p_task is None
                self.box_p2p_q.put("nothing_to_p2p")

            ready_event_2 = cast(CudaEvent, torch.cuda.Event())
            ready_event_2.record(torch.cuda.current_stream())
            if box_task.box_plan.writeback is not None:
                assert box_ts_nid is not None
                self.box_wr_q.put(
                    (
                        box_task.box_plan.writeback.data,
                        box_task.batch_idx,
                        box_ts_nid,
                        ready_event_2,
                    )
                )
            else:
                assert box_ts_nid is not None
                self.box_wr_q.put((None, box_task.batch_idx, box_ts_nid, ready_event_2))
