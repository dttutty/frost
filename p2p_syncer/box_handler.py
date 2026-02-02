import logging
import threading
from queue import Queue as Q
from typing import Literal, TypeVar

import torch

from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.task import BoxTask, P2PTask
from thread_base_with_stream import StreamThreadBase

from .utils.interaction_ops import generate_interaction_mails
from .utils.p2p_task_maker import make_p2p_task


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")


logger = logging.getLogger(__name__)


class BoxHandler(StreamThreadBase):
    def __init__(
        self,
        box_task_q: Q[tuple[BoxTask, torch.cuda.Event] | None],
        box_p2p_q: Q[tuple[P2PTask, torch.cuda.Event] | Literal["nothing_to_p2p"]],
        stop_ev: threading.Event,
        mail_combine: Literal["last", "mean", "all"],
        merge_src: bool,
        device: torch.device,
        box_writer_q: Q[tuple[torch.Tensor | None, BoxTsNid, torch.cuda.Event] | None],
        h2d_sem: threading.Semaphore,
    ) -> None:
        super().__init__(name="box_handler", stop_ev=stop_ev)

        self.device = device
        self.box_task_q = box_task_q
        self.box_p2p_q = box_p2p_q
        self.stop_ev = stop_ev
        self.mail_combine = mail_combine
        self.merge_src = merge_src
        self.box_writer_q = box_writer_q
        self.h2d_sem = h2d_sem

    def handle_small_box(
        self,
        new_mail: MailTsNid,
        mail_combine: Literal["last", "mean", "all"],
        box_task: BoxTask,
    ) -> tuple[P2PTask, BoxTsNid]:
        box = box_task.old_box
        box_plan = box_task.box_plan

        # handle mail sync
        old_mail = box.to_mail()

        if mail_combine == "last":
            new_mail.keep_last()
        else:
            assert mail_combine == "mean"

            new_mail.keep_mean()

        new_mail = new_mail.merge_with_priority(old_mail)

        box_ts_nid = new_mail.to_box()

        p2p_task = make_p2p_task(
            data_td_nid=box_ts_nid,
            plan=box_plan,
            strategy=mail_combine,
            device=self.device,
            meta=box_task.meta,
        )

        return p2p_task, box_ts_nid

    def handle_large_box(
        self,
        new_mail: MailTsNid,
        mail_combine: Literal["last", "mean", "all"],
        box_task: BoxTask,
    ) -> tuple[None, None]:
        return None, None

        # mail_delivery_tensor = box_task.r2n_mdt
        # assert mail_delivery_tensor is not None
        # box = box_task.old_box
        # box_plan = box_task.box_plan

        # # handle box sync
        # assert not self.merge_src, (
        #     "you are wasting mailbox slots, please make sure boxsize <= n_gpus when merge_src==True"
        # )

        # delivered_mails = new_mail.deliver(mail_delivery_tensor, include_self=True)
        # new_mail = delivered_mails

        # if mail_combine == "last":
        #     if __debug__:
        #         logger.debug("BoxHandler: Keeping last mail and merging with box")
        #     new_mail.keep_last()
        #     # box_ts_nid = box.merge_1_mail(new_mail)
        # elif mail_combine == "mean":
        #     if __debug__:
        #         logger.debug("BoxHandler: Keeping mean mail and merging with box")
        #     new_mail.keep_mean()
        #     # box_ts_nid = box.merge_1_mail(new_mail)
        # elif mail_combine == "all":
        #     if __debug__:
        #         logger.debug("BoxHandler: Merging all mails with box")
        #     # box_ts_nid = box.merge_n_mail(new_mail)

        # if __debug__:
        #     logger.debug(f"BoxHandler: Merged box size={box_ts_nid.nid.numel()}")
        # if __debug__:
        #     logger.debug("BoxHandler: Creating P2P task for large box")
        # p2p_task = make_p2p_task(
        #     data_td_nid=box_ts_nid,
        #     plan=box_plan,
        #     strategy="last",
        #     device=self.device,
        # )
        # if __debug__:
        #     logger.debug("BoxHandler: handle_large_box complete")
        # return p2p_task, box_ts_nid

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0
        if __debug__:
            logger.debug("BoxHandler: Starting logic loop")

        while not self.stop_ev.is_set():
            if __debug__:
                logger.debug(
                    f"BoxHandler: Waiting for box_task from queue (queue size: {self.box_task_q.qsize()})"
                )
            box_task_tuple = self.box_task_q.get(block=True)

            if box_task_tuple is None:
                if __debug__:
                    logger.debug("BoxHandler: received sentinel, exiting logic loop")
                if __debug__:
                    logger.debug(
                        f"BoxHandler: Processed {batch_count} box tasks before exit"
                    )
                self.box_writer_q.put(None)  # Sentinel for BoxWriter
                break

            batch_count += 1
            box_task, ready_event = box_task_tuple

            torch.cuda.current_stream().wait_event(ready_event)

            # handle mail or box sync
            # generate new mails from root_new_mem

            new_mail = generate_interaction_mails(
                box_task.root_new_mem_orig,
                box_task.pos_efeat,
            )

            if box_task.r2n_mdt is None:
                # r2n_mdt是用来发送mail到别的box的，如果没有,说明不用发送

                p2p_task, box_ts_nid = self.handle_small_box(
                    new_mail, self.mail_combine, box_task
                )
            else:
                p2p_task, box_ts_nid = self.handle_large_box(
                    new_mail, self.mail_combine, box_task
                )

            if box_task.is_last is False:
                ready_event: torch.cuda.Event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream())
                self.box_p2p_q.put((p2p_task, ready_event))
            else:
                assert p2p_task is None

                self.box_p2p_q.put("nothing_to_p2p")

            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            if box_task.box_plan.writeback is not None:
                assert box_ts_nid is not None
                self.box_writer_q.put(
                    (box_task.box_plan.writeback.data, box_ts_nid, ready_event)
                )
            else:
                assert box_ts_nid is not None
                self.box_writer_q.put((None, box_ts_nid, ready_event))
