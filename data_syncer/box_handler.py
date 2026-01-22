import threading
from queue import Queue
from typing import TypeVar

import torch

from schemas.datatypes.data_to_sync import BoxTsNid, MailTsNid, MemTsNid
from schemas.datatypes.task import (
    BoxTask,
    P2PTask,
)

from .utils.data_ops import keep_latest, keep_mean, merge_with_priority
from .utils.interaction_ops import generate_interaction_mails
from .utils.p2p import P2PHandler


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")


@torch.no_grad()
class BoxHandler(threading.Thread):
    def __init__(
        self,
        shared_box: torch.Tensor,
        box_task_q: Queue[BoxTask],
        box_p2p_q: Queue[P2PTask | None],
        shared_box_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        p2p_handler: P2PHandler,
        mail_combine: str,
        boxsize: int,
        deliver_to: str,
        merge_src: bool,
    ):
        super().__init__(daemon=True)
        self.shared_box = shared_box
        self.shared_box_ts = shared_box_ts
        self.h2d_sem = h2d_sem
        self.stream = torch.cuda.Stream()
        self.box_task_q = box_task_q
        self.box_p2p_q = box_p2p_q
        self.stop_ev = stop_ev
        self.p2p_handler = p2p_handler
        self.mail_combine = mail_combine
        self.boxsize = boxsize
        self.deliver_to = deliver_to
        self.merge_src = merge_src

    def write_back_box(self, writeback, box_ts_nid: BoxTsNid):
        if box_ts_nid is not None and writeback is not None:
            return
        mask = torch.isin(box_ts_nid.nid, writeback)
        write_back_box = BoxTsNid(
            nid=box_ts_nid.nid[mask],
            data=box_ts_nid.data[mask],
            ts=box_ts_nid.ts[mask],
        )

        id = write_back_box.nid.cpu().long()

        self.shared_box[id] = write_back_box.data.cpu()
        self.shared_box_ts[id] = write_back_box.ts.cpu()

    def handle_small_box(
        self, box, mail_combine, is_last, box_plan, macro_bid, new_mail
    ):
        # handle mail sync
        # if deliver_to is 'self', only deliver to self
        assert self.deliver_to == "self"
        old_mail = box.to_mail()

        if mail_combine == "last":
            keep_latest(new_mail)
        else:
            assert mail_combine == "mean"
            keep_mean(new_mail)
        new_mail = merge_with_priority(new_mail, old_mail)
        print(
            "DataSyncer: merged new mail with old mail",
            macro_bid,
        )

        if is_last is not None:
            self.p2p_handler.p2p(
                new_mail,
                box_plan.p2p,
                box_plan.rec_cts,
                box_plan.send_cts,
                self.mail_combine,
            )
        else:
            pass  # this case is handler by mem_handler, mem_handler will put None to p2p_q

        box_ts_nid = new_mail.to_box()
        return box_ts_nid

    def handle_large_box(
        self, box, mail_delivery_tensor, new_mail, mail_combine, is_last, box_plan
    ):
        # handle box sync
        # if deliver_to is 'neighbors', deliver to neighbors using deliver()
        assert self.boxsize > 1
        assert self.deliver_to == "neighbors"
        assert mail_delivery_tensor is not None
        assert not self.merge_src, (
            "you are wasting mailbox slots, please make sure boxsize <= n_gpus when merge_src==True"
        )
        delivered_mails = new_mail.deliver(mail_delivery_tensor, include_self=True)
        new_mail = delivered_mails
        mail_combine = self.mail_combine

        if mail_combine == "last":
            keep_latest(new_mail)
            box_ts_nid = box.merge_1_mail(new_mail)
        elif mail_combine == "mean":
            keep_mean(new_mail)
            box_ts_nid = box.merge_1_mail(new_mail)
        elif mail_combine == "all":
            box_ts_nid = box.merge_n_mail(new_mail)
        if is_last is not None:
            self.p2p_handler.p2p(
                box_ts_nid,
                box_plan.p2p,
                box_plan.rec_cts,
                box_plan.send_cts,
                "latest",
            )
        else:
            pass  # this case is handler by mem_handler, mem_handler will put None to p2p_q
        return box_ts_nid

    def logic(self):
        box_task: BoxTask = self.box_task_q.get(block=True)
        (
            root_new_mem_orig,
            is_last,
            macro_bid,
            box,
            pos_efeat,
            mail_delivery_tensor,
            box_plan,
        ) = (
            box_task.root_new_mem_orig,
            box_task.is_last,
            box_task.macro_bid,
            box_task.old_box,
            box_task.pos_efeat,
            box_task.mail_delivery_tensor,
            box_task.box_plan,
        )

        # handle mail or box sync
        # generate new mails from root_new_mem
        print("DataSyncer: generating new mails", macro_bid)
        new_mail = generate_interaction_mails(
            root_new_mem_orig,
            pos_efeat,
        )
        mail_combine = self.mail_combine
        print("DataSyncer: start p2p for mail/box", macro_bid)
        if self.boxsize == 1:
            box_ts_nid = self.handle_small_box(
                box, mail_combine, is_last, box_plan, macro_bid, new_mail
            )
        else:
            box_ts_nid = self.handle_large_box(
                box, mail_delivery_tensor, new_mail, mail_combine, is_last, box_plan
            )

        print("DataSyncer: finished p2p processing", macro_bid)

        self.write_back_box(
            box_plan.writeback,
            box_ts_nid,
        )
        print("DataSyncer: finished write back", macro_bid)
        self.h2d_sem.release()

    @torch.no_grad()
    def box_handler(self):
        with torch.cuda.stream(self.stream):
            while self.stop_ev.is_set() is False:
                self.logic()
