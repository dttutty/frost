import logging
import threading
from queue import Queue as Q
from typing import Literal, TypeVar

import torch

from buffer import Buffers, DyBuf
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.task import BoxTask, P2PTask
from thread_base_with_stream import StreamThreadBase

from .utils.data_ops import make_p2p_task
from .utils.interaction_ops import generate_interaction_mails


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")


logger = logging.getLogger(__name__)


@torch.no_grad()
class BoxHandler(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers,
        shared_box: torch.Tensor,
        box_task_q: Q[tuple[BoxTask, torch.cuda.Event] | None],
        box_p2p_q: Q[P2PTask | Literal["nothing_to_p2p"]],
        shared_box_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        mail_combine: Literal["last", "mean", "all"],
        merge_src: bool,
    ) -> None:
        super().__init__(name="box_handler", stop_ev=stop_ev)
        self.buf = buf
        self.shared_box = shared_box
        self.shared_box_ts = shared_box_ts
        self.h2d_sem = h2d_sem
        self.box_task_q = box_task_q
        self.box_p2p_q = box_p2p_q
        self.stop_ev = stop_ev
        self.mail_combine = mail_combine
        self.merge_src = merge_src

    def _write_back_via_pinned(
        self,
        valid_nid_gpu: torch.Tensor,
        valid_data_gpu: torch.Tensor,
        valid_ts_gpu: torch.Tensor,
        pinned_buf: DyBuf,
    ) -> None:
        """D2H Pinned Memory Optimization"""
        n = valid_nid_gpu.numel()
        if n > pinned_buf.data.shape[0]:
            # Fallback
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_box[valid_nid] = valid_data_gpu.cpu()
            self.shared_box_ts[valid_nid] = valid_ts_gpu.cpu()
            return

        buf_data = pinned_buf.data[:n]
        buf_nid = pinned_buf.nid[:n]
        buf_ts = pinned_buf.ts[:n]

        # Async D2H
        valid_data_gpu.to(buf_data, non_blocking=True)
        valid_ts_gpu.to(buf_ts.squeeze(1), non_blocking=True)
        valid_nid_gpu.to(dtype=buf_nid.dtype).to(buf_nid.squeeze(1), non_blocking=True)

        # Sync
        torch.cuda.current_stream().synchronize()

        # Copy to Shared
        idx = buf_nid.view(-1).long()
        self.shared_box[idx] = buf_data
        self.shared_box_ts[idx] = buf_ts.view(-1)

    def write_back_box(
        self,
        writeback: torch.Tensor | None,
        box_ts_nid: BoxTsNid,
    ) -> None:
        # [Bugfix] 应该是 or None 才 return, 之前是 not None 就 return
        if box_ts_nid is None or writeback is None:
            return

        # 1. Move writeback to GPU for mask calculation
        if writeback.device.type != "cuda":
            writeback = writeback.cuda()

        # 2. Filter on GPU
        mask = torch.isin(box_ts_nid.nid, writeback)

        valid_nid_gpu = box_ts_nid.nid[mask]
        valid_data_gpu = box_ts_nid.data[mask]
        valid_ts_gpu = box_ts_nid.ts[mask]

        # 3. Use Pinned Memory D2H
        if self.buf.box_d2h is not None:
            self._write_back_via_pinned(
                valid_nid_gpu, valid_data_gpu, valid_ts_gpu, self.buf.box_d2h
            )
        else:
            # Fallback
            id = valid_nid_gpu.cpu().long()
            self.shared_box[id] = valid_data_gpu.cpu()
            self.shared_box_ts[id] = valid_ts_gpu.cpu()

    def handle_small_box(
        self,
        new_mail: MailTsNid,
        mail_combine: Literal["last", "mean", "all"],
        box_task: BoxTask,
    ) -> tuple[P2PTask, BoxTsNid]:
        box = box_task.old_box
        box_plan = box_task.box_plan
        macro_bid = box_task.macro_bid
        # handle mail sync
        old_mail = box.to_mail()

        if mail_combine == "last":
            new_mail.keep_last()
        else:
            assert mail_combine == "mean"
            new_mail.keep_mean()
        new_mail = new_mail.merge_with_priority(old_mail)
        logger.debug("DataSyncer: merged new mail with old mail", macro_bid)

        p2p_task = make_p2p_task(
            data_td_nid=new_mail,
            plan=box_plan,
            strategy=mail_combine,
        )

        box_ts_nid = new_mail.to_box()
        return p2p_task, box_ts_nid

    def handle_large_box(
        self,
        new_mail: MailTsNid,
        mail_combine: Literal["last", "mean", "all"],
        box_task: BoxTask,
    ) -> tuple[P2PTask, BoxTsNid]:
        mail_delivery_tensor = box_task.r2n_mdt
        assert mail_delivery_tensor is not None
        box = box_task.old_box
        box_plan = box_task.box_plan
        # handle box sync
        assert not self.merge_src, (
            "you are wasting mailbox slots, please make sure boxsize <= n_gpus when merge_src==True"
        )
        delivered_mails = new_mail.deliver(mail_delivery_tensor, include_self=True)
        new_mail = delivered_mails

        if mail_combine == "last":
            new_mail.keep_last()
            box_ts_nid = box.merge_1_mail(new_mail)
        elif mail_combine == "mean":
            new_mail.keep_mean()
            box_ts_nid = box.merge_1_mail(new_mail)
        elif mail_combine == "all":
            box_ts_nid = box.merge_n_mail(new_mail)
        p2p_task = make_p2p_task(
            data_td_nid=box_ts_nid,
            plan=box_plan,
            strategy="last",
        )
        return p2p_task, box_ts_nid

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            box_task_tuple = self.box_task_q.get(block=True)
            if box_task_tuple is None:
                logger.debug("BoxHandler: received sentinel, exiting logic loop")
                break

            box_task, ready_event = box_task_tuple
            torch.cuda.current_stream().wait_event(ready_event)

            macro_bid = box_task.macro_bid

            # handle mail or box sync
            # generate new mails from root_new_mem
            logger.debug("DataSyncer: generating new mails", macro_bid)
            new_mail = generate_interaction_mails(
                box_task.root_new_mem_orig,
                box_task.pos_efeat,
            )
            logger.debug("DataSyncer: start p2p for mail/box", macro_bid)
            if box_task.r2n_mdt is None:
                # r2n_mdt是用来发送mail到别的box的，如果没有,说明不用发送
                p2p_task, box_ts_nid = self.handle_small_box(
                    new_mail, self.mail_combine, box_task
                )
            else:
                p2p_task, box_ts_nid = self.handle_large_box(
                    new_mail, self.mail_combine, box_task
                )

            if box_task.is_last is not None:
                self.box_p2p_q.put(p2p_task)
            else:
                assert p2p_task is None
                self.box_p2p_q.put("nothing_to_p2p")

            logger.debug("DataSyncer: finished p2p processing", macro_bid)

            if box_task.box_plan.writeback is not None:
                self.write_back_box(
                    box_task.box_plan.writeback.data,
                    box_ts_nid,
                )
            logger.debug("DataSyncer: finished write back", macro_bid)
            self.h2d_sem.release()
