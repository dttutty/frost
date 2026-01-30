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
        device: torch.device,
    ) -> None:
        super().__init__(name="box_handler", stop_ev=stop_ev)
        logger.debug(
            f"BoxHandler: Initializing, mail_combine={mail_combine}, merge_src={merge_src}"
        )
        self.buf = buf
        self.device = device
        self.shared_box = shared_box
        self.shared_box_ts = shared_box_ts
        self.h2d_sem = h2d_sem
        self.box_task_q = box_task_q
        self.box_p2p_q = box_p2p_q
        self.stop_ev = stop_ev
        self.mail_combine = mail_combine
        self.merge_src = merge_src
        logger.debug(
            f"BoxHandler: shared_box shape={shared_box.shape}, shared_box_ts shape={shared_box_ts.shape}"
        )
        logger.debug(
            f"BoxHandler: box_d2h buffer={'enabled' if buf.box_d2h is not None else 'disabled'}"
        )
        logger.debug("BoxHandler: Initialized")

    @torch.no_grad()
    def _write_back_via_pinned(
        self,
        valid_nid_gpu: torch.Tensor,
        valid_data_gpu: torch.Tensor,
        valid_ts_gpu: torch.Tensor,
        pinned_buf: DyBuf,
    ) -> None:
        """D2H Pinned Memory Optimization"""
        n = valid_nid_gpu.numel()
        if n == 0:
            logger.debug("BoxHandler: No valid entries to write back, returning")
            return
        logger.debug(
            f"BoxHandler: _write_back_via_pinned called, n={n}, buffer capacity={pinned_buf.data.shape[0]}"
        )

        if n > pinned_buf.data.shape[0]:
            logger.debug(
                f"BoxHandler: Buffer insufficient (need {n}, have {pinned_buf.data.shape[0]}), using fallback path"
            )
            # Fallback
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_box[valid_nid] = valid_data_gpu.cpu()
            self.shared_box_ts[valid_nid] = valid_ts_gpu.cpu()
            logger.debug("BoxHandler: Fallback write-back complete")
            return

        logger.debug("BoxHandler: Using pinned buffer for D2H transfer")
        buf_data = pinned_buf.data[:n]
        buf_nid = pinned_buf.nid[:n]
        buf_ts = pinned_buf.ts[:n]

        # Async D2H
        logger.debug("BoxHandler: Starting async D2H transfer")
        valid_data_gpu.to(buf_data, non_blocking=False)
        valid_ts_gpu.to(buf_ts.squeeze(1), non_blocking=False)
        valid_nid_gpu.to(dtype=buf_nid.dtype).to(buf_nid, non_blocking=False)

        # Sync
        logger.debug("BoxHandler: Synchronizing CUDA stream")
        torch.cuda.current_stream().synchronize()

        # Copy to Shared
        logger.debug("BoxHandler: Copying to shared memory")
        idx = buf_nid.long()
        self.shared_box[idx] = buf_data
        self.shared_box_ts[idx] = buf_ts.view(-1)
        logger.debug(
            f"BoxHandler: Write-back complete via pinned buffer, wrote {n} entries"
        )

    def write_back_box(
        self,
        writeback: torch.Tensor | None,
        box_ts_nid: BoxTsNid,
    ) -> None:
        logger.debug(
            f"BoxHandler: write_back_box called, writeback={'provided' if writeback is not None else 'none'}, box_ts_nid={'provided' if box_ts_nid is not None else 'none'}"
        )

        # [Bugfix] 应该是 or None 才 return, 之前是 not None 就 return
        if box_ts_nid is None or writeback is None:
            logger.debug(
                "BoxHandler: No write-back needed (box_ts_nid or writeback is None)"
            )
            return

        logger.debug(
            f"BoxHandler: writeback size={writeback.numel()}, box_ts_nid size={box_ts_nid.nid.numel()}"
        )

        # 1. Move writeback to GPU for mask calculation
        if writeback.device.type != "cuda":
            logger.debug("BoxHandler: Moving writeback tensor to GPU")
            writeback = writeback.cuda()

        # 2. Filter on GPU
        logger.debug("BoxHandler: Filtering valid entries on GPU")
        mask = torch.isin(box_ts_nid.nid, writeback)
        num_valid = mask.sum().item()
        logger.debug(f"BoxHandler: Found {num_valid} valid entries to write back")

        valid_nid_gpu = box_ts_nid.nid[mask]
        valid_data_gpu = box_ts_nid.data[mask]
        valid_ts_gpu = box_ts_nid.ts[mask]

        # 3. Use Pinned Memory D2H
        if self.buf.box_d2h is not None:
            logger.debug("BoxHandler: Using pinned memory path for write-back")
            self._write_back_via_pinned(
                valid_nid_gpu, valid_data_gpu, valid_ts_gpu, self.buf.box_d2h
            )
        else:
            logger.debug("BoxHandler: Using regular (blocking) path for write-back")
            # Fallback
            id = valid_nid_gpu.cpu().long()
            self.shared_box[id] = valid_data_gpu.cpu()
            self.shared_box_ts[id] = valid_ts_gpu.cpu()
            logger.debug(
                f"BoxHandler: Regular write-back complete, wrote {num_valid} entries"
            )

    def handle_small_box(
        self,
        new_mail: MailTsNid,
        mail_combine: Literal["last", "mean", "all"],
        box_task: BoxTask,
    ) -> tuple[P2PTask, BoxTsNid]:
        logger.debug(
            f"BoxHandler: handle_small_box called, macro_bid={box_task.macro_bid}, mail_combine={mail_combine}"
        )
        box = box_task.old_box
        box_plan = box_task.box_plan
        macro_bid = box_task.macro_bid

        # handle mail sync
        old_mail = box.to_mail()
        logger.debug(
            f"BoxHandler: old_mail size={old_mail.nid.numel()}, new_mail size={new_mail.nid.numel()}"
        )

        if mail_combine == "last":
            logger.debug("BoxHandler: Keeping last mail only")
            new_mail.keep_last()
        else:
            assert mail_combine == "mean"
            logger.debug("BoxHandler: Keeping mean of mails")
            new_mail.keep_mean()

        logger.debug("BoxHandler: Merging new mail with old mail")
        new_mail = new_mail.merge_with_priority(old_mail)
        logger.debug(
            f"BoxHandler: merged new mail with old mail, macro_bid={macro_bid}, merged size={new_mail.nid.numel()}"
        )

        box_ts_nid = new_mail.to_box()

        logger.debug("BoxHandler: Creating P2P task")
        p2p_task = make_p2p_task(
            data_td_nid=box_ts_nid,
            plan=box_plan,
            strategy=mail_combine,
            device=self.device,
        )

        logger.debug(
            f"BoxHandler: handle_small_box complete, box_ts_nid size={box_ts_nid.nid.numel()}"
        )
        return p2p_task, box_ts_nid

    def handle_large_box(
        self,
        new_mail: MailTsNid,
        mail_combine: Literal["last", "mean", "all"],
        box_task: BoxTask,
    ) -> tuple[P2PTask, BoxTsNid]:
        logger.debug(
            f"BoxHandler: handle_large_box called, macro_bid={box_task.macro_bid}, mail_combine={mail_combine}"
        )
        mail_delivery_tensor = box_task.r2n_mdt
        assert mail_delivery_tensor is not None
        box = box_task.old_box
        box_plan = box_task.box_plan

        # handle box sync
        assert not self.merge_src, (
            "you are wasting mailbox slots, please make sure boxsize <= n_gpus when merge_src==True"
        )

        logger.debug(
            f"BoxHandler: Delivering mails, new_mail size={new_mail.nid.numel()}"
        )
        delivered_mails = new_mail.deliver(mail_delivery_tensor, include_self=True)
        new_mail = delivered_mails
        logger.debug(f"BoxHandler: Delivered mails size={new_mail.nid.numel()}")

        if mail_combine == "last":
            logger.debug("BoxHandler: Keeping last mail and merging with box")
            new_mail.keep_last()
            box_ts_nid = box.merge_1_mail(new_mail)
        elif mail_combine == "mean":
            logger.debug("BoxHandler: Keeping mean mail and merging with box")
            new_mail.keep_mean()
            box_ts_nid = box.merge_1_mail(new_mail)
        elif mail_combine == "all":
            logger.debug("BoxHandler: Merging all mails with box")
            box_ts_nid = box.merge_n_mail(new_mail)

        logger.debug(f"BoxHandler: Merged box size={box_ts_nid.nid.numel()}")
        logger.debug("BoxHandler: Creating P2P task for large box")
        p2p_task = make_p2p_task(
            data_td_nid=box_ts_nid,
            plan=box_plan,
            strategy="last",
            device=self.device,
        )
        logger.debug("BoxHandler: handle_large_box complete")
        return p2p_task, box_ts_nid

    def logic(self) -> None:
        batch_count = 0
        logger.debug("BoxHandler: Starting logic loop")

        while not self.stop_ev.is_set():
            logger.debug(
                f"BoxHandler: Waiting for box_task from queue (queue size: {self.box_task_q.qsize()})"
            )
            box_task_tuple = self.box_task_q.get(block=True)

            if box_task_tuple is None:
                logger.debug("BoxHandler: received sentinel, exiting logic loop")
                logger.debug(
                    f"BoxHandler: Processed {batch_count} box tasks before exit"
                )
                break

            batch_count += 1
            box_task, ready_event = box_task_tuple
            macro_bid = box_task.macro_bid
            logger.debug(
                f"BoxHandler: Batch #{batch_count}, macro_bid={macro_bid}, waiting for ready_event"
            )
            torch.cuda.current_stream().wait_event(ready_event)

            # handle mail or box sync
            # generate new mails from root_new_mem
            logger.debug(
                f"BoxHandler: Batch #{batch_count}, generating new mails, macro_bid={macro_bid}"
            )
            new_mail = generate_interaction_mails(
                box_task.root_new_mem_orig,
                box_task.pos_efeat,
            )
            logger.debug(
                f"BoxHandler: Batch #{batch_count}, new_mail size={new_mail.nid.numel()}, macro_bid={macro_bid}"
            )
            logger.debug(
                f"BoxHandler: Batch #{batch_count}, start p2p for mail/box, macro_bid={macro_bid}"
            )

            if box_task.r2n_mdt is None:
                # r2n_mdt是用来发送mail到别的box的，如果没有,说明不用发送
                logger.debug(
                    f"BoxHandler: Batch #{batch_count}, handling small box (no r2n_mdt)"
                )
                p2p_task, box_ts_nid = self.handle_small_box(
                    new_mail, self.mail_combine, box_task
                )
            else:
                logger.debug(
                    f"BoxHandler: Batch #{batch_count}, handling large box (with r2n_mdt)"
                )
                p2p_task, box_ts_nid = self.handle_large_box(
                    new_mail, self.mail_combine, box_task
                )

            if box_task.is_last is not None:
                logger.debug(
                    f"BoxHandler: Batch #{batch_count}, putting p2p_task to queue"
                )
                self.box_p2p_q.put(p2p_task)
            else:
                assert p2p_task is None
                logger.debug(
                    f"BoxHandler: Batch #{batch_count}, no p2p needed, putting 'nothing_to_p2p'"
                )
                self.box_p2p_q.put("nothing_to_p2p")

            logger.debug(
                f"BoxHandler: Batch #{batch_count}, finished p2p processing, macro_bid={macro_bid}"
            )

            if box_task.box_plan.writeback is not None:
                logger.debug(
                    f"BoxHandler: Batch #{batch_count}, starting write back, writeback size={box_task.box_plan.writeback.data.numel()}"
                )
                self.write_back_box(
                    box_task.box_plan.writeback.data,
                    box_ts_nid,
                )
                logger.debug(
                    f"BoxHandler: Batch #{batch_count}, finished write back, macro_bid={macro_bid}"
                )
            else:
                logger.debug(f"BoxHandler: Batch #{batch_count}, no write back needed")

            logger.debug(f"BoxHandler: Batch #{batch_count}, releasing h2d_sem")
            self.h2d_sem.release()
            logger.debug(f"BoxHandler: Batch #{batch_count} complete")

        logger.debug(
            f"BoxHandler: Exiting logic loop, processed {batch_count} box tasks"
        )
