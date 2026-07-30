import logging
import threading
from typing import Literal, cast

import torch

from frost.runtime.worker.types import (
    BoxP2PQ,
    BoxTaskQ,
    BoxWrQ,
    MemP2PQ,
    MemTaskQ,
    MemWrQ,
)
from frost.datatypes.mem_mail_box._data_ops import (
    keep_last,
    keep_mean,
    mail_to_box,
    merge_with_priority,
)
from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.tasks import BoxTask, P2PTask
from frost.runtime.protocols import CudaEvent
from frost.models.protocols import MailboxRouterProto
from frost.runtime._thread_base import StreamThreadBase, put_shutdown_sentinel
from frost.runtime.worker._h2d_credit import CreditGate

from .utils._interaction_ops import generate_interaction_mails
from .utils._p2p_task_maker import make_p2p_task

logger = logging.getLogger(__name__)


def _dedupe_box_entry(entry: BoxEntry) -> BoxEntry:
    if entry.nid.numel() == 0:
        entry.is_sorted = True
        entry.is_unique = True
        return entry

    entry = entry.sort_by_nid()
    if entry.is_unique:
        return entry

    keep = torch.ones(entry.nid.numel(), dtype=torch.bool, device=entry.nid.device)
    if entry.nid.numel() > 1:
        keep[:-1] = entry.nid[:-1] != entry.nid[1:]

    if keep.sum().item() != entry.nid.numel():
        return BoxEntry(
            nid=entry.nid[keep],
            data=entry.data[keep],
            ts=entry.ts[keep],
            is_sorted=True,
            is_unique=True,
        )

    entry.is_unique = True
    return entry


def _merge_box_entries_with_priority(
    fresh_box: BoxEntry,
    stale_box: BoxEntry | None,
) -> BoxEntry:
    if stale_box is None or stale_box.nid.numel() == 0:
        return _dedupe_box_entry(fresh_box)

    fresh_box = _dedupe_box_entry(fresh_box)
    stale_box = _dedupe_box_entry(stale_box)

    if fresh_box.data.shape[1:] != stale_box.data.shape[1:]:
        logger.warning(
            "Skipping stale box merge because shapes differ: fresh=%s stale=%s",
            tuple(fresh_box.data.shape[1:]),
            tuple(stale_box.data.shape[1:]),
        )
        return fresh_box

    keep_low = ~torch.isin(stale_box.nid, fresh_box.nid)
    if not torch.any(keep_low):
        return fresh_box

    cat_nid = torch.cat((fresh_box.nid, stale_box.nid[keep_low]), dim=0)
    cat_data = torch.cat((fresh_box.data, stale_box.data[keep_low]), dim=0)
    cat_ts = torch.cat((fresh_box.ts, stale_box.ts[keep_low]), dim=0)
    sort_idx = torch.argsort(cat_nid)
    return BoxEntry(
        nid=cat_nid[sort_idx],
        data=cat_data[sort_idx],
        ts=cat_ts[sort_idx],
        is_sorted=True,
        is_unique=True,
    )


class MemTaskHandler(StreamThreadBase):
    def __init__(
        self,
        mem_task_q: MemTaskQ,
        mem_p2p_q: MemP2PQ,
        stop_ev: threading.Event,
        device: torch.device,
        mem_wr_q: MemWrQ,
        h2d_sem: CreditGate,
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
            try:
                if mem_task_tuple is None:
                    put_shutdown_sentinel(self.mem_wr_q, None)
                    break

                mem_task, ready_event = mem_task_tuple
                torch.cuda.current_stream().wait_event(ready_event)
                mem_plan = mem_task.mem_plan
                root_new_mem = keep_last(mem_task.root_new_mem)

                mem_ts_nid: MemEntry = root_new_mem
                if (
                    mem_task.stale_mem is not None
                    and mem_task.stale_mem.nid.numel() > 0
                ):
                    mem_ts_nid = cast(
                        MemEntry,
                        merge_with_priority(root_new_mem, mem_task.stale_mem),
                    )
                mini_batch_idx = mem_task.mini_batch_idx

                if mem_task.is_last is False:
                    p2p_task = make_p2p_task(
                        entry=mem_ts_nid,
                        plan=mem_plan,
                        device=self.device,
                        mini_batch_idx=mini_batch_idx,
                    )
                    p2p_event: CudaEvent = cast(CudaEvent, torch.cuda.Event())
                    p2p_event.record(torch.cuda.current_stream())
                    self.mem_p2p_q.put((p2p_task, p2p_event))
                else:
                    self.mem_p2p_q.put("nothing_to_p2p")

                writeback_event: CudaEvent = cast(CudaEvent, torch.cuda.Event())
                writeback_event.record(torch.cuda.current_stream())
                if mem_task.mem_plan.writeback is not None:
                    to_writeback = mem_task.mem_plan.writeback.data
                    self.mem_wr_q.put(
                        (to_writeback, mini_batch_idx, mem_ts_nid, writeback_event)
                    )
                else:
                    self.mem_wr_q.put(
                        (None, mini_batch_idx, mem_ts_nid, writeback_event)
                    )
            finally:
                self.mem_task_q.task_done()


class MessageHandler(StreamThreadBase):
    def __init__(
        self,
        box_task_q: BoxTaskQ,
        box_p2p_q: BoxP2PQ,
        stop_ev: threading.Event,
        mail_combine: Literal["last", "mean"],
        device: torch.device,
        box_wr_q: BoxWrQ,
        h2d_sem: CreditGate,
        mailbox_router: MailboxRouterProto | None,
        n_neg: int = 1,
    ) -> None:
        super().__init__(name="msg_handler", stop_ev=stop_ev)

        self.device = device
        self.box_task_q = box_task_q
        self.box_p2p_q = box_p2p_q
        self.stop_ev = stop_ev
        self.mail_combine: Literal["last", "mean"] = mail_combine
        self.mailbox_router = mailbox_router
        self.box_wr_q = box_wr_q
        self.h2d_sem = h2d_sem
        self.n_neg = n_neg

    def _process_messages(
        self,
        new_mail: MailEntry,
        mail_combine: Literal["last", "mean"],
        box_task: BoxTask,
    ) -> tuple[P2PTask, BoxEntry, MailEntry | None]:
        """Aggregate and route messages, producing a boxsize=1 BoxEntry for P2P.

        For neighborhood-mail contracts, routes seed mail to neighbors first.
        The CPU writeback uses MemBoxStore.write_mail() (ring buffer).

        Returns:
            (p2p_task, box_entry, seed_mail_cpu)
            seed_mail_cpu is set only when neighborhood routing is active;
            it carries the raw seed mails on CPU for the CPU-side writeback.
        """
        box_plan = box_task.box_plan
        seed_mail_cpu: MailEntry | None = None

        # Route seed mail to one-hop neighbors on the P2P path.
        # but also capture the raw seed mail for CPU-side writeback
        if self.mailbox_router is not None:
            # Snapshot seed mail before routing expands it.
            seed_mail_cpu = MailEntry(
                nid=new_mail.nid.cpu(),
                data=new_mail.data.cpu(),
                ts=new_mail.ts.cpu(),
                is_sorted=new_mail.is_sorted,
                is_unique=new_mail.is_unique,
            )
            graph = box_task.graph
            assert graph is not None, (
                "BoxTask.graph must not be None when mailbox routing is active"
            )
            new_mail = self.mailbox_router.deliver(graph, mail_data=new_mail.data)

        if mail_combine == "last":
            new_mail = keep_last(new_mail)
        else:
            assert mail_combine == "mean"
            new_mail = keep_mean(new_mail)

        box_entry = mail_to_box(new_mail)
        box_entry = _merge_box_entries_with_priority(box_entry, box_task.stale_box)

        p2p_task = make_p2p_task(
            entry=box_entry,
            plan=box_plan,
            device=self.device,
            mini_batch_idx=box_task.mini_batch_idx,
        )

        return p2p_task, box_entry, seed_mail_cpu

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0

        while not self.stop_ev.is_set():
            box_task_tuple = self.box_task_q.get(block=True)
            try:
                if box_task_tuple is None:
                    put_shutdown_sentinel(self.box_wr_q, None)
                    break

                batch_count += 1
                box_task, ready_event = box_task_tuple

                torch.cuda.current_stream().wait_event(ready_event)

                new_mail = generate_interaction_mails(
                    box_task.root_new_mem,
                    box_task.pos_efeat,
                )

                p2p_task, box_entry, seed_mail_cpu = self._process_messages(
                    new_mail, self.mail_combine, box_task
                )

                if box_task.is_last is False:
                    p2p_event = cast(CudaEvent, torch.cuda.Event())
                    p2p_event.record(torch.cuda.current_stream())
                    self.box_p2p_q.put((p2p_task, p2p_event))
                else:
                    self.box_p2p_q.put("nothing_to_p2p")

                writeback_event = cast(CudaEvent, torch.cuda.Event())
                writeback_event.record(torch.cuda.current_stream())
                if box_task.box_plan.writeback is not None:
                    # Send seed mail + CPU layout for CPU-side neighbor scatter.
                    if seed_mail_cpu is not None and box_task.flat_nid_cpu is not None:
                        self.box_wr_q.put(
                            (
                                box_task.box_plan.writeback.data,
                                box_task.mini_batch_idx,
                                (
                                    seed_mail_cpu,
                                    box_task.flat_nid_cpu,
                                    box_task.graph.n_neg
                                    if box_task.graph is not None
                                    else self.n_neg,
                                ),
                                writeback_event,
                            )
                        )
                    else:
                        self.box_wr_q.put(
                            (
                                box_task.box_plan.writeback.data,
                                box_task.mini_batch_idx,
                                box_entry,
                                writeback_event,
                            )
                        )
                else:
                    self.box_wr_q.put(
                        (None, box_task.mini_batch_idx, box_entry, writeback_event)
                    )
            finally:
                self.box_task_q.task_done()
