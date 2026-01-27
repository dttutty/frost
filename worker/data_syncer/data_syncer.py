import logging
import threading
from queue import Queue as Q
from typing import TypeVar

import torch

from schemas.datatypes.data_to_sync import BoxTsNid, MailTsNid, MemTsNid
from schemas.datatypes.misc import Shared
from schemas.datatypes.task import BoxTask, MemTask, P2PTask, SyncTask
from thread_base_with_stream import StreamThreadBase

from .box_handler import BoxHandler
from .mem_handler import MemHandler, P2PHandler


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")
logger = logging.getLogger(__name__)


class DataSyncer(StreamThreadBase):
    def __init__(
        self,
        mail_combine: str,
        merge_src: bool,
        n_gpu: int,
        shared: Shared,
        h2d_sem: threading.Semaphore,
        sync_q: Q[tuple[SyncTask, torch.cuda.Event] | None],
        stop_ev: threading.Event,
        mem_p2p_q: Q[P2PTask | None],
        box_p2p_q: Q[P2PTask | None],
    ) -> None:
        super().__init__(name="data_syncer")
        self.sync_q = sync_q
        # 将原 GNN init 中关于 mailbox 和 deliver 的配置移到这里
        self.h2d_sem = h2d_sem
        self.box_stream = torch.cuda.Stream()
        self.stop_ev = stop_ev
        self.mem_task_q: Q[MemTask | None] = Q()
        self.box_task_q: Q[BoxTask | None] = Q()

        self.p2p_handler = P2PHandler(
            n_gpu=n_gpu,
            mem_p2p_q=mem_p2p_q,
            box_p2p_q=box_p2p_q,
        )

        self.mem_handler = MemHandler(
            shared_mem=shared.mem,
            shared_mem_ts=shared.mem_ts,
            mem_task_q=self.mem_task_q,
            mem_p2p_q=mem_p2p_q,
            h2d_sem=h2d_sem,
            stop_ev=stop_ev,
            p2p_handler=self.p2p_handler,
        )
        self.mem_handler.start()
        self.box_handler = BoxHandler(
            shared_box=shared.box,
            shared_box_ts=shared.box_ts,
            box_task_q=self.box_task_q,
            box_p2p_q=box_p2p_q,
            h2d_sem=h2d_sem,
            stop_ev=stop_ev,
            p2p_handler=self.p2p_handler,
            mail_combine=mail_combine,
            merge_src=merge_src,
        )
        self.box_handler.start()

    @torch.no_grad()
    def logic(self) -> None:
        while self.stop_ev.is_set() is False:
            sync_q_data = self.sync_q.get(block=True)
            if sync_q_data is None:
                self.mem_task_q.put(None)
                self.box_task_q.put(None)
                self.mem_handler.join()
                self.box_handler.join()
                break
            sync_task, ready_event = sync_q_data
            root_new_mem = sync_task.root_new_mem
            old_mem_n_box = sync_task.old_mem_n_box
            torch.cuda.current_stream().wait_event(ready_event)

            root_new_mem_orig = root_new_mem.clone()  # 这里的ts是交互发生的时间

            # handle mem sync
            mem_task = MemTask(
                macro_bid=sync_task.macro_bid,
                root_new_mem=root_new_mem,
                old_mem_ts_nid=old_mem_n_box.mem,
                mem_plan=sync_task.mem_plan,
                is_last=sync_task.is_this_epoch_last_macro,
            )
            self.mem_task_q.put(mem_task)

            box_task = BoxTask(
                macro_bid=sync_task.macro_bid,
                root_new_mem_orig=root_new_mem_orig,
                pos_efeat=sync_task.pos_efeat,
                old_box=old_mem_n_box.box,
                box_plan=sync_task.box_plan,
                is_last=sync_task.is_this_epoch_last_macro,
                r2n_mdt=sync_task.r2n_mdt,
            )
            self.box_task_q.put(box_task)
