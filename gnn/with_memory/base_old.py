import logging
from queue import Queue
from typing import cast

import torch
import torch.nn as nn

from gnn.components import EdgePredictor
from gnn.with_memory.apan import APAN
from gnn.with_memory.jodie import JODIE
from gnn.with_memory.memory_updater import (
    MfgMemoryUpdater,
    NoMfgMemoryUpdater,
    TransformerMemoryUpdater,
)
from gnn.with_memory.tgn import TGN
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.misc import Meta
from schemas.datatypes.task import BoxTask, MemTask


logger = logging.getLogger(__name__)


class MemoryBaseGNN(nn.Module):
    def __init__(
        self,
        gnn: TGN | JODIE | APAN,
        memory_updater: TransformerMemoryUpdater
        | MfgMemoryUpdater
        | NoMfgMemoryUpdater,
        mem_task_q: Queue[tuple[MemTask, torch.cuda.Event] | None],
        box_task_q: Queue[tuple[BoxTask, torch.cuda.Event] | None],
        header: EdgePredictor,
    ) -> None:
        super().__init__()
        self.gnn = gnn
        self.memory_updater = memory_updater
        self.mem_task_q = mem_task_q
        self.box_task_q = box_task_q
        self.header = header
        self.updater_stream = torch.cuda.Stream()

    def forward(self, meta: Meta) -> torch.Tensor:
        self.update_and_sync_memory(meta=meta)
        if isinstance(self.gnn, TGN):
            h = self.gnn(mfgs=meta.n2r_mfgs)

        else:
            assert meta.mem_n_box is not None
            h = self.gnn(
                mem_n_box=meta.mem_n_box.mem,
                nfeat=meta.nfeat,
                ts=meta.root_ts,
            )

        return cast(torch.Tensor, self.header(h))

    def update_and_sync_memory(self, meta: Meta) -> None:
        n_root_src_n_pos_dst = meta.n_src * 2
        assert meta.mem_n_box is not None
        if meta.n2r_mfgs is not None:
            assert isinstance(self.memory_updater, MfgMemoryUpdater)
            # only support 1-layer gnn for memory update
            # only support 1-history window for memory update
            assert len(meta.n2r_mfgs) == 1 and len(meta.n2r_mfgs[0]) == 1
            block = meta.n2r_mfgs[0][0]

            # if block is not None:
            #     mem = block.srcdata["mem"]
            #     mem_ts = block.srcdata["mem_ts"]
            #     nid = block.srcdata["ID"]

            # 可能不跟root_ts一一对应

            # root_old_mem = MemTsNid(
            #     ts=mem_ts,
            #     data=mem,
            #     nid=nid,
            # )
            # root_old_mail = MailTsNid(
            #     ts=block.srcdata["box_ts"],
            #     data=block.srcdata["box"],
            #     nid=nid,
            # )
            # root_old_box = root_old_mail.to_box()

            root_old_mem = meta.mem_n_box.mem
            root_old_box = meta.mem_n_box.box
            if block is not None:
                block, updated_mem = self.memory_updater(block)
                ts = block.srcdata["ts"]
                nid = block.srcdata["ID"]

                # 可能不跟root_ts一一对应

                root_new_mem = MemTsNid(
                    ts=ts[:n_root_src_n_pos_dst],
                    data=updated_mem[:n_root_src_n_pos_dst],
                    nid=nid[:n_root_src_n_pos_dst],
                )

        else:
            assert meta.mem_n_box is not None
            assert isinstance(
                self.memory_updater, (TransformerMemoryUpdater, NoMfgMemoryUpdater)
            )
            root_old_mem = meta.mem_n_box.mem
            root_old_box = meta.mem_n_box.box
            mem_n_box = meta.mem_n_box
            updated_mem = self.memory_updater(
                mem_n_box=mem_n_box, root_ts=meta.root_ts.repeat_interleave(3)
            )
            root_new_mem = MemTsNid(
                ts=meta.root_ts.repeat_interleave(2),
                data=updated_mem[:n_root_src_n_pos_dst],
                nid=meta.root_nid[:n_root_src_n_pos_dst],
            )

        ready_event: torch.cuda.Event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())

        m_state = meta.m_state
        plan_set = meta.plan_set
        assert plan_set is not None
        assert plan_set.box is not None
        assert plan_set.mem is not None

        root_new_mem_orig = root_new_mem.clone()  # 这里的ts是交互发生的时间

        # handle mem sync
        mem_task = MemTask(
            macro_bid=m_state.cur_macro_of_epoch,
            root_new_mem=root_new_mem,
            old_mem_ts_nid=root_old_mem,
            mem_plan=plan_set.mem,
            is_last=m_state.is_last_macro_of_epoch,
            meta=meta,
        )
        self.mem_task_q.put((mem_task, ready_event))

        box_task = BoxTask(
            macro_bid=m_state.cur_macro_of_epoch,
            root_new_mem_orig=root_new_mem_orig,
            root_nid=meta.root_nid[:n_root_src_n_pos_dst],
            pos_efeat=meta.pos_efeat,
            old_box=root_old_box,
            box_plan=plan_set.box,
            is_last=m_state.is_last_macro_of_epoch,
            r2n_mdt=meta.r2n_mdt,
            meta=meta,
        )
        self.box_task_q.put((box_task, ready_event))
