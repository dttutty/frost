import logging
from queue import Queue

import torch
import torch.nn as nn

from schemas.datatypes.data_to_sync import MemTsNid
from schemas.datatypes.misc import ForwardCtx
from schemas.datatypes.task import SyncTask


logger = logging.getLogger(__name__)


class MemoryBaseGNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        memory_updater: nn.Module,
        sync_q: Queue[SyncTask],
        header: nn.Module,
    ):
        super().__init__()
        self.gnn = gnn
        self.memory_updater = memory_updater
        self.sync_q = sync_q
        self.header = header

    def forward(self, ctx: ForwardCtx):
        self._sync_memory(ctx=ctx)
        h = self.gnn(ctx.blockss)
        return self.header(h)

    @torch.no_grad()
    def _sync_memory(self, *, ctx: ForwardCtx):
        block = ctx.blockss[0][0]
        if block is not None:
            updated_mem = self.memory_updater(block)
            ts = block.srcdata["ts"]
            nid = block.srcdata["ID"]

            root_new_mem = MemTsNid(
                ts=ts[: ctx.n_root_src_n_pos_dst],
                data=updated_mem[: ctx.n_root_src_n_pos_dst],
                nid=nid[: ctx.n_root_src_n_pos_dst],
            )

        else:
            assert ctx.root_info is not None
            updated_mem = self.memory_updater(root_info=ctx.root_info)
            root_new_mem = MemTsNid(
                ts=ctx.root_info.ts[: ctx.n_root_src_n_pos_dst],
                data=updated_mem[: ctx.n_root_src_n_pos_dst],
                nid=ctx.root_info.nid[: ctx.n_root_src_n_pos_dst],
            )

        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())

        self.sync_q.put(
            SyncTask(
                macro_bid=ctx.macro_bid,
                plan_set=ctx.plan_set,
                root_new_mem=root_new_mem,
                old_x=ctx.x,
                mail_delivery_tensor=ctx.mail_delivery_tensor,
                is_last=ctx.is_last,
                ready_event=ready_event,
            )
        )
