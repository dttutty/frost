from queue import Queue

import torch
import torch.nn as nn

from schemas.datatypes import ForwardCtx, MemTsNid, SyncTask


class MemoryBaseGNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        memory_updater: nn.Module,
        sync_q: Queue[SyncTask],
        predictor: nn.Module,
    ):
        super().__init__()
        self.gnn = gnn
        self.memory_updater = memory_updater
        self.sync_q = sync_q
        self.predictor = predictor

    def forward(self, ctx: ForwardCtx):
        self._maybe_sync_memory(ctx=ctx)
        h = self.gnn(ctx.blockss)
        return self.predictor(h)

    @torch.no_grad()
    def _maybe_sync_memory(self, *, ctx: ForwardCtx):
        block = ctx.blockss[0][0]
        updated_mem = self.memory_updater(block=block)

        ts = block.srcdata["ts"]
        nid = block.srcdata["ID"]

        root_new_mem = MemTsNid(
            ts=ts[: ctx.n_root_src_n_pos_dst],
            data=updated_mem[: ctx.n_root_src_n_pos_dst],
            id=nid[: ctx.n_root_src_n_pos_dst],
        )

        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())

        self.sync_q.put(
            SyncTask(
                macro_bid=ctx.macro_bid,
                plan_set=ctx.plan_set,
                root_new_mem=root_new_mem,
                x=ctx.x,
                mail_delivery_tensor=ctx.mail_delivery_tensor,
                is_last=ctx.is_last,
                ready_event=ready_event,
            )
        )
