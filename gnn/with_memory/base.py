import logging
from queue import Queue

import torch
import torch.nn as nn

from gnn.components import EdgePredictor
from schemas.datatypes.data_to_sync import MemTsNid
from schemas.datatypes.misc import Meta
from schemas.datatypes.task import SyncTask


logger = logging.getLogger(__name__)


class MemoryBaseGNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        memory_updater: nn.Module,
        sync_q: Queue[SyncTask],
        header: EdgePredictor,
    ) -> None:
        super().__init__()
        self.gnn = gnn
        self.memory_updater = memory_updater
        self.sync_q = sync_q
        self.header = header

    def forward(self, meta: Meta) -> torch.Tensor:
        self._sync_memory(meta=meta)
        h = self.gnn(meta.gnn_mfgs)
        return self.header(h)

    @torch.no_grad()
    def _sync_memory(self, meta: Meta) -> None:
        assert meta.mem_n_box is not None
        n_root_src_n_pos_dst = meta.n_src * 2
        if meta.gnn_mfgs is not None:
            # only support 1-layer gnn for memory update
            # only support 1-history window for memory update
            assert len(meta.gnn_mfgs) == 1 and len(meta.gnn_mfgs[0]) == 1
            assert not meta.mem_n_box.is_unzipped
            block = meta.gnn_mfgs[0][0]
            if block is not None:
                updated_mem = self.memory_updater(block)
                ts = block.srcdata["ts"]
                nid = block.srcdata["ID"]

                root_new_mem = MemTsNid(
                    ts=ts[:n_root_src_n_pos_dst],
                    data=updated_mem[:n_root_src_n_pos_dst],
                    nid=nid[:n_root_src_n_pos_dst],
                )

        else:
            assert meta.mem_n_box.is_unzipped
            mem_n_box = meta.mem_n_box
            updated_mem = self.memory_updater(root_info=mem_n_box)
            root_new_mem = MemTsNid(
                ts=meta.root_ts.repeat_interleave(2),
                data=updated_mem[:n_root_src_n_pos_dst],
                nid=meta.root_nid,
            )

        ready_event: torch.cuda.Event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())

        epoch_split_iter_info = meta.epoch_split_iter_info
        plan_set = meta.plan_set
        assert plan_set
        assert plan_set.box is not None
        assert plan_set.mem is not None

        self.sync_q.put(
            SyncTask(
                macro_bid=epoch_split_iter_info.cur_macro_of_epoch,
                mem_plan=plan_set.mem,
                box_plan=plan_set.box,
                root_new_mem=root_new_mem,
                pos_efeat=meta.pos_efeat,
                old_x=meta.mem_n_box,
                mail_delivery_tensor=meta.mdt,
                all_epoch_last_macro=epoch_split_iter_info.is_last_macro_of_all_epoch,
                this_epoch_last_macro=epoch_split_iter_info.is_last_macro_of_epoch,
                ready_event=ready_event,
            )
        )
