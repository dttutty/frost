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
from schemas.protocol.misc import CudaEventLike


logger = logging.getLogger(__name__)


class MemoryBaseGNN(nn.Module):
    def __init__(
        self,
        gnn: TGN | JODIE | APAN,
        memory_updater: TransformerMemoryUpdater
        | MfgMemoryUpdater
        | NoMfgMemoryUpdater,
        mem_task_q: Queue[tuple[MemTask, CudaEventLike] | None],
        box_task_q: Queue[tuple[BoxTask, CudaEventLike] | None],
        header: EdgePredictor,
    ) -> None:
        super().__init__()
        self.gnn = gnn
        self.memory_updater = memory_updater
        self.mem_task_q = mem_task_q
        self.box_task_q = box_task_q
        self.box_stream = torch.cuda.Stream()
        self.header = header
        self.mu_ready_event = torch.cuda.Event(
            enable_timing=False, blocking=False, interprocess=False
        )
        self.box_ready_event = torch.cuda.Event(
            enable_timing=False, blocking=False, interprocess=False
        )

    def forward(self, meta: Meta) -> torch.Tensor:
        logger.info(f"Processing meta with meta.batch_id={meta.batch_idx}")
        self.update_and_sync_memory(meta=meta)
        # torch.cuda.current_stream().wait_event(mu_ready)
        if isinstance(self.gnn, TGN):
            h = self.gnn(mfgs=meta.n2r_mfgs)

        else:
            assert meta.mem_n_box is not None
            h = self.gnn(
                mem_n_box=meta.mem_n_box.mem,
                nfeat=meta.nfeat,
                ts=meta.root_ts,
            )

        ret = self.header(h)
        return cast(torch.Tensor, ret)

    def update_and_sync_memory(self, meta: Meta) -> None:
        n_root_src_n_pos_dst = meta.n_src * 2
        assert meta.mem_n_box is not None
        if meta.n2r_mfgs is not None:
            assert isinstance(self.memory_updater, MfgMemoryUpdater)
            # only support 1-layer gnn for memory update
            # only support 1-history window for memory update
            assert len(meta.n2r_mfgs) == 1 and len(meta.n2r_mfgs[0]) == 1
            block = meta.n2r_mfgs[0][0]

            root_old_mem = meta.mem_n_box.mem
            root_old_box = meta.mem_n_box.box
            if block is not None:
                block = self.memory_updater(block, n_root_src_n_pos_dst)
                updated_mem = self.memory_updater.detached_up_mem
                ts = self.memory_updater.detached_ts
                nid = self.memory_updater.detached_nid

        # else:
        #     assert meta.mem_n_box is not None
        #     assert isinstance(
        #         self.memory_updater, (TransformerMemoryUpdater, NoMfgMemoryUpdater)
        #     )
        #     root_old_mem = meta.mem_n_box.mem
        #     root_old_box = meta.mem_n_box.box
        #     mem_n_box = meta.mem_n_box
        #     updated_mem = self.memory_updater(
        #         mem_n_box=mem_n_box, root_ts=meta.root_ts.repeat_interleave(3)
        #     )

        with torch.no_grad():  # <--- 在这里开启无梯度模式
            # 这里的切片和创建对象操作如果不关梯度，会产生大量无用的计算图节点
            if meta.n2r_mfgs is not None and block is not None:
                root_new_mem = MemTsNid(
                    ts=ts,
                    data=updated_mem,
                    nid=nid,
                )
            elif meta.n2r_mfgs is None:
                root_new_mem = MemTsNid(
                    ts=meta.root_ts.repeat_interleave(2),
                    data=updated_mem[:n_root_src_n_pos_dst],
                    nid=meta.root_nid[:n_root_src_n_pos_dst],
                )

            self.mu_ready_event.record()

            m_state = meta.m_state
            plan_set = meta.plan_set

            assert plan_set is not None
            assert plan_set.box is not None
            assert plan_set.mem is not None

            # with torch.cuda.stream(self.box_stream):
            #     self.box_stream.wait_stream(torch.cuda.current_stream())
            root_new_mem_orig = root_new_mem.clone()  # 这里的ts是交互发生的时间
            # invalid_mask = (root_new_mem_orig.nid < 0) | (root_new_mem_orig.nid > 3000)
            # if invalid_mask.any():
            #     raise ValueError(
            #         f"Found invalid nids in root_new_mem_orig: {root_new_mem_orig.nid[invalid_mask]}"
            #     )
            self.box_ready_event.record()

            # handle mem sync
            mem_task = MemTask(
                macro_bid=m_state.cur_macro_of_epoch,
                root_new_mem=root_new_mem,
                old_mem_ts_nid=root_old_mem,
                mem_plan=plan_set.mem,
                is_last=m_state.is_last_macro_of_epoch,
                meta=meta,
            )
            self.mem_task_q.put((mem_task, self.mu_ready_event))

            # with torch.cuda.stream(self.box_stream):
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
            self.box_task_q.put((box_task, self.box_ready_event))
