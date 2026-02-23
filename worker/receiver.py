import logging
import threading
from queue import Queue as Q
from typing import Literal

import torch
import torch.distributed as dist

from schemas.datatypes.env import Env

# In utils.py
from schemas.datatypes.misc import Result
from schemas.shm_layout import MANIFEST_SIZE, ShmKeys
from thread_base_with_stream import StreamThreadBase


logger = logging.getLogger(__name__)


class Receiver(StreamThreadBase):
    def __init__(
        self,
        env: Env,
        gloo: dist.ProcessGroup,
        out_q: Q[torch.Tensor | None | Literal["end_of_epoch"]],
        result_q: Q[Result | None],
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="receiver", stop_ev=stop_ev)
        self.env: Env = env
        self.gloo = gloo
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.result_q = result_q
        self.manifest_buf = torch.zeros(MANIFEST_SIZE, dtype=torch.long)
        self._gather_seq = 0
        self._recv_seq = 0

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0

        while not self.stop_ev.is_set():
            self.manifest_buf.zero_()
            # dist.scatter(self.manifest_buf, src=self.env.host_rank, group=self.gloo)

            # dist.recv(self.manifest_buf, src=self.env.host_rank, group=self.gloo)
            dist.scatter(self.manifest_buf, src=self.env.host_rank, group=self.gloo)
            batch_idx = int(self.manifest_buf[ShmKeys.BATCH_IDX].item())
            self._recv_seq += 1
            # logger.debug(
            #     "Receiver recv: seq=%d batch_idx=%d rank=%d",
            #     self._recv_seq,
            #     batch_idx,
            #     self.env.local_rank,
            # )

            if batch_idx == -2:
                self.out_q.put(None, block=True)
                break
            elif batch_idx == -1:
                self.out_q.put("end_of_epoch", block=True)
                result = self.result_q.get(block=True)

                # --- [Fix] 开始修改 gather 逻辑 ---
                # 只有 dst rank (0) 需要准备接收容器，大小必须等于进程总数
                self._gather_seq += 1
                logger.debug(
                    "Receiver gather_object start: seq=%d batch_idx=%d rank=%d",
                    self._gather_seq,
                    batch_idx,
                    self.env.local_rank,
                )
                dist.gather_object(
                    obj=result,
                    object_gather_list=None,
                    dst=self.env.host_rank,
                    group=self.gloo,
                )
                logger.debug(
                    "Receiver gather_object end: seq=%d batch_idx=%d rank=%d",
                    self._gather_seq,
                    batch_idx,
                    self.env.local_rank,
                )
            else:
                batch_count += 1
                self.out_q.put(self.manifest_buf.clone(), block=True)
