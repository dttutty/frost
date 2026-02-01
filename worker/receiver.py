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

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0
        if __debug__:
            logger.debug(f"Receiver: starting logic loop on rank {self.env.local_rank}")

        while not self.stop_ev.is_set():
            self.manifest_buf.zero_()
            if __debug__:
                logger.debug(
                    f"Receiver: rank {self.env.local_rank} waiting for scatter from host rank {self.env.host_rank}"
                )
            # dist.scatter(self.manifest_buf, src=self.env.host_rank, group=self.gloo)

            dist.recv(self.manifest_buf, src=self.env.host_rank, group=self.gloo)
            batch_idx = int(self.manifest_buf[ShmKeys.BATCH_IDX].item())
            if __debug__:
                logger.debug(
                    f"Receiver: rank {self.env.local_rank} received manifest with batch_idx: {batch_idx}"
                )

            if batch_idx == -2:
                if __debug__:
                    logger.debug(
                        f"Receiver: rank {self.env.local_rank} received None, putting to out_q"
                    )
                self.out_q.put(None, block=True)
                if __debug__:
                    logger.debug(
                        f"Rank {self.env.local_rank}: Received stop. Exiting receiver."
                    )
                break
            elif batch_idx == -1:
                self.out_q.put("end_of_epoch", block=True)
                result = self.result_q.get(block=True)

                # --- [Fix] 开始修改 gather 逻辑 ---
                # 只有 dst rank (0) 需要准备接收容器，大小必须等于进程总数

                if __debug__:
                    logger.debug(
                        f"Receiver: Rank {self.env.local_rank} gathering result to dst=0"
                    )

                dist.gather_object(
                    obj=result,
                    object_gather_list=None,
                    dst=self.env.host_rank,
                    group=self.gloo,
                )
                if __debug__:
                    logger.debug(
                        f"Receiver: rank {self.env.local_rank} gather complete"
                    )
            else:
                batch_count += 1
                if __debug__:
                    logger.debug(
                        f"Receiver: rank {self.env.local_rank} received Meta batch #{batch_count}, putting to out_q"
                    )
                self.out_q.put(self.manifest_buf.clone(), block=True)
                if __debug__:
                    logger.debug(
                        f"Receiver: rank {self.env.local_rank} Meta batch #{batch_count} put successfully"
                    )

        if __debug__:
            logger.debug(
                f"Receiver: rank {self.env.local_rank} exiting, processed {batch_count} batches"
            )
