import logging
import threading
from queue import Queue as Q
from typing import Literal

import torch
import torch.distributed as dist

from schemas.datatypes.env import Env

# In utils.py
from schemas.datatypes.misc import Meta, Result
from thread_base_with_stream import StreamThreadBase


logger = logging.getLogger(__name__)


class Receiver(StreamThreadBase):
    def __init__(
        self,
        env: Env,
        gloo: dist.ProcessGroup,
        out_q: Q[Meta | None | Literal["end_of_epoch"]],
        result_q: Q[Result | None],
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="receiver", stop_ev=stop_ev)
        self.env: Env = env
        self.gloo = gloo
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.result_q = result_q

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0
        logger.debug(f"Receiver: starting logic loop on rank {self.env.local_rank}")

        while not self.stop_ev.is_set():
            logger.debug(
                f"Receiver: rank {self.env.local_rank} waiting for scatter from host rank {self.env.host_rank}"
            )
            meta_buf = [None]
            dist.scatter_object_list(
                meta_buf, None, src=self.env.host_rank, group=self.gloo
            )
            meta = meta_buf[0]
            logger.debug(
                f"Receiver: rank {self.env.local_rank} received meta type: {type(meta).__name__}"
            )

            if meta is None:
                logger.debug(
                    f"Receiver: rank {self.env.local_rank} received None, putting to out_q"
                )
                self.out_q.put(None, block=True)
                logger.debug(
                    f"Rank {self.env.local_rank}: Received stop. Exiting receiver."
                )
                break
            elif meta == "end_of_epoch":
                logger.debug(
                    f"Receiver: rank {self.env.local_rank} received end_of_epoch, putting to out_q"
                )
                self.out_q.put("end_of_epoch", block=True)
                logger.debug(
                    f"Receiver: rank {self.env.local_rank} waiting for result from result_q"
                )
                result = self.result_q.get(block=True)
                logger.debug(
                    f"Receiver: rank {self.env.local_rank} got result, gathering to dst=0"
                )
                dist.gather_object(
                    result,
                    object_gather_list=None,  # 非 dst rank 必须是 None
                    dst=0,
                )
                logger.debug(f"Receiver: rank {self.env.local_rank} gather complete")
            elif isinstance(meta, Meta):
                batch_count += 1
                logger.debug(
                    f"Receiver: rank {self.env.local_rank} received Meta batch #{batch_count}, putting to out_q"
                )
                self.out_q.put(meta, block=True)
                logger.debug(
                    f"Receiver: rank {self.env.local_rank} Meta batch #{batch_count} put successfully"
                )

        logger.debug(
            f"Receiver: rank {self.env.local_rank} exiting, processed {batch_count} batches"
        )
