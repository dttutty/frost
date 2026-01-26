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
        super().__init__(name="receiver")
        self.env: Env = env
        self.gloo = gloo
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.result_q = result_q

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            meta_buf = [None]
            dist.scatter_object_list(
                meta_buf, None, src=self.env.host_rank, group=self.gloo
            )
            meta = meta_buf[0]

            if meta is None:
                self.out_q.put(None, block=True)
                logger.debug(
                    f"Rank {self.env.local_rank}: Received stop. Exiting receiver."
                )
                break
            elif meta == "end_of_epoch":
                self.out_q.put("end_of_epoch", block=True)
                result = self.result_q.get(block=True)
                dist.gather_object(
                    result,
                    object_gather_list=None,  # 非 dst rank 必须是 None
                    dst=0,
                )
            elif isinstance(meta, Meta):
                self.out_q.put(meta, block=True)
