import logging
import threading
from queue import Queue as Q

import torch
import torch.distributed as dist

from schemas.datatypes.env import Env

# In utils.py
from schemas.datatypes.misc import Meta
from thread_base_with_stream import StreamThreadBase


logger = logging.getLogger(__name__)


class HostWorkerSyncer(StreamThreadBase):
    def __init__(
        self,
        env: Env,
        gloo: dist.ProcessGroup,
        out_q: Q[tuple[Meta, torch.cuda.Event] | None],
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="receiver")
        self.env: Env = env
        self.gloo = gloo
        self.out_q = out_q
        self.stop_ev = stop_ev

    @torch.no_grad()
    def logic(self) -> None:
        while not self.stop_ev.is_set():
            meta_buf = [None]
            dist.scatter_object_list(
                meta_buf, None, src=self.env.host_rank, group=self.gloo
            )
            meta = meta_buf[0]

            ready_event: torch.cuda.Event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            if meta is None:
                self.out_q.put(None, block=True)
                logger.debug(
                    f"Rank {self.env.local_rank}: Received stop. Exiting receiver."
                )
                break
            else:
                self.out_q.put((meta, ready_event), block=True)
