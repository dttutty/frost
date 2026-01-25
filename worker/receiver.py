import logging
import threading
from queue import Queue as Q
from typing import Literal

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
        cmd_q: Q[Literal["train", "end_of_epoch", "stop"]],
        result_q: Q,
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="receiver")
        self.env: Env = env
        self.gloo = gloo
        self.out_q = out_q
        self.cmd_q: Q[Literal["train", "end_of_epoch", "stop"]] = cmd_q
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
                self.cmd_q.put("stop", block=True)
                self.out_q.put(None, block=True)
                logger.debug(
                    f"Rank {self.env.local_rank}: Received stop. Exiting receiver."
                )
                break
            elif meta == "end_of_epoch":
                self.cmd_q.put("end_of_epoch", block=True)
                result = self.result_q.get(None, block=True)
                dist.gather_object(
                    result,
                    object_gather_list=None,  # 非 dst rank 必须是 None
                    dst=0,
                )
            elif isinstance(meta, Meta):
                self.cmd_q.put("train", block=True)
                ready_event: torch.cuda.Event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream())
                self.out_q.put((meta, ready_event), block=True)
