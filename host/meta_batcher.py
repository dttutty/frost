import logging
import threading
from queue import Queue
from typing import Literal

from schemas.alias import MetaBundle
from schemas.datatypes.misc import Meta
from thread_base import ThreadBase


logger = logging.getLogger(__name__)


class MetaBatcher(ThreadBase):
    def __init__(
        self,
        n_gpu: int,
        in_q: Queue[Meta | None | Literal["end_of_epoch"]],
        out_q: Queue[MetaBundle | None | Literal["end_of_epoch"]],
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="meta_batcher", stop_ev=stop_ev)
        self.n_gpu = n_gpu
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev

    def logic(self) -> None:
        batch_count = 0

        while not self.stop_ev.is_set():
            meta_list: list[Meta] = []

            while len(meta_list) < self.n_gpu:
                meta = self.in_q.get(block=True)

                if meta is None:
                    self.out_q.put(None)
                    return
                elif meta == "end_of_epoch":
                    self.out_q.put("end_of_epoch")
                else:
                    meta_list.append(meta)

            if len(meta_list) == self.n_gpu:
                batch_count += 1
                self.out_q.put(meta_list)
