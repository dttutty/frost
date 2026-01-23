import logging
import threading
from queue import Queue

from schemas.alias import MetaBundle
from schemas.datatypes.misc import Meta
from thread_base import ThreadBase


logger = logging.getLogger(__name__)


class MetaBatcher(ThreadBase):
    def __init__(
        self,
        n_gpus: int,
        in_q: Queue[Meta | None],
        out_q: Queue[MetaBundle | None],
        stop_ev: threading.Event,
    ):
        super().__init__(name="meta_batcher")
        self.n_gpus = n_gpus
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev

    def logic(self):
        while not self.stop_ev.is_set():
            meta_list = []
            while len(meta_list) < self.n_gpus:
                meta = self.in_q.get(block=True)
                if meta is None:
                    self.out_q.put(None)
                else:
                    meta_list.append(meta)
            self.out_q.put(meta_list)
