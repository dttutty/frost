import logging
import threading

from frost.aliases.host import AccessBundle, MetaBatcherInQ, MetaBundle, MetaBundleQ
from frost.runtime._thread_base import ThreadBase

logger = logging.getLogger(__name__)


class MetaBatcher(ThreadBase):
    def __init__(
        self,
        n_gpu: int,
        meta_in_q: MetaBatcherInQ,
        dest_q: MetaBundleQ,
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="meta_batcher", stop_ev=stop_ev)
        self.n_gpu = n_gpu
        self.meta_in_q = meta_in_q
        self.dest_q = dest_q
        self.stop_ev = stop_ev

    def logic(self) -> None:
        batch_count = 0

        while not self.stop_ev.is_set():
            meta_list: MetaBundle = []
            access_list: AccessBundle = []

            while len(meta_list) < self.n_gpu:
                item = self.meta_in_q.get(block=True)

                if item is None:
                    self.dest_q.put(None)
                    return
                elif item == "end_of_epoch":
                    self.dest_q.put("end_of_epoch")
                else:
                    meta, access = item
                    meta_list.append(meta)
                    access_list.append(access)

            if len(meta_list) == self.n_gpu:
                batch_count += 1
                self.dest_q.put((meta_list, access_list))
