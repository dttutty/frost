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
        if __debug__:
            logger.debug(f"MetaBatcher: initializing with n_gpu={n_gpu}")
        self.n_gpu = n_gpu
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev

    def logic(self) -> None:
        batch_count = 0
        if __debug__:
            logger.debug("MetaBatcher: starting logic loop")

        while not self.stop_ev.is_set():
            if __debug__:
                logger.debug(
                    f"MetaBatcher: starting new batch collection, target size={self.n_gpu}"
                )
            meta_list: list[Meta] = []

            while len(meta_list) < self.n_gpu:
                if __debug__:
                    logger.debug(
                        f"MetaBatcher: waiting for meta from in_q, current collected={len(meta_list)}/{self.n_gpu}"
                    )
                meta = self.in_q.get(block=True)

                if meta is None:
                    if __debug__:
                        logger.debug(
                            "MetaBatcher: received None, putting to out_q and exiting"
                        )
                    self.out_q.put(None)
                    return
                elif meta == "end_of_epoch":
                    if __debug__:
                        logger.debug(
                            "MetaBatcher: received end_of_epoch, putting to out_q"
                        )
                    self.out_q.put("end_of_epoch")
                else:
                    if __debug__:
                        logger.debug(
                            f"MetaBatcher: collected meta object, now have {len(meta_list) + 1}/{self.n_gpu}"
                        )
                    meta_list.append(meta)

            if len(meta_list) == self.n_gpu:
                batch_count += 1
                if __debug__:
                    logger.debug(
                        f"MetaBatcher: batch #{batch_count} complete with {len(meta_list)} metas, putting to out_q"
                    )
                self.out_q.put(meta_list)
                if __debug__:
                    logger.debug(
                        f"MetaBatcher: batch #{batch_count} put to out_q successfully"
                    )

        if __debug__:
            logger.debug(
                f"MetaBatcher: exiting logic loop, processed {batch_count} batches"
            )
