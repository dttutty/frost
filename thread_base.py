import logging
import threading
import traceback

import torch.distributed as dist


logger = logging.getLogger(__name__)


class ThreadBase(threading.Thread):
    def __init__(
        self,
        name: str,
    ):
        super().__init__(daemon=True, name=name)

    def logic(self):
        pass

    def run(self):
        try:
            self.logic()

        except Exception:
            cls_name = self.__class__.__name__
            thread_name = threading.current_thread().name
            rank = "Unknown"
            try:
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
            except Exception:
                pass
            import os
            import sys

            sys.stderr.write(
                f"\n!!! [{cls_name}] ({thread_name}) Crashed on Rank {rank} !!!\n"
            )

            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

            self.stop_ev.set()
            os._exit(1)
