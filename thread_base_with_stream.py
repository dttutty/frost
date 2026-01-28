import logging
import threading
import traceback

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


class StreamThreadBase(threading.Thread):
    def __init__(
        self,
        name: str,
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(daemon=False, name=name)
        self.stream: torch.cuda.Stream = torch.cuda.Stream()
        self.stop_ev = stop_ev

    def logic(self) -> None:
        pass

    def start(self) -> None:
        try:
            with torch.cuda.stream(self.stream):
                self.logic()

        except Exception:
            cls_name = self.__class__.__name__
            thread_name = threading.current_thread().name
            rank: str | int = "Unknown"
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
