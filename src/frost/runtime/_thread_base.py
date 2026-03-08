import logging
import threading
import traceback
from typing import cast

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ThreadBase(threading.Thread):
    """Base class for non-CUDA daemon threads in the FROST runtime.

    Subclasses override ``logic()`` to implement their work loop. On any
    unhandled exception ``run()`` prints a formatted crash report including the
    class name, thread name, and distributed rank, sets the shared stop event,
    and calls ``os._exit(1)`` to hard-terminate the process group.

    Attributes:
        logger: Logger named after the thread.
        stop_ev: Shared event used to signal all threads to stop.
    """

    def __init__(
        self,
        name: str,
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(
            daemon=False,
            name=name,
        )
        self.logger = logging.getLogger(name)
        self.stop_ev = stop_ev

    def logic(self) -> None:
        """Main work loop to be implemented by subclasses."""
        pass

    def run(self) -> None:
        """Execute ``logic()`` and handle uncaught exceptions with a crash report."""
        try:
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


class StreamThreadBase(threading.Thread):
    """Base class for threads that execute their logic inside a dedicated CUDA stream.

    Similar crash-handling semantics to ``ThreadBase``: any uncaught exception
    triggers a crash report, sets the stop event, and calls ``os._exit(1)``.
    The CUDA stream is created at construction and activated via a context manager
    in ``run()``, so all CUDA operations in ``logic()`` are enqueued on the same
    stream.

    Attributes:
        stream: A dedicated ``torch.cuda.Stream`` for this thread.
        stop_ev: Shared event used to signal all threads to stop.
    """

    def __init__(
        self,
        name: str,
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(daemon=False, name=name)
        self.stream = cast(torch.cuda.Stream, torch.cuda.Stream())
        self.stop_ev = stop_ev

    def logic(self) -> None:
        """Main work loop to be implemented by subclasses."""
        pass

    def run(self) -> None:
        """Execute ``logic()`` inside the thread's CUDA stream, with crash handling."""
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
