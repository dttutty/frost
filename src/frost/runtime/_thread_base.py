import logging
import _thread
import threading
import traceback
from datetime import timedelta
from queue import Empty, Full, Queue
from typing import cast

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def put_shutdown_sentinel(q: Queue, item: object) -> bool:
    """Best-effort queue put used only during shutdown.

    If the queue is full, stale items are dropped to make room for the
    shutdown sentinel. This prevents teardown from hanging behind abandoned
    work items.
    """
    try:
        q.put_nowait(item)
        return True
    except Full:
        pass

    for _ in range(4):
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
            return True
        except Full:
            continue

    return False


def join_thread(
    thread: threading.Thread | None,
    timeout_s: float,
    logger_: logging.Logger,
) -> None:
    """Join one thread with a timeout and log if it stays alive."""
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        logger_.warning("Thread %s did not exit within %.1fs", thread.name, timeout_s)


def destroy_process_group(
    sync: bool, logger_: logging.Logger, timeout_s: float = 5.0
) -> None:
    """Best-effort distributed teardown that avoids hanging forever."""
    if not dist.is_available() or not dist.is_initialized():
        return

    if sync:
        try:
            if hasattr(dist, "monitored_barrier"):
                dist.monitored_barrier(timeout=timedelta(seconds=timeout_s))
            else:
                dist.barrier()
        except Exception as exc:
            logger_.warning("Distributed barrier during shutdown failed: %s", exc)

    try:
        dist.destroy_process_group()
    except Exception as exc:
        logger_.warning("destroy_process_group() during shutdown failed: %s", exc)


class ThreadBase(threading.Thread):
    """Base class for non-CUDA daemon threads in the FROST runtime.

    Subclasses override ``logic()`` to implement their work loop. On any
    unhandled exception ``run()`` prints a formatted crash report including the
    class name, thread name, and distributed rank, sets the shared stop event,
    and interrupts the main thread so top-level shutdown can run.

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
            if self.stop_ev.is_set():
                self.logger.debug(
                    "Thread %s exited with an exception during shutdown.",
                    threading.current_thread().name,
                    exc_info=True,
                )
                return

            cls_name = self.__class__.__name__
            thread_name = threading.current_thread().name
            rank: str | int = "Unknown"
            try:
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
            except Exception:
                pass
            import sys

            sys.stderr.write(
                f"\n!!! [{cls_name}] ({thread_name}) Crashed on Rank {rank} !!!\n"
            )

            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

            self.stop_ev.set()
            _thread.interrupt_main()


class StreamThreadBase(threading.Thread):
    """Base class for threads that execute their logic inside a dedicated CUDA stream.

    Similar crash-handling semantics to ``ThreadBase``: any uncaught exception
    triggers a crash report, sets the stop event, and interrupts the main
    thread so top-level shutdown can run.
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
            if self.stop_ev.is_set():
                logger.debug(
                    "CUDA thread %s exited with an exception during shutdown.",
                    threading.current_thread().name,
                    exc_info=True,
                )
                return

            cls_name = self.__class__.__name__
            thread_name = threading.current_thread().name
            rank: str | int = "Unknown"
            try:
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
            except Exception:
                pass
            import sys

            sys.stderr.write(
                f"\n!!! [{cls_name}] ({thread_name}) Crashed on Rank {rank} !!!\n"
            )

            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

            self.stop_ev.set()
            _thread.interrupt_main()
