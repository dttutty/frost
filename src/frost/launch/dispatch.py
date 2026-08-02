from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from frost.runtime._thread_base import destroy_process_group
from frost.shm._core import cleanup_all_shm


logger = logging.getLogger(__name__)


class HostRuntime(Protocol):
    def run(self) -> None: ...

    def shutdown(self, graceful: bool, join_timeout_s: float = 5.0) -> None: ...


class WorkerRuntime(Protocol):
    def run(self) -> None: ...

    def cleanup(self, graceful: bool = True, join_timeout_s: float = 5.0) -> None: ...


def dispatch_host(factory: Callable[[], HostRuntime]) -> None:
    host: HostRuntime | None = None
    failed = True
    try:
        host = factory()
        host.run()
        failed = False
    except BaseException:
        # Runtime cleanup may need to call os._exit when a transport thread is
        # stuck.  Record the originating exception before cleanup so that the
        # forced exit cannot erase the only useful traceback.
        logger.exception("Host runtime failed before cleanup")
        raise
    finally:
        if host is not None:
            host.shutdown(graceful=not failed)
        else:
            destroy_process_group(sync=False, logger_=logger)
        cleanup_all_shm(is_host=True)


def dispatch_worker(factory: Callable[[], WorkerRuntime]) -> None:
    worker: WorkerRuntime | None = None
    failed = True
    try:
        worker = factory()
        worker.run()
        failed = False
    except BaseException:
        logger.exception("Worker runtime failed before cleanup")
        raise
    finally:
        if worker is not None:
            worker.cleanup(graceful=not failed)
        else:
            destroy_process_group(sync=False, logger_=logger)
        cleanup_all_shm(is_host=False)
