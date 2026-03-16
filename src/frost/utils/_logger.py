import logging
import os
from collections.abc import Callable
from typing import Any


def _get_level_from_env() -> int:
    # Prefer LOG_LEVEL, fall back to APP_ENV
    lvl = os.getenv("LOG_LEVEL", "").upper()
    if lvl:
        return getattr(logging, lvl, logging.WARNING)

    env_str = os.getenv("APP_ENV", "production").lower()
    return logging.DEBUG if env_str == "dev" else logging.WARNING  # default WARNING in production


def init_logging(local_rank: int, is_host: bool) -> logging.Logger:
    level = _get_level_from_env()

    logging.basicConfig(
        level=level,
        filename=f"rank_{local_rank}.log",
        filemode="w",
        format="%(asctime)s - %(name)s - [%(threadName)s] - %(levelname)s - %(message)s",
        force=True,
    )

    # Only the host process logs to the console, at the configured level
    # if is_host:
    #     console = logging.StreamHandler()
    #     console.setLevel(level)
    #     console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    #     root.addHandler(console)

    logging.getLogger("torch").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


class FastLogger:
    """
    A high-performance logger wrapper.
    In python -O mode, the debug method is optimized to a no-op,
    completely avoiding f-string concatenation and argument evaluation overhead.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, msg_provider: Callable[[], str]) -> None:
        """
        Uses a lambda for lazy evaluation.
        The lambda is only called when not in optimized mode and the log level permits it.
        """
        if __debug__ and self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(msg_provider())

    # Kept for compatibility with callers that pass a plain string rather than a lambda
    def debug_raw(self, msg: str) -> None:
        if __debug__:
            self._logger.debug(msg)

    def info(self, msg: Any, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: Any, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)


# Global helper; replace with a custom implementation if needed
def get_logger(name: str) -> FastLogger:
    return FastLogger(logging.getLogger(name))
