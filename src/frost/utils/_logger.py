import logging
import os
from collections.abc import Callable
from typing import Any


def _get_level_from_env() -> int:
    # 优先读 LOG_LEVEL，其次读 APP_ENV
    lvl = os.getenv("LOG_LEVEL", "").upper()
    if lvl:
        return getattr(logging, lvl, logging.WARNING)

    env_str = os.getenv("APP_ENV", "production").lower()
    return logging.DEBUG if env_str == "dev" else logging.WARNING  # 生产默认 WARNING


def init_logging(local_rank: int, is_host: bool) -> logging.Logger:
    level = _get_level_from_env()

    logging.basicConfig(
        level=level,
        filename=f"rank_{local_rank}.log",
        filemode="w",
        format="%(asctime)s - %(name)s - [%(threadName)s] - %(levelname)s - %(message)s",
        force=True,
    )

    root = logging.getLogger()

    # 只让 host 打到终端，且等级跟随 level
    # if is_host:
    #     console = logging.StreamHandler()
    #     console.setLevel(level)
    #     console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    #     root.addHandler(console)

    logging.getLogger("torch").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


class FastLogger:
    """
    一个高性能的 Logger 包装器。
    在 python -O 模式下，debug 方法会被优化成空操作，
    完全避免 f-string 的字符串拼接和参数计算开销。
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, msg_provider: Callable[[], str]) -> None:
        """
        使用 lambda 延迟求值。
        只有在非优化模式且日志级别允许时，才会执行 lambda 生成字符串。
        """
        if __debug__ and self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(msg_provider())

    # 为了兼容普通用法，也可以保留一个直接传 str 的方法，虽然这不如 lambda 快
    def debug_raw(self, msg: str) -> None:
        if __debug__:
            self._logger.debug(msg)

    def info(self, msg: Any, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: Any, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)


# 全局辅助函数，方便替换
def get_logger(name: str) -> FastLogger:
    return FastLogger(logging.getLogger(name))
