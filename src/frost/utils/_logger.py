import logging
import os


def _get_level_from_env() -> int:
    # Prefer LOG_LEVEL, fall back to APP_ENV
    lvl = os.getenv("LOG_LEVEL", "").upper()
    if lvl:
        return getattr(logging, lvl, logging.WARNING)

    env_str = os.getenv("APP_ENV", "production").lower()
    return logging.DEBUG if env_str == "dev" else logging.WARNING  # default WARNING in production


def init_logging(local_rank: int, output_dir: str | None = None) -> logging.Logger:
    level = _get_level_from_env()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"rank_{local_rank}.log")
    else:
        log_path = f"rank_{local_rank}.log"

    logging.basicConfig(
        level=level,
        filename=log_path,
        filemode="w",
        format="%(asctime)s - %(name)s - [%(threadName)s] - %(levelname)s - %(message)s",
        force=True,
    )

    logging.getLogger("torch").setLevel(logging.WARNING)

    return logging.getLogger(__name__)
