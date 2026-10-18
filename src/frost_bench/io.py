from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from frost_bench.spec import PreparedRun


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_rows_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_command(
    command: Sequence[str],
    env: Mapping[str, str],
    log_path: Path,
    cwd: Path | None = None,
) -> None:
    ensure_dir(log_path.parent)
    merged_env = os.environ.copy()
    merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            list(command),
            cwd=cwd,
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_prepared(prepared: PreparedRun, log_path: Path) -> None:
    try:
        run_command(prepared.command, prepared.env, log_path, prepared.cwd)
    finally:
        cleanup_prepared(prepared)


def cleanup_prepared(prepared: PreparedRun) -> None:
    """Remove backend-owned transient files after its processes have exited."""
    for path in prepared.cleanup_paths:
        path.unlink(missing_ok=True)
