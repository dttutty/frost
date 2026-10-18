#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import io
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from frost_bench.io import run_command


STOP_REQUESTED = False
COMMAND_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class GpuProcess:
    gpu_index: str
    gpu_uuid: str
    pid: str
    username: str
    process_name: str
    used_memory_mib: str


def handle_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def query_gpu_uuid_to_index() -> dict[str, str]:
    text = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    mapping: dict[str, str] = {}
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) != 2:
            continue
        gpu_index, gpu_uuid = [col.strip() for col in row]
        mapping[gpu_uuid] = gpu_index
    return mapping


def resolve_username(pid: str) -> str:
    try:
        text = run_command(["ps", "-o", "user=", "-p", pid])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return text.strip() or "unknown"


def query_gpu_processes(uuid_to_index: dict[str, str]) -> list[GpuProcess]:
    try:
        text = run_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    processes: list[GpuProcess] = []
    if not text:
        return processes

    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) != 4:
            continue
        gpu_uuid, pid, process_name, used_memory_mib = [col.strip() for col in row]
        processes.append(
            GpuProcess(
                gpu_index=uuid_to_index.get(gpu_uuid, "?"),
                gpu_uuid=gpu_uuid,
                pid=pid,
                username=resolve_username(pid),
                process_name=process_name,
                used_memory_mib=used_memory_mib,
            )
        )
    return processes


def identifier_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdecimal():
        return (0, int(value))
    return (1, value)


def write_snapshot(
    output_path: Path,
    *,
    current_user: str,
    processes: list[GpuProcess],
) -> None:
    foreign = [
        proc for proc in processes if proc.username not in {current_user, "unknown"}
    ]
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] current_user={current_user}\n")
        if not processes:
            handle.write("status=no_compute_processes\n\n")
            return

        if foreign:
            handle.write(
                "status=foreign_gpu_users_detected "
                f"count={len(foreign)} users={','.join(sorted({proc.username for proc in foreign}))}\n"
            )
        else:
            handle.write("status=no_foreign_gpu_users_detected\n")

        for proc in sorted(
            processes,
            key=lambda item: (
                identifier_sort_key(item.gpu_index),
                identifier_sort_key(item.pid),
            ),
        ):
            handle.write(
                "gpu="
                f"{proc.gpu_index} user={proc.username} pid={proc.pid} "
                f"mem_mib={proc.used_memory_mib} name={proc.process_name} uuid={proc.gpu_uuid}\n"
            )
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodically log GPU compute users and flag non-owner activity."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--user", default=os.environ.get("USER", "unknown"))
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval_seconds < 1:
        raise ValueError("--interval-seconds must be >= 1")

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    uuid_to_index = query_gpu_uuid_to_index()

    while not STOP_REQUESTED:
        try:
            processes = query_gpu_processes(uuid_to_index)
            write_snapshot(args.output, current_user=args.user, processes=processes)
        except Exception as exc:  # pragma: no cover - defensive monitor logging
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[{utc_now()}] status=monitor_error error={type(exc).__name__}: {exc}\n\n"
                )
        if args.once:
            return 0
        for _ in range(args.interval_seconds):
            if STOP_REQUESTED:
                return 0
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
