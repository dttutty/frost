#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously follow the newest Hydra rank log under outputs/<date>/<time>/."
    )
    parser.add_argument("ranks", nargs="+", type=int, help="Rank ids to follow.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Root Hydra outputs directory.",
    )
    parser.add_argument(
        "--poll-secs",
        type=float,
        default=0.5,
        help="Polling interval while waiting for a new log file.",
    )
    parser.add_argument(
        "--newer-than-start",
        action="store_true",
        help="Ignore log files older than the watcher start time.",
    )
    return parser.parse_args()


def latest_rank_log(
    outputs_dir: Path, rank: int, min_mtime: float | None
) -> Path | None:
    pattern = f"*/*/rank_{rank}.log"
    candidates: list[tuple[float, Path]] = []
    for path in outputs_dir.glob(pattern):
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if min_mtime is not None and mtime < min_mtime:
            continue
        candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return candidates[-1][1]


def stop_tail(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1)


def main() -> int:
    args = parse_args()
    min_mtime = time.time() if args.newer_than_start else None
    ranks = sorted(set(args.ranks))
    current_logs: dict[int, Path | None] = {rank: None for rank in ranks}
    tail_procs: dict[int, subprocess.Popen[bytes] | None] = {
        rank: None for rank in ranks
    }

    joined_ranks = ",".join(str(rank) for rank in ranks)
    print(f"READY rank watchers started: {joined_ranks}")
    sys.stdout.flush()

    try:
        while True:
            for rank in ranks:
                newest_log = latest_rank_log(args.outputs_dir, rank, min_mtime)
                if newest_log != current_logs[rank]:
                    stop_tail(tail_procs[rank])
                    tail_procs[rank] = None
                    current_logs[rank] = newest_log
                    if newest_log is not None:
                        print(f"FOLLOW rank {rank}: {newest_log}")
                        sys.stdout.flush()
                        tail_procs[rank] = subprocess.Popen(
                            ["tail", "-n", "+1", "-F", str(newest_log)],
                            cwd=REPO_ROOT,
                        )
                if tail_procs[rank] is not None and tail_procs[rank].poll() is not None:
                    tail_procs[rank] = None
                    current_logs[rank] = None
            time.sleep(args.poll_secs)
    except KeyboardInterrupt:
        return 0
    finally:
        for tail_proc in tail_procs.values():
            stop_tail(tail_proc)


if __name__ == "__main__":
    raise SystemExit(main())
