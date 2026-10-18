#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_SCRIPT = SCRIPT_DIR / "plot_throughput_frost_vs_dyglib.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor a running FROST vs DyGLib throughput sweep and auto-render artifacts."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--expected-total", type=int, default=0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def count_rows(results_tsv: Path) -> int:
    if not results_tsv.exists():
        return 0
    with results_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return sum(1 for row in reader if (row.get("model") or "").strip())


def load_tail_rows(results_tsv: Path, limit: int = 10) -> list[dict[str, str]]:
    if not results_tsv.exists():
        return []
    with results_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if (row.get("model") or "").strip()]
    return rows[-limit:]


def parse_expected_total(launcher_log: Path) -> int:
    if not launcher_log.exists():
        return 0
    text = launcher_log.read_text(encoding="utf-8", errors="ignore")

    def count_items(label: str) -> int:
        match = re.search(rf"^{re.escape(label)}:\s+(.+)$", text, flags=re.MULTILINE)
        if not match:
            return 0
        payload = match.group(1).strip()
        return len([item for item in payload.split() if item])

    models = count_items("Models")
    datasets = count_items("Datasets")
    seeds = count_items("Seeds")
    batch_sizes = count_items("Macro batch sizes")
    if 0 in (models, datasets, seeds, batch_sizes):
        return 0
    return models * datasets * seeds * batch_sizes


def run_plotter(run_dir: Path, expected_total: int) -> None:
    command = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--run-dir",
        str(run_dir),
    ]
    if expected_total > 0:
        command.extend(["--expected-total", str(expected_total)])
    subprocess.run(command, check=True, cwd=SCRIPT_DIR.parent)


def write_progress(
    run_dir: Path,
    pid: int,
    expected_total: int,
    status: str,
    rows: list[dict[str, str]],
) -> None:
    completed = len(rows)
    successful = sum(
        1
        for row in rows
        if row.get("frost_status") == "ok" and row.get("dyglib_status") == "ok"
    )
    failed = completed - successful
    remaining = max(expected_total - completed, 0) if expected_total > 0 else 0
    completion_pct = (completed / expected_total * 100.0) if expected_total > 0 else 0.0

    lines = [
        f"# Sweep Monitor ({utc_now()})",
        "",
        "## Status",
        f"- Run dir: `{run_dir}`",
        f"- PID: `{pid}`",
        f"- State: `{status}`",
        f"- Completed combos: `{completed}`"
        + (f" / `{expected_total}` (`{completion_pct:.1f}%`)" if expected_total > 0 else ""),
        f"- Successful combos: `{successful}`",
        f"- Failed combos: `{failed}`",
    ]
    if expected_total > 0:
        lines.append(f"- Remaining combos: `{remaining}`")
    lines.extend(
        [
            "",
            "## Outputs",
            f"- [results.tsv]({run_dir / 'results.tsv'})",
            f"- [summary.tsv]({run_dir / 'summary.tsv'})",
            f"- [report.md]({run_dir / 'report.md'})",
            f"- [speedup_by_dataset.png]({run_dir / 'speedup_by_dataset.png'})",
            f"- [throughput_scatter.png]({run_dir / 'throughput_scatter.png'})",
        ]
    )

    failures = [
        row
        for row in rows
        if not (
            row.get("frost_status") == "ok" and row.get("dyglib_status") == "ok"
        )
    ]
    lines.extend(["", "## Recent Combos"])
    if rows:
        for row in rows[-10:]:
            lines.append(
                "- "
                + f"{row.get('model', '')} / {row.get('dataset', '')} / mb{row.get('macro_batch_size', '')} / seed{row.get('seed', '')}: "
                + f"FROST=`{row.get('frost_status', '')}`, DyGLib=`{row.get('dyglib_status', '')}`, "
                + f"speedup=`{row.get('speedup_frost_over_dyglib', '')}`"
            )
    else:
        lines.append("- No completed combo yet.")

    lines.extend(["", "## Failures"])
    if failures:
        for row in failures[-10:]:
            lines.append(
                "- "
                + f"{row.get('model', '')} / {row.get('dataset', '')} / mb{row.get('macro_batch_size', '')} / seed{row.get('seed', '')}: "
                + f"FROST=`{row.get('frost_status', '')}`, DyGLib=`{row.get('dyglib_status', '')}`"
            )
    else:
        lines.append("- No failures recorded so far.")

    (run_dir / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    launcher_log = run_dir / "launcher.log"
    results_tsv = run_dir / "results.tsv"
    monitor_log = run_dir / "monitor.log"

    expected_total = args.expected_total or parse_expected_total(launcher_log)
    last_completed = -1

    with monitor_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] monitor started for pid={args.pid}\n")
        handle.flush()

        while True:
            alive = pid_alive(args.pid)
            completed = count_rows(results_tsv)
            rows = load_tail_rows(results_tsv, limit=completed)
            should_refresh = completed != last_completed or not alive
            status = "running" if alive else "finished"

            if should_refresh:
                run_plotter(run_dir, expected_total)
                write_progress(run_dir, args.pid, expected_total, status, rows)
                handle.write(
                    f"[{utc_now()}] status={status} completed={completed} expected={expected_total}\n"
                )
                handle.flush()
                last_completed = completed

            if not alive:
                break

            time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
