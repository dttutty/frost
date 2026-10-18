#!/usr/bin/env python3
"""Compute pooled Test AP means from experiment logs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

FILENAME_PATTERN = re.compile(
    r"^(?P<experiment>.+?)__(?P<dataset>.+?)__memdim(?P<mem_dim>\d+)__ep(?P<epochs>\d+)__bs(?P<batch_size>\d+)__nproc(?P<nproc>\d+)__seed(?P<seed>-?\d+)__rep(?P<repeat>\d+)\.log$"
)
EPOCH_TEST_AP_PATTERN = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)\].*?Test AP:\s*(?P<test_ap>[0-9]*\.?[0-9]+)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pooled Test AP means from .log files, grouped by config while "
            "ignoring seed/repeat identifiers."
        )
    )
    parser.add_argument("log_dir", type=Path, help="Directory containing .log files.")
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Only include epochs >= this value. Default: 1.",
    )
    parser.add_argument(
        "--exclude-zero-test-ap",
        action="store_true",
        help="Drop Test AP values equal to 0.0000 before pooling.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path to write in addition to stdout.",
    )
    return parser.parse_args()


def normalized_config_key(path: Path) -> str:
    stem = path.stem
    parts = [part for part in stem.split("__") if part]
    filtered = []
    for part in parts:
        lowered = part.lower()
        if lowered.startswith("seed"):
            continue
        if lowered.startswith("rep") or lowered.startswith("repeat"):
            continue
        filtered.append(part)
    return "__".join(filtered) if filtered else stem


def config_metadata(path: Path, log_dir: Path) -> dict[str, object]:
    match = FILENAME_PATTERN.match(path.name)
    parent_rel = path.parent.relative_to(log_dir)
    run_tag = "." if str(parent_rel) == "." else str(parent_rel)
    if match is None:
        return {
            "run_tag": run_tag,
            "config_key": normalized_config_key(path),
            "experiment": "",
            "dataset": "",
            "mem_dim": "",
            "epochs": "",
            "batch_size": "",
            "nproc_per_node": "",
            "num_gpus": "",
        }

    meta = match.groupdict()
    return {
        "run_tag": run_tag,
        "config_key": (
            f"{meta['experiment']}__{meta['dataset']}__memdim{meta['mem_dim']}"
            f"__ep{meta['epochs']}__bs{meta['batch_size']}__nproc{meta['nproc']}"
        ),
        "experiment": meta["experiment"],
        "dataset": meta["dataset"],
        "mem_dim": int(meta["mem_dim"]),
        "epochs": int(meta["epochs"]),
        "batch_size": int(meta["batch_size"]),
        "nproc_per_node": int(meta["nproc"]),
        "num_gpus": int(meta["nproc"]) - 1,
    }


def parse_epoch_test_ap(
    path: Path,
    *,
    start_epoch: int,
    exclude_zero_test_ap: bool,
) -> list[tuple[int, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: list[tuple[int, float]] = []
    for match in EPOCH_TEST_AP_PATTERN.finditer(text):
        epoch = int(match.group("epoch"))
        test_ap = float(match.group("test_ap"))
        if epoch < start_epoch:
            continue
        if exclude_zero_test_ap and test_ap == 0.0:
            continue
        values.append((epoch, test_ap))
    return values


def collect_rows(
    log_dir: Path,
    *,
    start_epoch: int,
    exclude_zero_test_ap: bool,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    log_files = sorted(path for path in log_dir.rglob("*.log") if path.is_file())

    for log_path in log_files:
        values = parse_epoch_test_ap(
            log_path,
            start_epoch=start_epoch,
            exclude_zero_test_ap=exclude_zero_test_ap,
        )
        if not values:
            continue

        meta = config_metadata(log_path, log_dir)
        key = (str(meta["run_tag"]), str(meta["config_key"]))
        row = grouped.get(key)
        if row is None:
            row = {
                **meta,
                "start_epoch": start_epoch,
                "exclude_zero_test_ap": exclude_zero_test_ap,
                "log_count": 0,
                "pooled_test_ap_points": 0,
                "pooled_test_ap_sum": 0.0,
                "first_observed_epoch": values[0][0],
                "last_observed_epoch": values[-1][0],
                "sample_log": str(log_path.relative_to(log_dir)),
            }
            grouped[key] = row

        row["log_count"] = int(row["log_count"]) + 1
        row["pooled_test_ap_points"] = int(row["pooled_test_ap_points"]) + len(values)
        row["pooled_test_ap_sum"] = float(row["pooled_test_ap_sum"]) + sum(
            test_ap for _, test_ap in values
        )
        row["first_observed_epoch"] = min(int(row["first_observed_epoch"]), values[0][0])
        row["last_observed_epoch"] = max(int(row["last_observed_epoch"]), values[-1][0])

    rows: list[dict[str, object]] = []
    for _, row in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        points = int(row["pooled_test_ap_points"])
        if points == 0:
            continue
        rows.append(
            {
                "run_tag": row["run_tag"],
                "config_key": row["config_key"],
                "experiment": row["experiment"],
                "dataset": row["dataset"],
                "mem_dim": row["mem_dim"],
                "epochs": row["epochs"],
                "batch_size": row["batch_size"],
                "nproc_per_node": row["nproc_per_node"],
                "num_gpus": row["num_gpus"],
                "start_epoch": row["start_epoch"],
                "exclude_zero_test_ap": row["exclude_zero_test_ap"],
                "log_count": row["log_count"],
                "pooled_test_ap_points": points,
                "pooled_test_ap_mean": f"{float(row['pooled_test_ap_sum']) / points:.6f}",
                "first_observed_epoch": row["first_observed_epoch"],
                "last_observed_epoch": row["last_observed_epoch"],
                "sample_log": row["sample_log"],
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.exists():
        raise SystemExit(f"log_dir does not exist: {log_dir}")

    rows = collect_rows(
        log_dir,
        start_epoch=args.start_epoch,
        exclude_zero_test_ap=args.exclude_zero_test_ap,
    )
    if not rows:
        print("No matching Test AP points found.", file=sys.stderr)
        return 1

    fieldnames = [
        "run_tag",
        "config_key",
        "experiment",
        "dataset",
        "mem_dim",
        "epochs",
        "batch_size",
        "nproc_per_node",
        "num_gpus",
        "start_epoch",
        "exclude_zero_test_ap",
        "log_count",
        "pooled_test_ap_points",
        "pooled_test_ap_mean",
        "first_observed_epoch",
        "last_observed_epoch",
        "sample_log",
    ]

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.output_csv is not None:
        write_rows(args.output_csv.expanduser().resolve(), rows, fieldnames)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
