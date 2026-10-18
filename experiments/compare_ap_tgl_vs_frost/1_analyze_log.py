#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

FILENAME_PATTERN = re.compile(
    r"^(?P<experiment>.+?)__(?P<dataset>.+?)__memdim(?P<mem_dim>\d+)__ep(?P<epochs>\d+)__bs(?P<batch_size>\d+)__nproc(?P<nproc>\d+)__seed(?P<seed>-?\d+)__rep(?P<repeat>\d+)\.log$"
)
EPOCH_TEST_AP_PATTERN = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)\].*?Test AP:\s*(?P<test_ap>[0-9]*\.?[0-9]+)",
    flags=re.IGNORECASE,
)
FINAL_RESULT_PATTERN = re.compile(
    r"Best Val AP:\s*(?P<best_val_ap>[0-9]*\.?[0-9]+)\s*@\s*Epoch\s*(?P<best_epoch>\d+)"
    r"(?:\s*\|\s*Val AUC:\s*(?P<best_val_auc>[0-9]*\.?[0-9]+))?"
    r"\s*\|\s*Test AP:\s*(?P<test_ap>[0-9]*\.?[0-9]+)"
    r"(?:\s*\|\s*Test AUC:\s*(?P<test_auc>[0-9]*\.?[0-9]+))?",
    flags=re.IGNORECASE,
)


def parse_one_log(
    log_path: Path,
    log_dir: Path,
    *,
    tail_start_epoch: int,
    exclude_zero_tail_test_ap: bool,
) -> dict[str, object] | None:
    match = FILENAME_PATTERN.match(log_path.name)
    if match is None:
        print(f"Skipping unrecognized file: {log_path}")
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    epoch_matches = [
        (
            int(epoch_match.group("epoch")),
            float(epoch_match.group("test_ap")),
        )
        for epoch_match in EPOCH_TEST_AP_PATTERN.finditer(text)
    ]
    result_matches = list(FINAL_RESULT_PATTERN.finditer(text))
    if not result_matches and not epoch_matches:
        print(f"Missing epoch AP lines and final result line in: {log_path}")
        return None

    tail_test_ap_values = [
        test_ap
        for epoch, test_ap in epoch_matches
        if epoch >= tail_start_epoch
        and (not exclude_zero_tail_test_ap or test_ap != 0.0)
    ]
    tail_epochs = [epoch for epoch, _ in epoch_matches if epoch >= tail_start_epoch]
    final_match = result_matches[-1] if result_matches else None
    best_val_auc = final_match.group("best_val_auc") if final_match else None
    test_auc = final_match.group("test_auc") if final_match else None
    meta = match.groupdict()
    parent_rel = log_path.parent.relative_to(log_dir)
    run_tag = "." if str(parent_rel) == "." else str(parent_rel)

    return {
        "run_tag": run_tag,
        "experiment": meta["experiment"],
        "dataset": meta["dataset"],
        "mem_dim": int(meta["mem_dim"]),
        "epochs": int(meta["epochs"]),
        "batch_size": int(meta["batch_size"]),
        "nproc_per_node": int(meta["nproc"]),
        "num_gpus": int(meta["nproc"]) - 1,
        "seed": int(meta["seed"]),
        "repeat": int(meta["repeat"]),
        "best_epoch": int(final_match.group("best_epoch")) if final_match else None,
        "best_val_ap": float(final_match.group("best_val_ap")) if final_match else None,
        "best_val_auc": float(best_val_auc) if best_val_auc is not None else None,
        "test_ap": float(final_match.group("test_ap")) if final_match else None,
        "test_auc": float(test_auc) if test_auc is not None else None,
        "tail_test_ap_start_epoch": tail_start_epoch,
        "tail_test_ap_points": len(tail_test_ap_values),
        "tail_test_ap_first_epoch": tail_epochs[0] if tail_epochs else None,
        "tail_test_ap_last_epoch": tail_epochs[-1] if tail_epochs else None,
        "tail_test_ap_mean": (
            statistics.mean(tail_test_ap_values) if tail_test_ap_values else None
        ),
        "tail_test_ap_sum": sum(tail_test_ap_values) if tail_test_ap_values else None,
        "log_file": str(log_path.relative_to(log_dir)),
    }


def collect_complete_optional_metric(
    group: list[dict[str, object]], key: str
) -> list[float]:
    values = [row[key] for row in group if row[key] is not None]
    if len(values) != len(group):
        return []
    return [float(value) for value in values]


def collect_optional_metric(group: list[dict[str, object]], key: str) -> list[float]:
    return [float(value) for row in group if (value := row[key]) is not None]


def format_optional_mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{statistics.mean(values):.6f}"


def format_optional_std(values: list[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return "0.000000"
    return f"{statistics.stdev(values):.6f}"


def format_optional_min(values: list[float]) -> str:
    if not values:
        return ""
    return f"{min(values):.6f}"


def format_optional_max(values: list[float]) -> str:
    if not values:
        return ""
    return f"{max(values):.6f}"


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["run_tag"],
            row["experiment"],
            row["dataset"],
            row["mem_dim"],
            row["epochs"],
            row["batch_size"],
            row["nproc_per_node"],
            row["num_gpus"],
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: int(row["repeat"]))
        best_val_ap_values = collect_optional_metric(group, "best_val_ap")
        test_ap_values = collect_optional_metric(group, "test_ap")
        tail_test_ap_values = collect_optional_metric(group, "tail_test_ap_mean")
        tail_test_ap_sums = collect_optional_metric(group, "tail_test_ap_sum")
        tail_test_ap_points = [
            int(points) for row in group if (points := row["tail_test_ap_points"]) is not None
        ]
        pooled_tail_test_ap_mean = ""
        if tail_test_ap_sums and tail_test_ap_points and sum(tail_test_ap_points) > 0:
            pooled_tail_test_ap_mean = (
                f"{sum(tail_test_ap_sums) / sum(tail_test_ap_points):.6f}"
            )
        best_val_auc_values = collect_optional_metric(group, "best_val_auc")
        test_auc_values = collect_optional_metric(group, "test_auc")
        summary_rows.append(
            {
                "run_tag": key[0],
                "experiment": key[1],
                "dataset": key[2],
                "mem_dim": key[3],
                "epochs": key[4],
                "batch_size": key[5],
                "nproc_per_node": key[6],
                "num_gpus": key[7],
                "runs": len(group),
                "tail_test_ap_start_epoch": group[0]["tail_test_ap_start_epoch"],
                "mean_best_val_ap": format_optional_mean(best_val_ap_values),
                "std_best_val_ap": format_optional_std(best_val_ap_values),
                "mean_best_val_auc": format_optional_mean(best_val_auc_values),
                "std_best_val_auc": format_optional_std(best_val_auc_values),
                "mean_test_ap": format_optional_mean(test_ap_values),
                "std_test_ap": format_optional_std(test_ap_values),
                "min_test_ap": format_optional_min(test_ap_values),
                "max_test_ap": format_optional_max(test_ap_values),
                "mean_test_auc": format_optional_mean(test_auc_values),
                "std_test_auc": format_optional_std(test_auc_values),
                "mean_tail_test_ap": format_optional_mean(tail_test_ap_values),
                "std_tail_test_ap": format_optional_std(tail_test_ap_values),
                "min_tail_test_ap": format_optional_min(tail_test_ap_values),
                "max_tail_test_ap": format_optional_max(tail_test_ap_values),
                "pooled_tail_test_ap_points": sum(tail_test_ap_points),
                "pooled_tail_test_ap_mean": pooled_tail_test_ap_mean,
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_log_dir = script_dir / "logs"

    parser = argparse.ArgumentParser(
        description="Parse FROST compare_ap_tgl_vs_frost logs and summarize final AP/AUC."
    )
    parser.add_argument("--log_dir", type=Path, default=default_log_dir)
    parser.add_argument("--output", type=Path, default=None, help="Per-run parsed CSV path.")
    parser.add_argument(
        "--summary_output", type=Path, default=None, help="Aggregated summary CSV path."
    )
    parser.add_argument(
        "--tail_start_epoch",
        type=int,
        default=20,
        help="Start epoch (inclusive) for averaged tail Test AP statistics.",
    )
    parser.add_argument(
        "--include_zero_tail_test_ap",
        action="store_true",
        help="Keep zero-valued Test AP when computing tail averages.",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    output = args.output.resolve() if args.output else log_dir / "results.csv"
    summary_output = (
        args.summary_output.resolve() if args.summary_output else log_dir / "summary.csv"
    )

    if not log_dir.exists():
        raise SystemExit(f"log_dir does not exist: {log_dir}")

    log_files = sorted(path for path in log_dir.rglob("*.log") if path.is_file())
    if not log_files:
        print(f"No .log files found under {log_dir}")
        return

    rows: list[dict[str, object]] = []
    for log_file in log_files:
        row = parse_one_log(
            log_file,
            log_dir,
            tail_start_epoch=args.tail_start_epoch,
            exclude_zero_tail_test_ap=not args.include_zero_tail_test_ap,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid final AP results parsed from logs.")
        return

    rows.sort(
        key=lambda row: (
            str(row["run_tag"]),
            str(row["experiment"]),
            str(row["dataset"]),
            int(row["mem_dim"]),
            int(row["batch_size"]),
            int(row["seed"]),
            int(row["repeat"]),
        )
    )
    summary_rows = build_summary(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    write_csv(
        output,
        rows,
        [
            "run_tag",
            "experiment",
            "dataset",
            "mem_dim",
            "epochs",
            "batch_size",
            "nproc_per_node",
            "num_gpus",
            "seed",
            "repeat",
            "best_epoch",
            "best_val_ap",
            "best_val_auc",
            "test_ap",
            "test_auc",
            "tail_test_ap_start_epoch",
            "tail_test_ap_first_epoch",
            "tail_test_ap_last_epoch",
            "tail_test_ap_points",
            "tail_test_ap_mean",
            "tail_test_ap_sum",
            "log_file",
        ],
    )
    write_csv(
        summary_output,
        summary_rows,
        [
            "run_tag",
            "experiment",
            "dataset",
            "mem_dim",
            "epochs",
            "batch_size",
            "nproc_per_node",
            "num_gpus",
            "runs",
            "tail_test_ap_start_epoch",
            "mean_best_val_ap",
            "std_best_val_ap",
            "mean_best_val_auc",
            "std_best_val_auc",
            "mean_test_ap",
            "std_test_ap",
            "min_test_ap",
            "max_test_ap",
            "mean_test_auc",
            "std_test_auc",
            "mean_tail_test_ap",
            "std_tail_test_ap",
            "min_tail_test_ap",
            "max_tail_test_ap",
            "pooled_tail_test_ap_points",
            "pooled_tail_test_ap_mean",
        ],
    )

    print(f"Parsed {len(rows)} log(s)")
    print(f"Per-run results: {output}")
    print(f"Summary: {summary_output}")
    print("")
    print(
        "run_tag,experiment,dataset,mem_dim,batch_size,runs,tail_test_ap_start_epoch,mean_tail_test_ap,pooled_tail_test_ap_mean,pooled_tail_test_ap_points"
    )
    for row in summary_rows:
        print(
            f"{row['run_tag']},{row['experiment']},{row['dataset']},{row['mem_dim']},{row['batch_size']},"
            f"{row['runs']},{row['tail_test_ap_start_epoch']},{row['mean_tail_test_ap']},"
            f"{row['pooled_tail_test_ap_mean']},{row['pooled_tail_test_ap_points']}"
        )


if __name__ == "__main__":
    main()
