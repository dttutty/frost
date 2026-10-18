#!/usr/bin/env python3
"""Run a matched LASTFM/TGN AP comparison across TGL and FROST GPU counts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
TGL_ROOT = REPO_ROOT / "third_party" / "tgl"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "comparisons" / "tgl_1gpu_vs_frost_2gpu_ap_20260729"
)
SYSTEMS = ("tgl_serial_1gpu", "tgl_1gpu", "frost_2gpu", "frost_1gpu")

FROST_BEST_RE = re.compile(
    r"Best Val AP: (?P<val>[0-9.]+) @ Epoch (?P<epoch>\d+) .*?Test AP: (?P<test>[0-9.]+)"
)
TGL_EPOCH_RE = re.compile(r"^Epoch (?P<epoch>\d+):$", re.MULTILINE)
TGL_VAL_RE = re.compile(r"^\[VAL\] Global AP: (?P<val>[0-9.]+)", re.MULTILINE)
TGL_VAL_PRECISE_RE = re.compile(
    r"^\s*train loss:.*?\sval ap:(?P<val>[0-9.]+)", re.MULTILINE
)
TGL_BEST_RE = re.compile(
    r"Best model at epoch (?P<epoch>\d+)\.\s+test ap:(?P<test>[0-9.]+)"
)
TGL_SERIAL_EPOCH_METRICS_RE = re.compile(
    r"^\s*train loss:.*?\sval ap:(?P<val>[0-9.]+).*?\stest ap:(?P<test>[0-9.]+)",
    re.MULTILINE,
)
TGL_SERIAL_BEST_RE = re.compile(
    r"Best model at epoch (?P<epoch>\d+) had val AP: (?P<val>[0-9.]+)"
    r"\s+test AP:(?P<test>[0-9.]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Result:
    system: str
    n_gpu: int
    seed: int
    global_batch_size: int
    per_gpu_batch_size: int
    epochs: int
    best_epoch: int
    val_ap: float
    test_ap: float
    log_path: str


def write_tgl_config(destination: Path, *, epochs: int, batch_size: int) -> None:
    source = TGL_ROOT / "config" / "frost_parity" / "TGN.yml"
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["train"][0]["epoch"] = epochs
    config["train"][0]["batch_size"] = batch_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def parse_frost(log_text: str) -> tuple[int, float, float]:
    matches = list(FROST_BEST_RE.finditer(log_text))
    if not matches:
        raise ValueError("Could not parse FROST best AP summary")
    match = matches[-1]
    return int(match["epoch"]), float(match["val"]), float(match["test"])


def parse_tgl(log_text: str) -> tuple[int, float, float]:
    epochs = [int(match["epoch"]) for match in TGL_EPOCH_RE.finditer(log_text)]
    val_aps_rounded = [float(match["val"]) for match in TGL_VAL_RE.finditer(log_text)]
    val_aps = [float(match["val"]) for match in TGL_VAL_PRECISE_RE.finditer(log_text)]
    best_matches = list(TGL_BEST_RE.finditer(log_text))
    if (
        not epochs
        or len(epochs) != len(val_aps_rounded)
        or len(epochs) != len(val_aps)
        or not best_matches
    ):
        raise ValueError(
            "Could not parse complete TGL epoch/global-AP records: "
            f"epochs={len(epochs)}, rounded_val={len(val_aps_rounded)}, "
            f"precise_val={len(val_aps)}, best={len(best_matches)}"
        )
    best_match = best_matches[-1]
    best_epoch_zero_based = int(best_match["epoch"])
    if best_epoch_zero_based not in epochs:
        raise ValueError(
            f"TGL best epoch {best_epoch_zero_based} is absent from epoch logs"
        )
    best_idx = epochs.index(best_epoch_zero_based)
    return best_epoch_zero_based + 1, val_aps[best_idx], float(best_match["test"])


def parse_tgl_serial(log_text: str) -> tuple[int, float, float]:
    epochs = [int(match["epoch"]) for match in TGL_EPOCH_RE.finditer(log_text)]
    epoch_metrics = list(TGL_SERIAL_EPOCH_METRICS_RE.finditer(log_text))
    best_matches = list(TGL_SERIAL_BEST_RE.finditer(log_text))
    if not epochs or len(epochs) != len(epoch_metrics) or not best_matches:
        raise ValueError(
            "Could not parse complete serial TGL records: "
            f"epochs={len(epochs)}, metrics={len(epoch_metrics)}, "
            f"best={len(best_matches)}"
        )
    best_match = best_matches[-1]
    best_epoch_zero_based = int(best_match["epoch"])
    if best_epoch_zero_based not in epochs:
        raise ValueError(
            f"Serial TGL best epoch {best_epoch_zero_based} is absent from epoch logs"
        )
    best_idx = epochs.index(best_epoch_zero_based)
    return (
        best_epoch_zero_based + 1,
        float(epoch_metrics[best_idx]["val"]),
        float(best_match["test"]),
    )


def run_and_tee(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def command_for(
    system: str,
    *,
    seed: int,
    epochs: int,
    global_batch_size: int,
    tgl_config: Path,
    master_port: int,
) -> tuple[list[str], str, int]:
    if system == "tgl_serial_1gpu":
        return (
            [
                "uv",
                "run",
                "--project",
                str(TGL_ROOT),
                "python",
                str(TGL_ROOT / "train.py"),
                "--data",
                "LASTFM",
                "--config",
                str(tgl_config),
                "--gpu",
                "0",
                "--seed",
                str(seed),
            ],
            "0",
            1,
        )

    if system == "tgl_1gpu":
        return (
            [
                "uv",
                "run",
                "--project",
                str(TGL_ROOT),
                "python",
                str(TGL_ROOT / "train_dist.py"),
                "--dataset",
                "lastfm",
                "--config",
                str(tgl_config),
                "--num_gpus",
                "1",
                "--seed",
                str(seed),
                "--omp_num_threads",
                "8",
                "--rnd_edim",
                "0",
                "--rnd_ndim",
                "0",
            ],
            "0",
            1,
        )

    n_gpu = 2 if system == "frost_2gpu" else 1
    return (
        [
            "uv",
            "run",
            "python",
            "-m",
            "frost_bench.cli.train",
            "tgn",
            "--nproc-per-node",
            str(n_gpu + 1),
            "--master-port",
            str(master_port),
            "dataset=LASTFM",
            f"epoch={epochs}",
            f"macro_batch_size={global_batch_size}",
            f"seed={seed}",
            "stable_mode=true",
            "compile=false",
            "compute.tf32=false",
            "tqdm=false",
        ],
        "0,1" if n_gpu == 2 else "0",
        n_gpu,
    )


def summarize(results: list[Result]) -> dict[str, object]:
    grouped = {
        system: sorted(
            (row for row in results if row.system == system), key=lambda row: row.seed
        )
        for system in SYSTEMS
    }
    summary: dict[str, object] = {"systems": {}}
    for system, rows in grouped.items():
        values = [row.test_ap for row in rows]
        if not values:
            continue
        summary["systems"][system] = {
            "n": len(values),
            "mean_test_ap": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }

    comparisons: dict[str, object] = {}
    comparison_pairs = (
        ("tgl_serial_1gpu", "frost_2gpu"),
        ("tgl_serial_1gpu", "frost_1gpu"),
        ("tgl_serial_1gpu", "tgl_1gpu"),
        ("tgl_1gpu", "frost_2gpu"),
        ("tgl_1gpu", "frost_1gpu"),
        ("frost_2gpu", "frost_1gpu"),
    )
    for left_name, right_name in comparison_pairs:
        left = {row.seed: row.test_ap for row in grouped[left_name]}
        right = {row.seed: row.test_ap for row in grouped[right_name]}
        seeds = sorted(left.keys() & right.keys())
        if not seeds:
            continue
        diffs = [left[seed] - right[seed] for seed in seeds]
        mean_diff = statistics.fmean(diffs)
        if len(diffs) > 1:
            sem = stats.sem(diffs)
            ci_low, ci_high = stats.t.interval(
                0.95, df=len(diffs) - 1, loc=mean_diff, scale=sem
            )
            paired_p = float(
                stats.ttest_rel(
                    [left[seed] for seed in seeds],
                    [right[seed] for seed in seeds],
                ).pvalue
            )
        else:
            ci_low = ci_high = mean_diff
            paired_p = float("nan")
        comparisons[f"{left_name}_minus_{right_name}"] = {
            "seeds": seeds,
            "differences": diffs,
            "mean_difference": mean_diff,
            "ci95": [float(ci_low), float(ci_high)],
            "paired_p": paired_p,
        }
    summary["comparisons"] = comparisons
    return summary


def write_outputs(
    output_dir: Path, results: list[Result], summary: dict[str, object]
) -> None:
    result_path = output_dir / "results.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        "# TGL 1GPU vs FROST 2GPU AP summary",
        "",
        "| system | n | test AP values | mean | sample SD |",
        "|---|---:|---|---:|---:|",
    ]
    for system, values in summary["systems"].items():
        rendered = ", ".join(f"{value:.6f}" for value in values["values"])
        lines.append(
            f"| {system} | {values['n']} | {rendered} | "
            f"{values['mean_test_ap']:.6f} | {values['sample_sd']:.6f} |"
        )
    lines.extend(["", "## Paired differences", ""])
    for name, values in summary["comparisons"].items():
        lines.append(
            f"- `{name}`: mean {values['mean_difference']:+.6f}, "
            f"95% CI [{values['ci95'][0]:+.6f}, {values['ci95'][1]:+.6f}], "
            f"paired p={values['paired_p']:.6g}."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--global-batch-size", type=int, default=4000)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.global_batch_size < 1:
        parser.error("epochs and global batch size must be positive")
    if args.global_batch_size % 2 and "frost_2gpu" in args.systems:
        parser.error("FROST 2GPU requires a global batch size divisible by 2")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tgl_config = output_dir / "tgl_tgn_lastfm.yml"
    write_tgl_config(tgl_config, epochs=args.epochs, batch_size=args.global_batch_size)

    results: list[Result] = []
    for seed in args.seeds:
        for system_idx, system in enumerate(args.systems):
            log_path = output_dir / "logs" / f"{system}_seed{seed}.log"
            command, visible_devices, n_gpu = command_for(
                system,
                seed=seed,
                epochs=args.epochs,
                global_batch_size=args.global_batch_size,
                tgl_config=tgl_config,
                master_port=30200 + seed * 10 + system_idx,
            )
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": visible_devices,
                    # Keep FROST's always-on benchmark telemetry out of the AP logs.
                    # This caps only metric collection; it does not stop training.
                    "FROST_THROUGHPUT_MEASURE_BATCHES": "1",
                    "OMP_NUM_THREADS": "8",
                    "MKL_NUM_THREADS": "8",
                }
            )
            if args.no_resume or not log_path.is_file():
                run_and_tee(command, env=env, log_path=log_path)
            log_text = log_path.read_text(encoding="utf-8")
            if system == "tgl_serial_1gpu":
                best_epoch, val_ap, test_ap = parse_tgl_serial(log_text)
            elif system == "tgl_1gpu":
                best_epoch, val_ap, test_ap = parse_tgl(log_text)
            else:
                best_epoch, val_ap, test_ap = parse_frost(log_text)
            results.append(
                Result(
                    system=system,
                    n_gpu=n_gpu,
                    seed=seed,
                    global_batch_size=args.global_batch_size,
                    per_gpu_batch_size=args.global_batch_size // n_gpu,
                    epochs=args.epochs,
                    best_epoch=best_epoch,
                    val_ap=val_ap,
                    test_ap=test_ap,
                    log_path=str(log_path.relative_to(REPO_ROOT)),
                )
            )
            write_outputs(output_dir, results, summarize(results))

    summary = summarize(results)
    write_outputs(output_dir, results, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
