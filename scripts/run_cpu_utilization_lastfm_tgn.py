#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from json import JSONDecoder
from pathlib import Path

from frost_bench.adapters.tgl import write_tgl_config
from frost_bench.io import ensure_dir, read_text, run_command, utc_timestamp
from frost_bench.metrics import parse_all_json_after_prefix
from frost_bench.workload import compute_tgl_max_train_steps

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments" / "cpu_utilization_lastfm_tgn"
TGL_ROOT = REPO_ROOT / "third_party" / "tgl"
TGL_CONFIG_SRC = TGL_ROOT / "config" / "frost_parity" / "TGN.yml"
DATASET = "LASTFM"
MODEL = "tgn"
DATASET_LOWER = DATASET.lower()
BATCH_SIZES = [600, 1200, 2000, 4000]
N_GPU = 2
SEED = 0
WARMUP_BATCHES = 5
MEASURE_BATCHES = 100
OMP_THREADS = "8"
MKL_THREADS = "8"
FROST_EPOCHS = 1
TGL_EPOCHS = 1


def parse_json_after_prefix(text: str, prefix: str) -> dict:
    decoder = JSONDecoder()
    for match in re.finditer(re.escape(prefix) + r"\s+", text):
        start = match.end()
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        return obj
    raise RuntimeError(f"Could not find `{prefix}` JSON payload in log.")


def parse_frost_bottleneck_eps(text: str) -> tuple[float, int]:
    matches = re.findall(
        r"TRAIN_THROUGHPUT_PER_GPU\s+rank=(\d+)\s+events_per_sec_per_gpu=([0-9eE+\-.]+).*?measured_batches=([0-9]+)",
        text,
    )
    if not matches:
        raise RuntimeError("Could not parse FROST throughput.")
    by_rank: dict[int, tuple[float, int]] = {}
    for rank_s, eps_s, measured_s in matches:
        by_rank[int(rank_s)] = (float(eps_s), int(measured_s))
    eps_values = [value[0] for value in by_rank.values()]
    measured_values = [value[1] for value in by_rank.values()]
    return min(eps_values), min(measured_values)


def parse_tgl_bottleneck_eps(text: str) -> tuple[float, int]:
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_E2E_SUMMARY")
    eps_values = [
        float(row["events_per_sec_per_gpu"])
        for row in rows
        if "events_per_sec_per_gpu" in row
    ]
    measured_values = [int(row["interval_count"]) for row in rows]
    if not eps_values:
        raise RuntimeError("Could not parse TGL throughput.")
    return min(eps_values), min(measured_values)


def parse_cpu_curve(text: str) -> dict:
    return parse_json_after_prefix(text, "APPENDIX_TRAIN_CPU_UTIL_CURVE")


def plot_curves(run_dir: Path, curves: list[dict]) -> None:
    if plt is None:
        return

    used_path = run_dir / "cpu_used_cores_curves.png"
    util_path = run_dir / "cpu_utilization_pct_curves.png"

    fig_used, axes_used = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig_util, axes_util = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes_used = axes_used.flatten()
    axes_util = axes_util.flatten()

    for idx, batch_size in enumerate(BATCH_SIZES):
        used_ax = axes_used[idx]
        util_ax = axes_util[idx]
        batch_rows = [row for row in curves if row["macro_batch_size"] == batch_size]
        for row in batch_rows:
            x = list(range(1, len(row["used_cores_curve"]) + 1))
            label = row["system"].upper()
            used_ax.plot(x, row["used_cores_curve"], label=label, linewidth=1.5)
            util_ax.plot(
                x,
                row["utilization_pct_curve"],
                label=label,
                linewidth=1.5,
            )
        used_ax.set_title(f"LASTFM + TGN, macro-batch={batch_size}")
        util_ax.set_title(f"LASTFM + TGN, macro-batch={batch_size}")
        used_ax.set_ylabel("CPU used cores")
        util_ax.set_ylabel("CPU utilization (%)")
        used_ax.grid(True, alpha=0.25)
        util_ax.grid(True, alpha=0.25)
        used_ax.legend()
        util_ax.legend()

    for ax in axes_used:
        ax.set_xlabel("Measured train interval index")
    for ax in axes_util:
        ax.set_xlabel("Measured train interval index")

    fig_used.suptitle("FROST vs TGL train_dist: CPU used cores over 100 train intervals")
    fig_util.suptitle(
        "FROST vs TGL train_dist: CPU utilization over 100 train intervals"
    )
    fig_used.tight_layout()
    fig_util.tight_layout()
    fig_used.savefig(used_path, dpi=180)
    fig_util.savefig(util_path, dpi=180)
    plt.close(fig_used)
    plt.close(fig_util)


def main() -> int:
    run_dir = OUTPUT_ROOT / f"cpu_util_lastfm_tgn_{utc_timestamp()}"
    cfg_dir = run_dir / "configs"
    log_dir = run_dir / "logs"
    ensure_dir(cfg_dir)
    ensure_dir(log_dir)

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    print(f"Writing outputs to {run_dir}")
    for idx, macro_batch_size in enumerate(BATCH_SIZES):
        per_gpu_batch_size = macro_batch_size // N_GPU
        master_port = 33300 + idx
        frost_log = log_dir / f"frost_lastfm_tgn_mb{macro_batch_size}.log"
        tgl_log = log_dir / f"tgl_train_dist_lastfm_tgn_mb{macro_batch_size}.log"
        tgl_cfg = cfg_dir / f"tgn_lastfm_mb{macro_batch_size}.yml"

        write_tgl_config(TGL_CONFIG_SRC, tgl_cfg, TGL_EPOCHS, per_gpu_batch_size)
        tgl_max_train_steps = compute_tgl_max_train_steps(
            DATASET,
            per_gpu_batch_size,
            N_GPU,
            WARMUP_BATCHES,
            MEASURE_BATCHES,
        )

        frost_env = os.environ.copy()
        frost_env.update(
            {
                "OMP_NUM_THREADS": OMP_THREADS,
                "MKL_NUM_THREADS": MKL_THREADS,
                "FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH": str(WARMUP_BATCHES),
                "FROST_THROUGHPUT_MEASURE_BATCHES": str(MEASURE_BATCHES),
                "FROST_CPU_UTIL_BENCH": "1",
            }
        )
        frost_cmd = [
            "uv",
            "run",
            "python",
            "-m",
        "frost_bench.cli.train",
            MODEL,
            "--nproc-per-node",
            str(N_GPU + 1),
            "--master-port",
            str(master_port),
            f"dataset={DATASET}",
            f"epoch={FROST_EPOCHS}",
            f"macro_batch_size={macro_batch_size}",
            "stable_mode=true",
            f"seed={SEED}",
            "compile=false",
            "compute.tf32=false",
            "tqdm=true",
        ]

        tgl_env = os.environ.copy()
        tgl_env.update(
            {
                "OMP_NUM_THREADS": OMP_THREADS,
                "MKL_NUM_THREADS": MKL_THREADS,
            }
        )
        tgl_cmd = [
            "uv",
            "run",
            "--project",
            str(TGL_ROOT),
            "python",
            str(TGL_ROOT / "train_dist.py"),
            "--dataset",
            DATASET_LOWER,
            "--config",
            str(tgl_cfg),
            "--num_gpus",
            str(N_GPU),
            "--seed",
            str(SEED),
            "--rnd_edim",
            "0",
            "--rnd_ndim",
            "0",
            "--appendix-train-e2e",
            "--cpu-util-benchmark",
            "--measure-start-epoch",
            "0",
            "--warmup-batches",
            str(WARMUP_BATCHES),
            "--measure-batches",
            str(MEASURE_BATCHES),
            "--max-train-steps",
            str(tgl_max_train_steps),
            "--tqdm",
        ]

        print(f"[FROST] LASTFM/TGN macro_batch_size={macro_batch_size}")
        run_command(frost_cmd, frost_env, frost_log)
        print(f"[TGL] LASTFM/TGN macro_batch_size={macro_batch_size}")
        run_command(tgl_cmd, tgl_env, tgl_log)

        frost_text = read_text(frost_log)
        tgl_text = read_text(tgl_log)
        frost_eps, frost_measured = parse_frost_bottleneck_eps(frost_text)
        tgl_eps, tgl_measured = parse_tgl_bottleneck_eps(tgl_text)
        frost_cpu = parse_cpu_curve(frost_text)
        tgl_cpu = parse_cpu_curve(tgl_text)

        summary_rows.append(
            {
                "dataset": DATASET,
                "model": MODEL,
                "macro_batch_size": macro_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "frost_events_per_sec_per_gpu": round(frost_eps, 6),
                "tgl_events_per_sec_per_gpu": round(tgl_eps, 6),
                "speedup_frost_over_tgl": round(frost_eps / tgl_eps, 6),
                "frost_measured_batches": frost_measured,
                "tgl_measured_batches": tgl_measured,
                "frost_cpu_used_cores_mean": round(
                    float(frost_cpu.get("used_cores_mean", 0.0)),
                    6,
                ),
                "tgl_cpu_used_cores_mean": round(
                    float(tgl_cpu.get("used_cores_mean", 0.0)),
                    6,
                ),
                "frost_cpu_utilization_pct_mean": round(
                    float(frost_cpu.get("utilization_pct_mean", 0.0)),
                    6,
                ),
                "tgl_cpu_utilization_pct_mean": round(
                    float(tgl_cpu.get("utilization_pct_mean", 0.0)),
                    6,
                ),
                "frost_log": str(frost_log),
                "tgl_log": str(tgl_log),
            }
        )

        for system, payload, log_path in [
            ("frost", frost_cpu, frost_log),
            ("tgl", tgl_cpu, tgl_log),
        ]:
            curve_rows.append(
                {
                    "system": system,
                    "dataset": DATASET,
                    "model": MODEL,
                    "macro_batch_size": macro_batch_size,
                    "log": str(log_path),
                    "tracked_pid_count": int(payload.get("tracked_pid_count", 0)),
                    "logical_cpu_count": int(payload.get("logical_cpu_count", 0)),
                    "used_cores_curve": payload["used_cores_curve"],
                    "utilization_pct_curve": payload["utilization_pct_curve"],
                    "wall_interval_s_curve": payload["wall_interval_s_curve"],
                }
            )

    summary_tsv = run_dir / "summary.tsv"
    with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    curves_json = run_dir / "curves.json"
    curves_json.write_text(json.dumps(curve_rows, indent=2), encoding="utf-8")

    curves_tsv = run_dir / "curves.tsv"
    with curves_tsv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "system",
            "dataset",
            "model",
            "macro_batch_size",
            "batch_index",
            "used_cores",
            "utilization_pct",
            "wall_interval_s",
            "tracked_pid_count",
            "logical_cpu_count",
            "log",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in curve_rows:
            used = row["used_cores_curve"]
            util = row["utilization_pct_curve"]
            wall = row["wall_interval_s_curve"]
            for batch_index, (used_v, util_v, wall_v) in enumerate(
                zip(used, util, wall),
                start=1,
            ):
                writer.writerow(
                    {
                        "system": row["system"],
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "macro_batch_size": row["macro_batch_size"],
                        "batch_index": batch_index,
                        "used_cores": used_v,
                        "utilization_pct": util_v,
                        "wall_interval_s": wall_v,
                        "tracked_pid_count": row["tracked_pid_count"],
                        "logical_cpu_count": row["logical_cpu_count"],
                        "log": row["log"],
                    }
                )

    plot_curves(run_dir, curve_rows)

    latest_link = OUTPUT_ROOT / "latest"
    if latest_link.is_symlink() or latest_link.is_file():
        latest_link.unlink()
    elif latest_link.is_dir():
        shutil.rmtree(latest_link)
    latest_link.symlink_to(run_dir.name)

    print(f"Summary: {summary_tsv}")
    print(f"Curves: {curves_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
