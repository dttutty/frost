#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

from frost_bench.adapters.tgl import write_tgl_config
from frost_bench.io import ensure_dir, read_text, run_command, utc_timestamp
from frost_bench.metrics import parse_all_json_after_prefix, parse_frost_rank_throughput, parse_tgl_rank_throughput, select_bottleneck_rank
from frost_bench.telemetry import (
    parse_nvidia_smi_samples,
    start_nvidia_smi_sampler,
    stop_nvidia_smi_sampler,
    summarize_utilization,
)
from frost_bench.workload import compute_tgl_max_train_steps

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments" / "nvidia_smi_gpu_util_lastfm_tgn"
TGL_ROOT = REPO_ROOT / "third_party" / "tgl"
TGL_CONFIG_SRC = TGL_ROOT / "config" / "frost_parity" / "TGN.yml"
DATASET = "LASTFM"
MODEL = "tgn"
DATASET_LOWER = DATASET.lower()
BATCH_SIZES = [600, 1200, 2000, 4000]
GPU_IDS = "0,1"
GPU_INDEXES = [0, 1]
N_GPU = 2
SEED = 0
WARMUP_BATCHES = 5
MEASURE_BATCHES = 100
OMP_THREADS = "8"
MKL_THREADS = "8"
FROST_EPOCHS = 1
TGL_EPOCHS = 1
NVIDIA_SMI_LOOP_MS = "50"


def parse_rank_windows(text: str) -> dict[int, dict]:
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_WINDOW")
    return {int(row["rank"]): row for row in rows}


def filter_samples_for_window(
    rows: list[dict[str, float | int]],
    *,
    gpu_index: int,
    window_start_unix_s: float,
    window_end_unix_s: float,
) -> list[dict[str, float | int]]:
    selected = [
        row
        for row in rows
        if int(row["gpu_index"]) == gpu_index
        and window_start_unix_s <= float(row["timestamp_unix_s"]) <= window_end_unix_s
    ]
    if not selected:
        raise RuntimeError(
            f"No nvidia-smi samples for gpu_index={gpu_index} between "
            f"{window_start_unix_s} and {window_end_unix_s}."
        )
    return selected


def plot_curves(run_dir: Path, curve_rows: list[dict]) -> None:
    if plt is None:
        return

    fig_gpu, axes_gpu = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    fig_mem, axes_mem = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes_gpu = axes_gpu.flatten()
    axes_mem = axes_mem.flatten()

    for idx, batch_size in enumerate(BATCH_SIZES):
        gpu_ax = axes_gpu[idx]
        mem_ax = axes_mem[idx]
        rows = [row for row in curve_rows if row["macro_batch_size"] == batch_size]
        for row in rows:
            t0 = row["relative_time_s_curve"]
            gpu_ax.plot(t0, row["gpu_util_curve"], label=row["system"].upper(), linewidth=1.5)
            mem_ax.plot(t0, row["mem_util_curve"], label=row["system"].upper(), linewidth=1.5)
        gpu_ax.set_title(f"LASTFM + TGN, macro-batch={batch_size}")
        mem_ax.set_title(f"LASTFM + TGN, macro-batch={batch_size}")
        gpu_ax.set_ylabel("nvidia-smi gpu util (%)")
        mem_ax.set_ylabel("nvidia-smi mem util (%)")
        gpu_ax.set_xlabel("Time since measured window start (s)")
        mem_ax.set_xlabel("Time since measured window start (s)")
        gpu_ax.grid(True, alpha=0.25)
        mem_ax.grid(True, alpha=0.25)
        gpu_ax.legend()
        mem_ax.legend()

    fig_gpu.suptitle("FROST vs TGL train_dist: nvidia-smi GPU util over measured window")
    fig_mem.suptitle("FROST vs TGL train_dist: nvidia-smi memory util over measured window")
    fig_gpu.tight_layout()
    fig_mem.tight_layout()
    fig_gpu.savefig(run_dir / "gpu_util_curves.png", dpi=180)
    fig_mem.savefig(run_dir / "memory_util_curves.png", dpi=180)
    plt.close(fig_gpu)
    plt.close(fig_mem)


def main() -> int:
    run_dir = OUTPUT_ROOT / f"nvidia_smi_lastfm_tgn_{utc_timestamp()}"
    cfg_dir = run_dir / "configs"
    log_dir = run_dir / "logs"
    ensure_dir(cfg_dir)
    ensure_dir(log_dir)

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    print(f"Writing outputs to {run_dir}")
    for idx, macro_batch_size in enumerate(BATCH_SIZES):
        per_gpu_batch_size = macro_batch_size // N_GPU
        master_port = 33900 + idx
        frost_log = log_dir / f"frost_lastfm_tgn_mb{macro_batch_size}.log"
        tgl_log = log_dir / f"tgl_train_dist_lastfm_tgn_mb{macro_batch_size}.log"
        frost_samples_path = log_dir / f"frost_lastfm_tgn_mb{macro_batch_size}_nvidia_smi.csv"
        tgl_samples_path = log_dir / f"tgl_train_dist_lastfm_tgn_mb{macro_batch_size}_nvidia_smi.csv"
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
                "CUDA_VISIBLE_DEVICES": GPU_IDS,
                "FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH": str(WARMUP_BATCHES),
                "FROST_THROUGHPUT_MEASURE_BATCHES": str(MEASURE_BATCHES),
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
                "CUDA_VISIBLE_DEVICES": GPU_IDS,
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
        frost_sampler, frost_sampler_handle = start_nvidia_smi_sampler(
            frost_samples_path,
            gpu_ids=GPU_IDS,
            loop_ms=int(NVIDIA_SMI_LOOP_MS),
            cwd=REPO_ROOT,
        )
        time.sleep(0.2)
        try:
            run_command(frost_cmd, frost_env, frost_log)
        finally:
            stop_nvidia_smi_sampler(frost_sampler, frost_sampler_handle)

        print(f"[TGL] LASTFM/TGN macro_batch_size={macro_batch_size}")
        tgl_sampler, tgl_sampler_handle = start_nvidia_smi_sampler(
            tgl_samples_path,
            gpu_ids=GPU_IDS,
            loop_ms=int(NVIDIA_SMI_LOOP_MS),
            cwd=REPO_ROOT,
        )
        time.sleep(0.2)
        try:
            run_command(tgl_cmd, tgl_env, tgl_log)
        finally:
            stop_nvidia_smi_sampler(tgl_sampler, tgl_sampler_handle)

        frost_text = read_text(frost_log)
        tgl_text = read_text(tgl_log)
        frost_throughput_by_rank = parse_frost_rank_throughput(frost_text)
        tgl_throughput_by_rank = parse_tgl_rank_throughput(tgl_text)
        frost_windows_by_rank = parse_rank_windows(frost_text)
        tgl_windows_by_rank = parse_rank_windows(tgl_text)
        frost_rank, frost_eps, frost_measured = select_bottleneck_rank(
            frost_throughput_by_rank
        )
        tgl_rank, tgl_eps, tgl_measured = select_bottleneck_rank(
            tgl_throughput_by_rank
        )
        frost_window = frost_windows_by_rank[frost_rank]
        tgl_window = tgl_windows_by_rank[tgl_rank]
        frost_samples = filter_samples_for_window(
            parse_nvidia_smi_samples(frost_samples_path),
            gpu_index=frost_rank,
            window_start_unix_s=float(frost_window["window_start_unix_s"]),
            window_end_unix_s=float(frost_window["window_end_unix_s"]),
        )
        tgl_samples = filter_samples_for_window(
            parse_nvidia_smi_samples(tgl_samples_path),
            gpu_index=tgl_rank,
            window_start_unix_s=float(tgl_window["window_start_unix_s"]),
            window_end_unix_s=float(tgl_window["window_end_unix_s"]),
        )
        frost_util = summarize_utilization(frost_samples)
        tgl_util = summarize_utilization(tgl_samples)

        summary_rows.append(
            {
                "dataset": DATASET,
                "model": MODEL,
                "macro_batch_size": macro_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "frost_bottleneck_rank": frost_rank,
                "tgl_bottleneck_rank": tgl_rank,
                "frost_events_per_sec_per_gpu": round(frost_eps, 6),
                "tgl_events_per_sec_per_gpu": round(tgl_eps, 6),
                "speedup_frost_over_tgl": round(frost_eps / tgl_eps, 6),
                "frost_measured_batches": frost_measured,
                "tgl_measured_batches": tgl_measured,
                "frost_sample_count": len(frost_samples),
                "tgl_sample_count": len(tgl_samples),
                "frost_gpu_util_mean": round(frost_util["gpu_util_mean"], 6),
                "tgl_gpu_util_mean": round(tgl_util["gpu_util_mean"], 6),
                "frost_gpu_util_p90": round(frost_util["gpu_util_p90"], 6),
                "tgl_gpu_util_p90": round(tgl_util["gpu_util_p90"], 6),
                "frost_mem_util_mean": round(frost_util["mem_util_mean"], 6),
                "tgl_mem_util_mean": round(tgl_util["mem_util_mean"], 6),
                "frost_mem_util_p90": round(frost_util["mem_util_p90"], 6),
                "tgl_mem_util_p90": round(tgl_util["mem_util_p90"], 6),
                "frost_log": str(frost_log),
                "tgl_log": str(tgl_log),
                "frost_samples": str(frost_samples_path),
                "tgl_samples": str(tgl_samples_path),
            }
        )

        for system, rank, samples, window, log_path in [
            ("frost", frost_rank, frost_samples, frost_window, frost_log),
            ("tgl", tgl_rank, tgl_samples, tgl_window, tgl_log),
        ]:
            start_ts = float(window["window_start_unix_s"])
            curve_rows.append(
                {
                    "system": system,
                    "dataset": DATASET,
                    "model": MODEL,
                    "macro_batch_size": macro_batch_size,
                    "rank": rank,
                    "sample_count": len(samples),
                    "window_start_unix_s": start_ts,
                    "window_end_unix_s": float(window["window_end_unix_s"]),
                    "relative_time_s_curve": [
                        round(float(sample["timestamp_unix_s"]) - start_ts, 4)
                        for sample in samples
                    ],
                    "gpu_util_curve": [
                        float(sample["gpu_util"]) for sample in samples
                    ],
                    "mem_util_curve": [
                        float(sample["mem_util"]) for sample in samples
                    ],
                    "log": str(log_path),
                }
            )

    summary_tsv = run_dir / "summary.tsv"
    with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys()),
            delimiter="\t",
        )
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
            "rank",
            "sample_index",
            "relative_time_s",
            "gpu_util",
            "mem_util",
            "log",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in curve_rows:
            for sample_index, (rel_t, gpu_util, mem_util) in enumerate(
                zip(
                    row["relative_time_s_curve"],
                    row["gpu_util_curve"],
                    row["mem_util_curve"],
                ),
                start=1,
            ):
                writer.writerow(
                    {
                        "system": row["system"],
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "macro_batch_size": row["macro_batch_size"],
                        "rank": row["rank"],
                        "sample_index": sample_index,
                        "relative_time_s": rel_t,
                        "gpu_util": gpu_util,
                        "mem_util": mem_util,
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
