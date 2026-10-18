#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from frost_bench.adapters.tgl import write_tgl_config
from frost_bench.io import ensure_dir, read_text, run_command, utc_timestamp
from frost_bench.metrics import (
    normalized_progress as progress_curve,
    parse_all_json_after_prefix,
    parse_frost_rank_throughput,
    parse_tgl_rank_throughput,
    summarize_distribution as summarize_curve,
)
from frost_bench.workload import compute_frost_epochs, compute_tgl_max_train_steps

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments" / "memory_footprint_progress_lastfm_tgn"
TGL_ROOT = REPO_ROOT / "third_party" / "tgl"
TGL_CONFIG_SRC = TGL_ROOT / "config" / "frost_parity" / "TGN.yml"
DATASET = "LASTFM"
MODEL = "tgn"
DATASET_LOWER = DATASET.lower()
BATCH_SIZES = [600, 1200, 2000, 4000]
GPU_IDS = "0,1"
N_GPU = 2
SEED = 0
WARMUP_BATCHES = 10
MEASURE_BATCHES = 500
OMP_THREADS = "8"
MKL_THREADS = "8"
TGL_EPOCHS = 1000


def parse_rank_memory_curves(text: str) -> dict[int, dict]:
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_MEMORY_CURVE")
    return {
        int(row["rank"]): row
        for row in rows
        if int(row["rank"]) < N_GPU
    }


def aggregate_max_curve(memory_rows_by_rank: dict[int, dict]) -> tuple[list[float], list[float]]:
    if not memory_rows_by_rank:
        raise RuntimeError("No rank-local memory curves found.")
    reserved_curves = [
        [float(v) for v in row["reserved_mib_curve"]]
        for row in memory_rows_by_rank.values()
    ]
    allocated_curves = [
        [float(v) for v in row["allocated_mib_curve"]]
        for row in memory_rows_by_rank.values()
    ]
    min_len = min(len(curve) for curve in reserved_curves)
    if min_len <= 0:
        raise RuntimeError("Empty reserved memory curve.")
    reserved_max = [
        max(curve[idx] for curve in reserved_curves)
        for idx in range(min_len)
    ]
    allocated_max = [
        max(curve[idx] for curve in allocated_curves)
        for idx in range(min_len)
    ]
    return reserved_max, allocated_max


def plot_memory_progress_pdf(run_dir: Path, curve_rows: list[dict]) -> Path | None:
    if plt is None:
        return None

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 3.4), sharex=True, sharey=True)
    labels = {"frost": "FROST", "tgl": "TGL-dist"}
    colors = {"frost": "#1b9e77", "tgl": "#d95f02"}

    for idx, batch_size in enumerate(BATCH_SIZES):
        ax = axes[idx]
        rows = [row for row in curve_rows if row["macro_batch_size"] == batch_size]
        for row in rows:
            system = str(row["system"])
            x_curve = [float(v) * MEASURE_BATCHES / 100.0 for v in row["progress_pct_curve"]]
            ax.plot(
                x_curve,
                row["reserved_mib_curve"],
                label=labels[system],
                linewidth=1.2,
                color=colors[system],
                alpha=0.95,
            )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlim(0.0, float(MEASURE_BATCHES))
        ax.grid(True, alpha=0.25)
        if idx == len(BATCH_SIZES) - 1:
            ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.supxlabel("Measured macro-batch index")
    fig.supylabel("Reserved GPU memory (MiB)")
    fig.tight_layout()
    pdf_path = run_dir / "memory_footprint_progress.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def write_summary_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_curves_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "system",
        "dataset",
        "model",
        "macro_batch_size",
        "sample_index",
        "progress_pct",
        "reserved_mib",
        "allocated_mib",
        "log",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            for sample_index, (progress_pct, reserved_mib, allocated_mib) in enumerate(
                zip(
                    row["progress_pct_curve"],
                    row["reserved_mib_curve"],
                    row["allocated_mib_curve"],
                ),
                start=1,
            ):
                writer.writerow(
                    {
                        "system": row["system"],
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "macro_batch_size": row["macro_batch_size"],
                        "sample_index": sample_index,
                        "progress_pct": progress_pct,
                        "reserved_mib": reserved_mib,
                        "allocated_mib": allocated_mib,
                        "log": row["log"],
                    }
                )


def main() -> int:
    run_dir = OUTPUT_ROOT / f"memory_footprint_progress_lastfm_tgn_{utc_timestamp()}"
    cfg_dir = run_dir / "configs"
    log_dir = run_dir / "logs"
    ensure_dir(cfg_dir)
    ensure_dir(log_dir)

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    print(f"Writing outputs to {run_dir}")
    for idx, macro_batch_size in enumerate(BATCH_SIZES):
        per_gpu_batch_size = macro_batch_size // N_GPU
        master_port = 34500 + idx
        frost_log = log_dir / f"frost_lastfm_tgn_mb{macro_batch_size}.log"
        tgl_log = log_dir / f"tgl_train_dist_lastfm_tgn_mb{macro_batch_size}.log"
        tgl_cfg = cfg_dir / f"tgn_lastfm_mb{macro_batch_size}.yml"
        frost_epochs = compute_frost_epochs(
            DATASET,
            macro_batch_size,
            WARMUP_BATCHES,
            MEASURE_BATCHES,
        )
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
                "FROST_THROUGHPUT_WARMUP_BATCHES_TOTAL": str(WARMUP_BATCHES),
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
            f"epoch={frost_epochs}",
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

        print(f"[FROST] LASTFM/TGN macro_batch_size={macro_batch_size} epoch={frost_epochs}")
        run_command(frost_cmd, frost_env, frost_log)
        print(f"[TGL-dist] LASTFM/TGN macro_batch_size={macro_batch_size} max_train_steps={tgl_max_train_steps}")
        run_command(tgl_cmd, tgl_env, tgl_log)

        frost_text = read_text(frost_log)
        tgl_text = read_text(tgl_log)
        frost_throughput_by_rank = parse_frost_rank_throughput(frost_text)
        tgl_throughput_by_rank = parse_tgl_rank_throughput(tgl_text)
        frost_memory_rows = parse_rank_memory_curves(frost_text)
        tgl_memory_rows = parse_rank_memory_curves(tgl_text)
        frost_reserved_max, frost_allocated_max = aggregate_max_curve(frost_memory_rows)
        tgl_reserved_max, tgl_allocated_max = aggregate_max_curve(tgl_memory_rows)
        frost_stats = summarize_curve(frost_reserved_max)
        tgl_stats = summarize_curve(tgl_reserved_max)
        frost_eps = min(
            float(payload["events_per_sec_per_gpu"])
            for payload in frost_throughput_by_rank.values()
        )
        tgl_eps = min(
            float(payload["events_per_sec_per_gpu"])
            for payload in tgl_throughput_by_rank.values()
        )

        summary_rows.append(
            {
                "dataset": DATASET,
                "model": MODEL,
                "macro_batch_size": macro_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "frost_epochs": frost_epochs,
                "tgl_max_train_steps": tgl_max_train_steps,
                "frost_events_per_sec_per_gpu": round(frost_eps, 6),
                "tgl_events_per_sec_per_gpu": round(tgl_eps, 6),
                "speedup_frost_over_tgl": round(frost_eps / tgl_eps, 6),
                "frost_reserved_mib_mean_per_gpu_max": round(frost_stats["mean"], 6),
                "tgl_reserved_mib_mean_per_gpu_max": round(tgl_stats["mean"], 6),
                "frost_reserved_mib_p90_per_gpu_max": round(frost_stats["p90"], 6),
                "tgl_reserved_mib_p90_per_gpu_max": round(tgl_stats["p90"], 6),
                "frost_reserved_mib_max_per_gpu_max": round(frost_stats["max"], 6),
                "tgl_reserved_mib_max_per_gpu_max": round(tgl_stats["max"], 6),
                "frost_memory_measured_batches": len(frost_reserved_max),
                "tgl_memory_measured_batches": len(tgl_reserved_max),
                "frost_log": str(frost_log),
                "tgl_log": str(tgl_log),
            }
        )

        curve_rows.append(
            {
                "system": "frost",
                "dataset": DATASET,
                "model": MODEL,
                "macro_batch_size": macro_batch_size,
                "progress_pct_curve": [round(v, 4) for v in progress_curve(frost_reserved_max)],
                "reserved_mib_curve": [round(v, 4) for v in frost_reserved_max],
                "allocated_mib_curve": [round(v, 4) for v in frost_allocated_max],
                "log": str(frost_log),
            }
        )
        curve_rows.append(
            {
                "system": "tgl",
                "dataset": DATASET,
                "model": MODEL,
                "macro_batch_size": macro_batch_size,
                "progress_pct_curve": [round(v, 4) for v in progress_curve(tgl_reserved_max)],
                "reserved_mib_curve": [round(v, 4) for v in tgl_reserved_max],
                "allocated_mib_curve": [round(v, 4) for v in tgl_allocated_max],
                "log": str(tgl_log),
            }
        )

    write_summary_tsv(run_dir / "summary.tsv", summary_rows)
    (run_dir / "curves.json").write_text(json.dumps(curve_rows, indent=2), encoding="utf-8")
    write_curves_tsv(run_dir / "curves.tsv", curve_rows)

    pdf_path = plot_memory_progress_pdf(run_dir, curve_rows)

    latest_link = OUTPUT_ROOT / "latest"
    if latest_link.is_symlink() or latest_link.is_file():
        latest_link.unlink()
    elif latest_link.is_dir():
        shutil.rmtree(latest_link)
    latest_link.symlink_to(run_dir.name)

    print(f"Summary: {run_dir / 'summary.tsv'}")
    if pdf_path is not None:
        print(f"PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
