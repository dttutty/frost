#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from frost_bench.adapters.tgl import write_tgl_config
from frost_bench.io import ensure_dir, read_text, run_command, utc_timestamp
from frost_bench.metrics import parse_all_json_after_prefix, parse_frost_rank_throughput, parse_tgl_rank_throughput, select_bottleneck_rank
from frost_bench.workload import compute_tgl_max_train_steps

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments" / "gpu_utilization_lastfm_tgn"
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


def parse_rank_gpu_curve(text: str) -> dict[int, dict]:
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_GPU_UTIL_CURVE")
    return {int(row["rank"]): row for row in rows}


def plot_curves(run_dir: Path, curve_rows: list[dict]) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    for idx, batch_size in enumerate(BATCH_SIZES):
        ax = axes[idx]
        rows = [row for row in curve_rows if row["macro_batch_size"] == batch_size]
        for row in rows:
            x = list(range(1, len(row["gpu_busy_pct_curve"]) + 1))
            label = f"{row['system'].upper()} (rank {row['rank']})"
            ax.plot(x, row["gpu_busy_pct_curve"], label=label, linewidth=1.5)
        ax.set_title(f"LASTFM + TGN, macro-batch={batch_size}")
        ax.set_ylabel("GPU busy ratio (%)")
        ax.set_xlabel("Measured train interval index")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(
        "FROST vs TGL train_dist: bottleneck-rank GPU busy ratio over 100 train intervals"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "gpu_busy_ratio_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    run_dir = OUTPUT_ROOT / f"gpu_util_lastfm_tgn_{utc_timestamp()}"
    cfg_dir = run_dir / "configs"
    log_dir = run_dir / "logs"
    ensure_dir(cfg_dir)
    ensure_dir(log_dir)

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    print(f"Writing outputs to {run_dir}")
    for idx, macro_batch_size in enumerate(BATCH_SIZES):
        per_gpu_batch_size = macro_batch_size // N_GPU
        master_port = 33600 + idx
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
                "FROST_GPU_UTIL_CURVE_BENCH": "1",
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
            "--gpu-util-curve",
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
        frost_throughput_by_rank = parse_frost_rank_throughput(frost_text)
        tgl_throughput_by_rank = parse_tgl_rank_throughput(tgl_text)
        frost_curve_by_rank = parse_rank_gpu_curve(frost_text)
        tgl_curve_by_rank = parse_rank_gpu_curve(tgl_text)

        frost_rank, frost_eps, frost_measured = select_bottleneck_rank(
            frost_throughput_by_rank
        )
        tgl_rank, tgl_eps, tgl_measured = select_bottleneck_rank(
            tgl_throughput_by_rank
        )
        frost_curve = frost_curve_by_rank[frost_rank]
        tgl_curve = tgl_curve_by_rank[tgl_rank]

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
                "frost_gpu_busy_pct_mean": round(
                    float(frost_curve.get("gpu_busy_pct_mean", 0.0)),
                    6,
                ),
                "tgl_gpu_busy_pct_mean": round(
                    float(tgl_curve.get("gpu_busy_pct_mean", 0.0)),
                    6,
                ),
                "frost_gpu_busy_pct_p90": round(
                    float(frost_curve.get("gpu_busy_pct_p90", 0.0)),
                    6,
                ),
                "tgl_gpu_busy_pct_p90": round(
                    float(tgl_curve.get("gpu_busy_pct_p90", 0.0)),
                    6,
                ),
                "frost_log": str(frost_log),
                "tgl_log": str(tgl_log),
            }
        )

        for system, rank, curve, log_path in [
            ("frost", frost_rank, frost_curve, frost_log),
            ("tgl", tgl_rank, tgl_curve, tgl_log),
        ]:
            curve_rows.append(
                {
                    "system": system,
                    "dataset": DATASET,
                    "model": MODEL,
                    "macro_batch_size": macro_batch_size,
                    "rank": rank,
                    "log": str(log_path),
                    "gpu_busy_pct_curve": curve["gpu_busy_pct_curve"],
                    "gpu_ms_curve": curve["gpu_ms_curve"],
                    "wall_ms_curve": curve["wall_ms_curve"],
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
            "batch_index",
            "gpu_busy_pct",
            "gpu_ms",
            "wall_ms",
            "log",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in curve_rows:
            for batch_index, (gpu_busy_pct, gpu_ms, wall_ms) in enumerate(
                zip(
                    row["gpu_busy_pct_curve"],
                    row["gpu_ms_curve"],
                    row["wall_ms_curve"],
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
                        "batch_index": batch_index,
                        "gpu_busy_pct": gpu_busy_pct,
                        "gpu_ms": gpu_ms,
                        "wall_ms": wall_ms,
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
