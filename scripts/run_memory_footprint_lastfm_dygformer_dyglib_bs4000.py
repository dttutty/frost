#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from frost_bench.io import ensure_dir, read_text, run_command, utc_timestamp
from frost_bench.metrics import (
    normalized_progress as progress_curve,
    parse_rank_payloads,
    sum_memory_curves_by_rank as sum_curves_by_rank,
    summarize_distribution as summarize_curve,
)
from frost_bench.workload import compute_frost_epochs, train_edge_end

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DYGLIB_ROOT = REPO_ROOT / "third_party" / "DyGLib"
OUTPUT_ROOT = REPO_ROOT / "experiments" / "memory_footprint_lastfm_dygformer_dyglib"

DATASET = "LASTFM"
MODEL = "dygformer"
DYGLIB_MODEL = "DyGFormer"
BATCH_SIZE = 4000
SEED = 0
WARMUP_BATCHES = 10
MEASURE_BATCHES = 50
OMP_THREADS = "8"
MKL_THREADS = "8"
FROST_NO_MEMORY_QUEUE_SIZE = "16"
FROST_1GPU_IDS = "0"
FROST_2GPU_IDS = "0,1"
DYGLIB_GPU_ID = "0"
DYGLIB_NUM_EPOCHS = 1000


def compute_dyglib_max_train_batches(dataset: str, batch_size: int, warmup: int, target: int) -> int:
    edge_end = train_edge_end(dataset)
    steps_per_epoch = math.ceil(edge_end / batch_size)
    measured_intervals_per_epoch = max(steps_per_epoch - warmup - 1, 0)
    if measured_intervals_per_epoch <= 0:
        raise RuntimeError(
            f"Not enough train batches per epoch for DyGLib measurement: {steps_per_epoch}"
        )
    required_epochs = max(1, math.ceil(target / measured_intervals_per_epoch))
    return required_epochs * steps_per_epoch + 2


def write_summary_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
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
        "reserved_mib_total",
        "allocated_mib_total",
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
                        "reserved_mib_total": reserved_mib,
                        "allocated_mib_total": allocated_mib,
                        "log": row["log"],
                    }
                )


def plot_memory_pdf(run_dir: Path, curve_rows: list[dict[str, object]]) -> Path | None:
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = {
        "frost_1gpu": "FROST-1GPU",
        "frost_2gpu_total": "FROST-2GPU(total)",
        "dyglib_1gpu": "DyGLib-1GPU",
    }
    colors = {
        "frost_1gpu": "#1b9e77",
        "frost_2gpu_total": "#7570b3",
        "dyglib_1gpu": "#d95f02",
    }
    for row in curve_rows:
        system = str(row["system"])
        x_curve = [float(v) * MEASURE_BATCHES / 100.0 for v in row["progress_pct_curve"]]
        ax.plot(
            x_curve,
            row["reserved_mib_curve"],
            label=labels[system],
            linewidth=1.4,
            color=colors[system],
            alpha=0.95,
        )
    ax.set_xlim(0.0, float(MEASURE_BATCHES))
    ax.set_xlabel("Measured macro-batch index")
    ax.set_ylabel("Reserved GPU memory (MiB)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, facecolor="white", framealpha=1.0, edgecolor="#cccccc")
    fig.tight_layout()
    pdf_path = run_dir / "memory_footprint.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def frost_command(*, n_gpu: int, macro_batch_size: int, epoch: int, master_port: int) -> tuple[list[str], dict[str, str]]:
    gpu_ids = FROST_1GPU_IDS if n_gpu == 1 else FROST_2GPU_IDS
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": OMP_THREADS,
            "MKL_NUM_THREADS": MKL_THREADS,
            "CUDA_VISIBLE_DEVICES": gpu_ids,
            "FROST_THROUGHPUT_WARMUP_BATCHES_TOTAL": str(WARMUP_BATCHES),
            "FROST_THROUGHPUT_MEASURE_BATCHES": str(MEASURE_BATCHES),
        }
    )
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "frost_bench.cli.train",
        MODEL,
        "--nproc-per-node",
        str(n_gpu + 1),
        "--master-port",
        str(master_port),
        f"dataset={DATASET}",
        f"epoch={epoch}",
        f"macro_batch_size={macro_batch_size}",
        f"seed={SEED}",
        "compile=false",
        "compute.tf32=false",
        "stable_mode=false",
        f"no_memory_queue_size={FROST_NO_MEMORY_QUEUE_SIZE}",
        "tqdm=true",
    ]
    return cmd, env


def dyglib_command(*, macro_batch_size: int, max_train_batches: int) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": OMP_THREADS,
            "MKL_NUM_THREADS": MKL_THREADS,
            "CUDA_VISIBLE_DEVICES": DYGLIB_GPU_ID,
        }
    )
    cmd = [
        "uv",
        "run",
        "--project",
        str(DYGLIB_ROOT),
        "python",
        str(DYGLIB_ROOT / "train_link_prediction.py"),
        "--seed-start",
        str(SEED),
        "--dataset_name",
        DATASET.lower(),
        "--model_name",
        DYGLIB_MODEL,
        "--load_best_configs",
        "--gpu",
        "0",
        "--num_runs",
        "1",
        "--num_epochs",
        str(DYGLIB_NUM_EPOCHS),
        "--batch_size",
        str(macro_batch_size),
        "--test_interval_epochs",
        "1",
        "--appendix-train-e2e",
        "--warmup-batches",
        str(WARMUP_BATCHES),
        "--measure-batches",
        str(MEASURE_BATCHES),
        "--max-train-batches",
        str(max_train_batches),
        "--features-in-host-memory",
    ]
    return cmd, env


def main() -> int:
    run_dir = OUTPUT_ROOT / f"memory_footprint_lastfm_dygformer_dyglib_bs4000_{utc_timestamp()}"
    log_dir = run_dir / "logs"
    ensure_dir(log_dir)

    frost_epochs = compute_frost_epochs(DATASET, BATCH_SIZE, WARMUP_BATCHES, MEASURE_BATCHES)
    dyglib_max_train_batches = compute_dyglib_max_train_batches(
        DATASET, BATCH_SIZE, WARMUP_BATCHES, MEASURE_BATCHES
    )

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    jobs = [
        ("frost_1gpu", 1, log_dir / "frost_1gpu_lastfm_dygformer_mb4000.log"),
        ("frost_2gpu_total", 2, log_dir / "frost_2gpu_lastfm_dygformer_mb4000.log"),
    ]

    print(f"Writing outputs to {run_dir}")
    for idx, (system_name, n_gpu, log_path) in enumerate(jobs):
        cmd, env = frost_command(
            n_gpu=n_gpu,
            macro_batch_size=BATCH_SIZE,
            epoch=frost_epochs,
            master_port=34800 + idx,
        )
        print(f"[{system_name}] LASTFM/DyGFormer mb={BATCH_SIZE} epoch={frost_epochs}")
        run_command(cmd, env, log_path)
        text = read_text(log_path)
        memory_rows = parse_rank_payloads(text, "APPENDIX_TRAIN_MEMORY_CURVE")
        reserved_total_curve, allocated_total_curve = sum_curves_by_rank(memory_rows)
        stats = summarize_curve(reserved_total_curve)
        summary_rows.append(
            {
                "system": system_name,
                "dataset": DATASET,
                "model": MODEL,
                "macro_batch_size": BATCH_SIZE,
                "warmup_batches": WARMUP_BATCHES,
                "measured_batches": len(reserved_total_curve),
                "n_gpus": n_gpu,
                "reserved_mib_mean_total": round(stats["mean"], 6),
                "reserved_mib_p50_total": round(stats["p50"], 6),
                "reserved_mib_p90_total": round(stats["p90"], 6),
                "reserved_mib_max_total": round(stats["max"], 6),
                "allocated_mib_mean_total": round(summarize_curve(allocated_total_curve)["mean"], 6),
                "log": str(log_path),
            }
        )
        curve_rows.append(
            {
                "system": system_name,
                "dataset": DATASET,
                "model": MODEL,
                "macro_batch_size": BATCH_SIZE,
                "progress_pct_curve": [round(v, 4) for v in progress_curve(reserved_total_curve)],
                "reserved_mib_curve": [round(v, 4) for v in reserved_total_curve],
                "allocated_mib_curve": [round(v, 4) for v in allocated_total_curve],
                "log": str(log_path),
            }
        )

    dyglib_log = log_dir / "dyglib_1gpu_lastfm_dygformer_mb4000.log"
    dyglib_cmd, dyglib_env = dyglib_command(
        macro_batch_size=BATCH_SIZE,
        max_train_batches=dyglib_max_train_batches,
    )
    print(f"[dyglib_1gpu] LASTFM/DyGFormer mb={BATCH_SIZE} max_train_batches={dyglib_max_train_batches}")
    run_command(dyglib_cmd, dyglib_env, dyglib_log, cwd=DYGLIB_ROOT)
    dyglib_text = read_text(dyglib_log)
    dyglib_row = parse_rank_payloads(dyglib_text, "APPENDIX_TRAIN_MEMORY_CURVE")[0]
    dyglib_reserved_curve = [float(value) for value in dyglib_row["reserved_mib_curve"]]
    dyglib_allocated_curve = [float(value) for value in dyglib_row["allocated_mib_curve"]]
    dyglib_stats = summarize_curve(dyglib_reserved_curve)
    summary_rows.append(
        {
            "system": "dyglib_1gpu",
            "dataset": DATASET,
            "model": MODEL,
            "macro_batch_size": BATCH_SIZE,
            "warmup_batches": WARMUP_BATCHES,
            "measured_batches": len(dyglib_reserved_curve),
            "n_gpus": 1,
            "reserved_mib_mean_total": round(dyglib_stats["mean"], 6),
            "reserved_mib_p50_total": round(dyglib_stats["p50"], 6),
            "reserved_mib_p90_total": round(dyglib_stats["p90"], 6),
            "reserved_mib_max_total": round(dyglib_stats["max"], 6),
            "allocated_mib_mean_total": round(summarize_curve(dyglib_allocated_curve)["mean"], 6),
            "log": str(dyglib_log),
        }
    )
    curve_rows.append(
        {
            "system": "dyglib_1gpu",
            "dataset": DATASET,
            "model": MODEL,
            "macro_batch_size": BATCH_SIZE,
            "progress_pct_curve": [round(v, 4) for v in progress_curve(dyglib_reserved_curve)],
            "reserved_mib_curve": [round(v, 4) for v in dyglib_reserved_curve],
            "allocated_mib_curve": [round(v, 4) for v in dyglib_allocated_curve],
            "log": str(dyglib_log),
        }
    )

    summary_path = run_dir / "summary.tsv"
    curves_json_path = run_dir / "curves.json"
    curves_tsv_path = run_dir / "curves.tsv"
    write_summary_tsv(summary_path, summary_rows)
    curves_json_path.write_text(json.dumps(curve_rows, indent=2), encoding="utf-8")
    write_curves_tsv(curves_tsv_path, curve_rows)
    pdf_path = plot_memory_pdf(run_dir, curve_rows)

    print(f"Summary TSV: {summary_path}")
    print(f"Curves TSV:  {curves_tsv_path}")
    print(f"Curves JSON: {curves_json_path}")
    if pdf_path is not None:
        print(f"Plot PDF:    {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
