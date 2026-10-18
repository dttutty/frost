#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

from frost_bench.io import ensure_dir, read_text, run_command, utc_timestamp
from frost_bench.metrics import parse_all_json_after_prefix, parse_frost_rank_throughput
from frost_bench.telemetry import (
    parse_nvidia_smi_samples,
    start_nvidia_smi_sampler,
    stop_nvidia_smi_sampler,
    summarize_utilization,
)
from frost_bench.workload import compute_frost_epochs, train_edge_end

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments" / "nvidia_smi_gpu_util_progress_lastfm_dygformer_dyglib"
TGL_ROOT = REPO_ROOT / "third_party" / "tgl"
DYGLIB_ROOT = REPO_ROOT / "third_party" / "DyGLib"
DATASET = "LASTFM"
MODEL = "dygformer"
DYGLIB_MODEL = "DyGFormer"
BATCH_SIZES = [600, 1200, 2000, 4000]
SEED = 0
WARMUP_BATCHES = 10
MEASURE_BATCHES = 500
OMP_THREADS = "8"
MKL_THREADS = "8"
NVIDIA_SMI_LOOP_MS = "50"
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


def parse_dyglib_summary(text: str) -> dict[str, float | int]:
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_E2E_SUMMARY")
    if len(rows) != 1:
        raise RuntimeError(f"Expected one DyGLib summary row, got {len(rows)}")
    row = rows[0]
    return {
        "events_per_sec_per_gpu": float(row["events_per_sec_per_gpu"]),
        "measured_batches": int(row["measured_batches"]),
    }


def parse_rank_intervals(text: str) -> dict[int, dict]:
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_INTERVALS")
    return {int(row["rank"]): row for row in rows}


def filter_samples_for_intervals(
    rows: list[dict[str, float | int]],
    *,
    gpu_index: int,
    interval_start_unix_s: list[float],
    interval_end_unix_s: list[float],
) -> list[dict[str, float | int]]:
    if len(interval_start_unix_s) != len(interval_end_unix_s):
        raise RuntimeError("Interval start/end length mismatch.")
    gpu_rows = [row for row in rows if int(row["gpu_index"]) == gpu_index]
    selected: list[dict[str, float | int]] = []
    interval_idx = 0
    for row in gpu_rows:
        ts = float(row["timestamp_unix_s"])
        while interval_idx < len(interval_end_unix_s) and ts > float(interval_end_unix_s[interval_idx]):
            interval_idx += 1
        if interval_idx >= len(interval_start_unix_s):
            break
        start_ts = float(interval_start_unix_s[interval_idx])
        end_ts = float(interval_end_unix_s[interval_idx])
        if start_ts <= ts <= end_ts:
            selected.append({**row, "interval_index": interval_idx})
    if not selected:
        raise RuntimeError(
            f"No nvidia-smi samples for gpu_index={gpu_index} within measured intervals."
        )
    return selected


def progress_curve(
    samples: list[dict[str, float | int]],
    *,
    interval_start_unix_s: list[float],
    interval_end_unix_s: list[float],
) -> tuple[list[float], list[float], list[float]]:
    n_intervals = max(len(interval_start_unix_s), 1)
    progress: list[float] = []
    for sample in samples:
        interval_idx = int(sample["interval_index"])
        start_ts = float(interval_start_unix_s[interval_idx])
        end_ts = float(interval_end_unix_s[interval_idx])
        duration = max(end_ts - start_ts, 1e-9)
        frac = max(0.0, min(1.0, (float(sample["timestamp_unix_s"]) - start_ts) / duration))
        progress.append(((interval_idx + frac) / n_intervals) * 100.0)
    gpu_util = [float(sample["gpu_util"]) for sample in samples]
    mem_util = [float(sample["mem_util"]) for sample in samples]
    return progress, gpu_util, mem_util


def aggregate_curves_mean(curves: list[tuple[list[float], list[float], list[float]]]) -> tuple[list[float], list[float], list[float]]:
    if not curves:
        raise RuntimeError("No curves to aggregate.")
    if len(curves) == 1:
        return curves[0]
    bin_count = 100
    progress_axis = [(i + 0.5) * 100.0 / bin_count for i in range(bin_count)]
    gpu_bins: list[list[float]] = [[] for _ in range(bin_count)]
    mem_bins: list[list[float]] = [[] for _ in range(bin_count)]
    for progress, gpu_curve, mem_curve in curves:
        for p, g, m in zip(progress, gpu_curve, mem_curve):
            idx = min(bin_count - 1, max(0, int((p / 100.0) * bin_count)))
            gpu_bins[idx].append(float(g))
            mem_bins[idx].append(float(m))
    gpu_mean = [sum(vals) / len(vals) if vals else float("nan") for vals in gpu_bins]
    mem_mean = [sum(vals) / len(vals) if vals else float("nan") for vals in mem_bins]
    return progress_axis, gpu_mean, mem_mean


def _nanmean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def resample_curve_mean(
    progress: list[float],
    gpu_curve: list[float],
    mem_curve: list[float],
    *,
    bin_count: int = 160,
) -> tuple[list[float], list[float], list[float]]:
    progress_axis = [(i + 0.5) * 100.0 / bin_count for i in range(bin_count)]
    gpu_bins: list[list[float]] = [[] for _ in range(bin_count)]
    mem_bins: list[list[float]] = [[] for _ in range(bin_count)]
    for p, g, m in zip(progress, gpu_curve, mem_curve):
        idx = min(bin_count - 1, max(0, int((float(p) / 100.0) * bin_count)))
        gpu_bins[idx].append(float(g))
        mem_bins[idx].append(float(m))
    gpu_mean = [_nanmean(vals) for vals in gpu_bins]
    mem_mean = [_nanmean(vals) for vals in mem_bins]
    return progress_axis, gpu_mean, mem_mean


def moving_average(values: list[float], *, window: int = 9) -> list[float]:
    if window <= 1:
        return values[:]
    radius = window // 2
    smoothed: list[float] = []
    n = len(values)
    for idx in range(n):
        left = max(0, idx - radius)
        right = min(n, idx + radius + 1)
        smoothed.append(_nanmean(values[left:right]))
    return smoothed


def plot_gpu_progress_pdf(run_dir: Path, curve_rows: list[dict]) -> Path | None:
    if plt is None:
        return None

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 3.5), sharex=True, sharey=True)
    labels = {
        "frost_1gpu": "FROST-1GPU",
        "frost_2gpu": "FROST-2GPU",
        "dyglib_1gpu": "DyGLib-1GPU",
    }
    colors = {
        "frost_1gpu": "#1b9e77",
        "frost_2gpu": "#7570b3",
        "dyglib_1gpu": "#d95f02",
    }

    for idx, batch_size in enumerate(BATCH_SIZES):
        ax = axes[idx]
        rows = [row for row in curve_rows if row["macro_batch_size"] == batch_size]
        for row in rows:
            system = str(row["system"])
            progress_pct_curve, gpu_util_curve, _ = resample_curve_mean(
                [float(v) for v in row["progress_pct_curve"]],
                [float(v) for v in row["gpu_util_curve"]],
                [float(v) for v in row["mem_util_curve"]],
                bin_count=160,
            )
            gpu_util_curve = moving_average(gpu_util_curve, window=11)
            x_curve = [float(v) * MEASURE_BATCHES / 100.0 for v in progress_pct_curve]
            ax.plot(
                x_curve,
                gpu_util_curve,
                label=labels[system],
                linewidth=2.0,
                color=colors[system],
                alpha=0.9,
            )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlim(0.0, float(MEASURE_BATCHES))
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.25)
        if idx == len(BATCH_SIZES) - 1:
            ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.supxlabel("Measured macro-batch index")
    fig.supylabel("nvidia-smi GPU util (%)")
    fig.tight_layout()
    pdf_path = run_dir / "gpu_util_progress.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


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
        "gpu_util",
        "mem_util",
        "log",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            for sample_index, (progress_pct, gpu_util, mem_util) in enumerate(
                zip(row["progress_pct_curve"], row["gpu_util_curve"], row["mem_util_curve"]),
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
                        "gpu_util": gpu_util,
                        "mem_util": mem_util,
                        "log": row["log"],
                    }
                )


def frost_command(*, n_gpu: int, macro_batch_size: int, epoch: int, master_port: int) -> tuple[list[str], dict[str, str], str]:
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
    return cmd, env, gpu_ids


def dyglib_command(*, macro_batch_size: int, max_train_batches: int) -> tuple[list[str], dict[str, str], str]:
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
    return cmd, env, DYGLIB_GPU_ID


def collect_frost_system(
    *,
    system_name: str,
    n_gpu: int,
    macro_batch_size: int,
    epoch: int,
    master_port: int,
    log_path: Path,
    sample_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    cmd, env, gpu_ids = frost_command(
        n_gpu=n_gpu,
        macro_batch_size=macro_batch_size,
        epoch=epoch,
        master_port=master_port,
    )
    sampler, sampler_handle = start_nvidia_smi_sampler(
        sample_path,
        gpu_ids=gpu_ids,
        loop_ms=int(NVIDIA_SMI_LOOP_MS),
        cwd=REPO_ROOT,
    )
    time.sleep(0.2)
    try:
        run_command(cmd, env, log_path)
    finally:
        stop_nvidia_smi_sampler(sampler, sampler_handle)

    text = read_text(log_path)
    throughput_by_rank = parse_frost_rank_throughput(text)
    intervals_by_rank = parse_rank_intervals(text)
    smi_rows = parse_nvidia_smi_samples(sample_path)

    rank_stats = []
    rank_curves = []
    for rank, payload in sorted(throughput_by_rank.items()):
        intervals = intervals_by_rank[rank]
        samples = filter_samples_for_intervals(
            smi_rows,
            gpu_index=rank,
            interval_start_unix_s=list(intervals["interval_start_unix_s"]),
            interval_end_unix_s=list(intervals["interval_end_unix_s"]),
        )
        util = summarize_utilization(samples)
        progress_pct_curve, gpu_util_curve, mem_util_curve = progress_curve(
            samples,
            interval_start_unix_s=list(intervals["interval_start_unix_s"]),
            interval_end_unix_s=list(intervals["interval_end_unix_s"]),
        )
        rank_stats.append(
            {
                "rank": rank,
                "events_per_sec_per_gpu": float(payload["events_per_sec_per_gpu"]),
                "measured_batches": int(payload["measured_batches"]),
                "sample_count": len(samples),
                **util,
            }
        )
        rank_curves.append((progress_pct_curve, gpu_util_curve, mem_util_curve))

    mean_eps = sum(row["events_per_sec_per_gpu"] for row in rank_stats) / len(rank_stats)
    mean_measured = sum(row["measured_batches"] for row in rank_stats) / len(rank_stats)
    mean_gpu_util = sum(row["gpu_util_mean"] for row in rank_stats) / len(rank_stats)
    mean_mem_util = sum(row["mem_util_mean"] for row in rank_stats) / len(rank_stats)
    total_samples = sum(int(row["sample_count"]) for row in rank_stats)
    agg_progress, agg_gpu_util, agg_mem_util = aggregate_curves_mean(rank_curves)

    summary = {
        "system": system_name,
        "n_gpu": n_gpu,
        "events_per_sec_per_gpu_mean": mean_eps,
        "measured_batches_mean": mean_measured,
        "gpu_util_mean_avg": mean_gpu_util,
        "mem_util_mean_avg": mean_mem_util,
        "sample_count_total": total_samples,
        "log": str(log_path),
        "samples": str(sample_path),
    }
    curve = {
        "system": system_name,
        "dataset": DATASET,
        "model": MODEL,
        "macro_batch_size": macro_batch_size,
        "progress_pct_curve": agg_progress,
        "gpu_util_curve": agg_gpu_util,
        "mem_util_curve": agg_mem_util,
        "log": str(log_path),
    }
    return summary, curve


def collect_dyglib_system(
    *,
    macro_batch_size: int,
    max_train_batches: int,
    log_path: Path,
    sample_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    cmd, env, gpu_ids = dyglib_command(
        macro_batch_size=macro_batch_size,
        max_train_batches=max_train_batches,
    )
    sampler, sampler_handle = start_nvidia_smi_sampler(
        sample_path,
        gpu_ids=gpu_ids,
        loop_ms=int(NVIDIA_SMI_LOOP_MS),
        cwd=REPO_ROOT,
    )
    time.sleep(0.2)
    try:
        run_command(cmd, env, log_path, cwd=DYGLIB_ROOT)
    finally:
        stop_nvidia_smi_sampler(sampler, sampler_handle)

    text = read_text(log_path)
    throughput = parse_dyglib_summary(text)
    intervals = parse_rank_intervals(text)[0]
    smi_rows = parse_nvidia_smi_samples(sample_path)
    samples = filter_samples_for_intervals(
        smi_rows,
        gpu_index=0,
        interval_start_unix_s=list(intervals["interval_start_unix_s"]),
        interval_end_unix_s=list(intervals["interval_end_unix_s"]),
    )
    util = summarize_utilization(samples)
    progress_pct_curve, gpu_util_curve, mem_util_curve = progress_curve(
        samples,
        interval_start_unix_s=list(intervals["interval_start_unix_s"]),
        interval_end_unix_s=list(intervals["interval_end_unix_s"]),
    )
    summary = {
        "system": "dyglib_1gpu",
        "n_gpu": 1,
        "events_per_sec_per_gpu_mean": float(throughput["events_per_sec_per_gpu"]),
        "measured_batches_mean": int(throughput["measured_batches"]),
        "gpu_util_mean_avg": util["gpu_util_mean"],
        "mem_util_mean_avg": util["mem_util_mean"],
        "sample_count_total": len(samples),
        "log": str(log_path),
        "samples": str(sample_path),
    }
    curve = {
        "system": "dyglib_1gpu",
        "dataset": DATASET,
        "model": MODEL,
        "macro_batch_size": macro_batch_size,
        "progress_pct_curve": progress_pct_curve,
        "gpu_util_curve": gpu_util_curve,
        "mem_util_curve": mem_util_curve,
        "log": str(log_path),
    }
    return summary, curve


def main() -> int:
    run_dir = OUTPUT_ROOT / f"nvidia_smi_progress_lastfm_dygformer_dyglib_{utc_timestamp()}"
    log_dir = run_dir / "logs"
    ensure_dir(log_dir)

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    print(f"Writing outputs to {run_dir}")
    for idx, macro_batch_size in enumerate(BATCH_SIZES):
        frost_epochs = compute_frost_epochs(DATASET, macro_batch_size, WARMUP_BATCHES, MEASURE_BATCHES)
        dyglib_max_train_batches = compute_dyglib_max_train_batches(
            DATASET, macro_batch_size, WARMUP_BATCHES, MEASURE_BATCHES
        )

        frost1_log = log_dir / f"frost_1gpu_lastfm_dygformer_mb{macro_batch_size}.log"
        frost1_smi = log_dir / f"frost_1gpu_lastfm_dygformer_mb{macro_batch_size}_nvidia_smi.csv"
        frost2_log = log_dir / f"frost_2gpu_lastfm_dygformer_mb{macro_batch_size}.log"
        frost2_smi = log_dir / f"frost_2gpu_lastfm_dygformer_mb{macro_batch_size}_nvidia_smi.csv"
        dyglib_log = log_dir / f"dyglib_1gpu_lastfm_dygformer_mb{macro_batch_size}.log"
        dyglib_smi = log_dir / f"dyglib_1gpu_lastfm_dygformer_mb{macro_batch_size}_nvidia_smi.csv"

        print(f"[FROST-1GPU] LASTFM/DyGFormer mb={macro_batch_size} epoch={frost_epochs}")
        frost1_summary, frost1_curve = collect_frost_system(
            system_name="frost_1gpu",
            n_gpu=1,
            macro_batch_size=macro_batch_size,
            epoch=frost_epochs,
            master_port=35100 + idx,
            log_path=frost1_log,
            sample_path=frost1_smi,
        )

        print(f"[FROST-2GPU] LASTFM/DyGFormer mb={macro_batch_size} epoch={frost_epochs}")
        frost2_summary, frost2_curve = collect_frost_system(
            system_name="frost_2gpu",
            n_gpu=2,
            macro_batch_size=macro_batch_size,
            epoch=frost_epochs,
            master_port=35200 + idx,
            log_path=frost2_log,
            sample_path=frost2_smi,
        )

        print(
            f"[DyGLib-1GPU] LASTFM/DyGFormer mb={macro_batch_size} "
            f"max_train_batches={dyglib_max_train_batches}"
        )
        dyglib_summary, dyglib_curve = collect_dyglib_system(
            macro_batch_size=macro_batch_size,
            max_train_batches=dyglib_max_train_batches,
            log_path=dyglib_log,
            sample_path=dyglib_smi,
        )

        summary_rows.append(
            {
                "dataset": DATASET,
                "model": MODEL,
                "macro_batch_size": macro_batch_size,
                "frost_1gpu_events_per_sec_per_gpu_mean": round(
                    float(frost1_summary["events_per_sec_per_gpu_mean"]), 6
                ),
                "frost_2gpu_events_per_sec_per_gpu_mean": round(
                    float(frost2_summary["events_per_sec_per_gpu_mean"]), 6
                ),
                "dyglib_1gpu_events_per_sec_per_gpu_mean": round(
                    float(dyglib_summary["events_per_sec_per_gpu_mean"]), 6
                ),
                "frost_1gpu_gpu_util_mean": round(float(frost1_summary["gpu_util_mean_avg"]), 6),
                "frost_2gpu_gpu_util_mean": round(float(frost2_summary["gpu_util_mean_avg"]), 6),
                "dyglib_1gpu_gpu_util_mean": round(float(dyglib_summary["gpu_util_mean_avg"]), 6),
                "frost_1gpu_mem_util_mean": round(float(frost1_summary["mem_util_mean_avg"]), 6),
                "frost_2gpu_mem_util_mean": round(float(frost2_summary["mem_util_mean_avg"]), 6),
                "dyglib_1gpu_mem_util_mean": round(float(dyglib_summary["mem_util_mean_avg"]), 6),
                "frost_1gpu_measured_batches": int(float(frost1_summary["measured_batches_mean"])),
                "frost_2gpu_measured_batches": int(float(frost2_summary["measured_batches_mean"])),
                "dyglib_1gpu_measured_batches": int(float(dyglib_summary["measured_batches_mean"])),
                "frost_1gpu_sample_count": int(frost1_summary["sample_count_total"]),
                "frost_2gpu_sample_count": int(frost2_summary["sample_count_total"]),
                "dyglib_1gpu_sample_count": int(dyglib_summary["sample_count_total"]),
                "frost_1gpu_log": str(frost1_log),
                "frost_2gpu_log": str(frost2_log),
                "dyglib_1gpu_log": str(dyglib_log),
            }
        )
        curve_rows.extend([frost1_curve, frost2_curve, dyglib_curve])

    summary_path = run_dir / "summary.tsv"
    curves_tsv_path = run_dir / "curves.tsv"
    curves_json_path = run_dir / "curves.json"
    write_summary_tsv(summary_path, summary_rows)
    write_curves_tsv(curves_tsv_path, curve_rows)
    curves_json_path.write_text(json.dumps(curve_rows, indent=2), encoding="utf-8")
    pdf_path = plot_gpu_progress_pdf(run_dir, curve_rows)

    print()
    print(f"Summary TSV: {summary_path}")
    print(f"Curves TSV:  {curves_tsv_path}")
    print(f"Curves JSON: {curves_json_path}")
    if pdf_path is not None:
        print(f"Plot PDF:    {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
