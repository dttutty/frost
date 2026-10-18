#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from frost_bench.io import ensure_dir, read_text, run_command, utc_timestamp, write_rows_tsv
from frost_bench.metrics import mean_events_per_sec, parse_frost_rank_throughput, select_bottleneck_rank
from frost_bench.workload import compute_frost_epochs

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments" / "com_throughput_lastfm_tgn_dygformer_frost_124"
DATASET = "LASTFM"
FROST_MODELS = ("tgn", "dygformer")
GPU_BATCH_PAIRS = ((1, 2000), (2, 4000), (4, 8000))


@dataclass(frozen=True)
class FrostJob:
    model: str
    n_gpu: int
    macro_batch_size: int


def frost_command(
    *,
    model: str,
    gpu_ids: str,
    n_gpu: int,
    macro_batch_size: int,
    epochs: int,
    seed: int,
    warmup_batches: int,
    measure_batches: int,
    master_port: int,
    omp_threads: int,
    mkl_threads: int,
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(mkl_threads),
            "CUDA_VISIBLE_DEVICES": gpu_ids,
            "FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH": "0",
            "FROST_THROUGHPUT_WARMUP_BATCHES_TOTAL": str(warmup_batches),
            "FROST_THROUGHPUT_MEASURE_BATCHES": str(measure_batches),
        }
    )
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "frost_bench.cli.train",
        model,
        "--nproc-per-node",
        str(n_gpu + 1),
        "--master-port",
        str(master_port),
        f"dataset={DATASET}",
        f"epoch={epochs}",
        f"macro_batch_size={macro_batch_size}",
        "stable_mode=true",
        f"seed={seed}",
        "compile=false",
        "compute.tf32=false",
        "tqdm=true",
    ]
    return cmd, env


def render_progress(
    progress_path: Path,
    *,
    run_dir: Path,
    total_jobs: int,
    completed_rows: list[dict[str, object]],
    pending_jobs: list[str],
) -> None:
    lines = [
        "# LASTFM FROST Throughput Progress",
        "",
        f"- Run dir: {run_dir}",
        f"- Completed: {len(completed_rows)}/{total_jobs}",
        "",
        "## Completed Jobs",
        "",
    ]
    if completed_rows:
        for row in completed_rows:
            bits = [
                f"{row['model']}",
                f"{row['n_gpu']} GPU",
                f"mb={row['macro_batch_size']}",
                f"status={row['status']}",
            ]
            if row["status"] == "ok":
                bits.append(f"ev/s/gpu={float(row['events_per_sec_per_gpu']):.6f}")
                bits.append(f"measured={int(row['measured_batches'])}")
            lines.append("- " + ", ".join(bits))
    else:
        lines.append("- none yet")
    lines.extend(["", "## Pending Jobs", ""])
    if pending_jobs:
        for item in pending_jobs:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(summary_rows: list[dict[str, object]], run_dir: Path) -> list[Path]:
    if plt is None:
        return []
    ok_rows = [row for row in summary_rows if row["status"] == "ok"]
    if not ok_rows:
        return []

    plot_paths: list[Path] = []
    labels_by_gpu = {1: "1 GPU\nmb2000", 2: "2 GPU\nmb4000", 4: "4 GPU\nmb8000"}
    model_order = [model for model in FROST_MODELS if any(row["model"] == model for row in ok_rows)]

    fig, axes = plt.subplots(1, len(model_order), figsize=(6.2 * len(model_order), 4.2), sharey=True)
    if len(model_order) == 1:
        axes = [axes]
    for ax, model in zip(axes, model_order):
        rows = sorted(
            [row for row in ok_rows if row["model"] == model],
            key=lambda item: int(item["n_gpu"]),
        )
        x = [int(row["n_gpu"]) for row in rows]
        y = [float(row["events_per_sec_per_gpu"]) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#1b9e77")
        ax.set_title("TGN" if model == "tgn" else "DyGFormer")
        ax.set_xlabel("GPU count / macro-batch")
        ax.set_ylabel("Events / sec / GPU")
        ax.set_xticks(x, [labels_by_gpu.get(v, str(v)) for v in x])
        ax.grid(True, alpha=0.25)
    fig.suptitle("LASTFM FROST throughput scaling")
    fig.tight_layout()
    per_gpu_path = run_dir / "throughput_per_gpu.png"
    fig.savefig(per_gpu_path, dpi=180)
    plt.close(fig)
    plot_paths.append(per_gpu_path)

    fig, axes = plt.subplots(1, len(model_order), figsize=(6.2 * len(model_order), 4.2), sharey=True)
    if len(model_order) == 1:
        axes = [axes]
    for ax, model in zip(axes, model_order):
        rows = sorted(
            [row for row in ok_rows if row["model"] == model],
            key=lambda item: int(item["n_gpu"]),
        )
        x = [int(row["n_gpu"]) for row in rows]
        y = [float(row["events_per_sec_total"]) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#377eb8")
        ax.set_title("TGN" if model == "tgn" else "DyGFormer")
        ax.set_xlabel("GPU count / macro-batch")
        ax.set_ylabel("Aggregate events / sec")
        ax.set_xticks(x, [labels_by_gpu.get(v, str(v)) for v in x])
        ax.grid(True, alpha=0.25)
    fig.suptitle("LASTFM FROST aggregate throughput scaling")
    fig.tight_layout()
    total_path = run_dir / "throughput_total.png"
    fig.savefig(total_path, dpi=180)
    plt.close(fig)
    plot_paths.append(total_path)

    return plot_paths


def write_report(
    report_path: Path,
    *,
    run_dir: Path,
    summary_rows: list[dict[str, object]],
    plot_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# LASTFM FROST Throughput Report",
        "",
        f"- Dataset: {DATASET}",
        f"- Models: {' '.join(FROST_MODELS)}",
        "- GPU/macro-batch pairs: 1/2000, 2/4000, 4/8000",
        f"- Warmup/measure: {args.warmup_batches}/{args.measure_batches}",
        f"- OMP_NUM_THREADS={args.omp_threads}, MKL_NUM_THREADS={args.mkl_threads}",
        f"- Run dir: {run_dir}",
        "",
        "## Summary",
        "",
        "| model | n_gpu | macro_batch_size | events/sec/gpu | total events/sec | measured_batches | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(
        summary_rows,
        key=lambda item: (str(item["model"]), int(item["n_gpu"]), int(item["macro_batch_size"])),
    ):
        if row["status"] == "ok":
            lines.append(
                f"| {row['model']} | {row['n_gpu']} | {row['macro_batch_size']} | "
                f"{float(row['events_per_sec_per_gpu']):.6f} | {float(row['events_per_sec_total']):.6f} | "
                f"{int(row['measured_batches'])} | ok |"
            )
        else:
            lines.append(
                f"| {row['model']} | {row['n_gpu']} | {row['macro_batch_size']} | -- | -- | -- | failed |"
            )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for path in plot_paths:
            lines.append(f"- {path.name}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serial LASTFM FROST throughput benchmark for TGN and DyGFormer on 1/2/4 GPUs."
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--mkl-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=36600)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be >= 0")
    if args.measure_batches < 1:
        raise ValueError("--measure-batches must be >= 1")
    if args.omp_threads < 1 or args.mkl_threads < 1:
        raise ValueError("--omp-threads and --mkl-threads must be >= 1")

    gpu_list = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    if len(gpu_list) < 4:
        raise ValueError("--gpu-ids must include at least 4 GPUs for this sweep")

    jobs = [
        FrostJob(model=model, n_gpu=n_gpu, macro_batch_size=macro_batch_size)
        for model in FROST_MODELS
        for n_gpu, macro_batch_size in GPU_BATCH_PAIRS
    ]
    total_jobs = len(jobs)

    run_name = args.run_name or (
        f"lastfm_tgn_dygformer_frost_124_w{args.warmup_batches}_m{args.measure_batches}_"
        f"omp{args.omp_threads}_mkl{args.mkl_threads}_live_{utc_timestamp()}"
    )
    run_dir = args.output_root / run_name
    log_dir = run_dir / "logs"
    ensure_dir(log_dir)

    results_path = run_dir / "results.tsv"
    summary_path = run_dir / "summary.tsv"
    progress_path = run_dir / "progress.md"
    report_path = run_dir / "report.md"

    completed_rows: list[dict[str, object]] = []
    pending_jobs = [
        f"{job.model}/{job.n_gpu}gpu/mb{job.macro_batch_size}" for job in jobs
    ]
    render_progress(
        progress_path,
        run_dir=run_dir,
        total_jobs=total_jobs,
        completed_rows=completed_rows,
        pending_jobs=pending_jobs,
    )

    for idx, job in enumerate(jobs):
        log_path = log_dir / f"frost_{job.model}_{job.n_gpu}gpu_mb{job.macro_batch_size}.log"
        row: dict[str, object] = {
            "system": "frost",
            "model": job.model,
            "n_gpu": job.n_gpu,
            "macro_batch_size": job.macro_batch_size,
            "status": "failed",
            "log_path": str(log_path),
        }
        try:
            frost_epochs = compute_frost_epochs(
                DATASET, job.macro_batch_size, args.warmup_batches, args.measure_batches
            )
            cmd, env = frost_command(
                model=job.model,
                gpu_ids=",".join(gpu_list[: job.n_gpu]),
                n_gpu=job.n_gpu,
                macro_batch_size=job.macro_batch_size,
                epochs=frost_epochs,
                seed=args.seed,
                warmup_batches=args.warmup_batches,
                measure_batches=args.measure_batches,
                master_port=args.master_port_base + idx,
                omp_threads=args.omp_threads,
                mkl_threads=args.mkl_threads,
            )
            run_command(cmd, env, log_path, REPO_ROOT)
            throughput_by_rank = parse_frost_rank_throughput(read_text(log_path))
            bottleneck_rank, bottleneck_eps, measured_batches = select_bottleneck_rank(throughput_by_rank)
            row.update(
                {
                    "status": "ok",
                    "epochs": frost_epochs,
                    "events_per_sec_per_gpu": bottleneck_eps,
                    "events_per_sec_mean_per_gpu": mean_events_per_sec(throughput_by_rank),
                    "events_per_sec_total": bottleneck_eps * job.n_gpu,
                    "measured_batches": measured_batches,
                    "bottleneck_rank": bottleneck_rank,
                }
            )
        except Exception as exc:
            row["error"] = str(exc)
        completed_rows.append(row)
        pending_jobs.pop(0)
        write_rows_tsv(results_path, completed_rows)
        render_progress(
            progress_path,
            run_dir=run_dir,
            total_jobs=total_jobs,
            completed_rows=completed_rows,
            pending_jobs=pending_jobs,
        )

    summary_rows = [
        {
            "system": row["system"],
            "model": row["model"],
            "n_gpu": row["n_gpu"],
            "macro_batch_size": row["macro_batch_size"],
            "events_per_sec_per_gpu": row.get("events_per_sec_per_gpu"),
            "events_per_sec_total": row.get("events_per_sec_total"),
            "measured_batches": row.get("measured_batches"),
            "status": row["status"],
            "log_path": row["log_path"],
        }
        for row in completed_rows
    ]
    write_rows_tsv(summary_path, summary_rows)
    plot_paths = plot_results(summary_rows, run_dir)
    write_report(
        report_path,
        run_dir=run_dir,
        summary_rows=summary_rows,
        plot_paths=plot_paths,
        args=args,
    )

    print(f"Run dir: {run_dir}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
