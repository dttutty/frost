#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from frost_bench.adapters import DyGLibAdapter, FrostAdapter
from frost_bench.io import (
    ensure_dir,
    read_text,
    run_prepared,
    utc_timestamp,
    write_rows_tsv,
)
from frost_bench.metrics import (
    mean_events_per_sec,
    parse_dyglib_summary,
    parse_frost_rank_throughput,
    select_bottleneck_rank,
)
from frost_bench.paths import artifact_root
from frost_bench.spec import BatchSizeScope, ExperimentSpec
from frost_bench.workload import (
    compute_dyglib_max_train_batches,
    compute_frost_epochs,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None  # type: ignore[assignment]


OUTPUT_ROOT = artifact_root() / "throughput" / "frost_vs_dyglib_scaling"
DATASET = "LASTFM"
FROST_MODELS = ("tgn", "dygformer")
FROST_ADAPTER = FrostAdapter()
DYGLIB_ADAPTER = DyGLibAdapter()


@dataclass(frozen=True)
class FrostJob:
    model: str
    n_gpu: int
    macro_batch_size: int


def render_progress(
    progress_path: Path,
    *,
    run_dir: Path,
    total_jobs: int,
    completed_rows: list[dict[str, object]],
    pending_jobs: list[str],
) -> None:
    lines = [
        "# LASTFM Throughput Progress",
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
                f"{row['system']}",
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

    frost_rows = [
        row
        for row in summary_rows
        if row["system"] == "frost" and row["status"] == "ok"
    ]
    if not frost_rows:
        return []

    plot_paths: list[Path] = []
    model_order = [
        model
        for model in FROST_MODELS
        if any(row["model"] == model for row in frost_rows)
    ]
    labels_by_gpu = {
        1: "1 GPU\nmb2000",
        2: "2 GPU\nmb4000",
        3: "3 GPU\nmb6000",
        4: "4 GPU\nmb8000",
    }

    fig, axes = plt.subplots(
        1, len(model_order), figsize=(6.2 * len(model_order), 4.2), sharey=True
    )
    if len(model_order) == 1:
        axes = [axes]
    for ax, model in zip(axes, model_order):
        rows = sorted(
            [row for row in frost_rows if row["model"] == model],
            key=lambda item: int(item["n_gpu"]),
        )
        x = [int(row["n_gpu"]) for row in rows]
        y = [float(row["events_per_sec_per_gpu"]) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#1b9e77", label="FROST")
        if model == "dygformer":
            baselines = [
                row
                for row in summary_rows
                if row["system"] == "dyglib"
                and row["model"] == "dygformer"
                and row["status"] == "ok"
            ]
            if baselines:
                baseline = float(baselines[0]["events_per_sec_per_gpu"])
                ax.axhline(
                    baseline,
                    color="#d95f02",
                    linestyle="--",
                    linewidth=2.0,
                    label="DyGLib 1GPU mb2000",
                )
        ax.set_title(f"{model.upper() if model == 'tgn' else 'DyGFormer'}")
        ax.set_xlabel("GPU count / macro-batch")
        ax.set_ylabel("Events / sec / GPU")
        ax.set_xticks(x, [labels_by_gpu.get(v, str(v)) for v in x])
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("LASTFM throughput scaling")
    fig.tight_layout()
    per_gpu_path = run_dir / "throughput_per_gpu.png"
    fig.savefig(per_gpu_path, dpi=180)
    plt.close(fig)
    plot_paths.append(per_gpu_path)

    fig, axes = plt.subplots(
        1, len(model_order), figsize=(6.2 * len(model_order), 4.2), sharey=True
    )
    if len(model_order) == 1:
        axes = [axes]
    for ax, model in zip(axes, model_order):
        rows = sorted(
            [row for row in frost_rows if row["model"] == model],
            key=lambda item: int(item["n_gpu"]),
        )
        x = [int(row["n_gpu"]) for row in rows]
        y = [float(row["events_per_sec_total"]) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#377eb8")
        ax.set_title(f"{model.upper() if model == 'tgn' else 'DyGFormer'}")
        ax.set_xlabel("GPU count / macro-batch")
        ax.set_ylabel("Aggregate events / sec")
        ax.set_xticks(x, [labels_by_gpu.get(v, str(v)) for v in x])
        ax.grid(True, alpha=0.25)
    fig.suptitle("LASTFM aggregate throughput scaling")
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
        "# LASTFM Throughput Report",
        "",
        f"- Dataset: {DATASET}",
        f"- FROST models: {' '.join(FROST_MODELS)}",
        "- DyGLib baseline: DyGFormer / 1 GPU / mb=2000",
        f"- Warmup/measure: {args.warmup_batches}/{args.measure_batches}",
        f"- OMP_NUM_THREADS={args.omp_threads}, MKL_NUM_THREADS={args.mkl_threads}",
        f"- Run dir: {run_dir}",
        "",
        "## Summary",
        "",
        "| system | model | n_gpu | macro_batch_size | events/sec/gpu | total events/sec | measured_batches | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(
        summary_rows,
        key=lambda item: (
            str(item["model"]),
            str(item["system"]),
            int(item["n_gpu"]),
            int(item["macro_batch_size"]),
        ),
    ):
        if row["status"] == "ok":
            lines.append(
                f"| {row['system']} | {row['model']} | {row['n_gpu']} | {row['macro_batch_size']} | "
                f"{float(row['events_per_sec_per_gpu']):.6f} | {float(row['events_per_sec_total']):.6f} | "
                f"{int(row['measured_batches'])} | ok |"
            )
        else:
            lines.append(
                f"| {row['system']} | {row['model']} | {row['n_gpu']} | {row['macro_batch_size']} | "
                f"-- | -- | -- | failed |"
            )

    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for path in plot_paths:
            lines.append(f"- {path.name}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serial LASTFM throughput benchmark for FROST TGN/DyGFormer scaling "
            "plus DyGLib DyGFormer 1GPU baseline."
        )
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--mkl-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=36400)
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
        raise ValueError(
            "--gpu-ids must include at least 4 GPUs for the requested sweep"
        )

    frost_jobs = [
        FrostJob(model=model, n_gpu=n_gpu, macro_batch_size=macro_batch_size)
        for model in FROST_MODELS
        for n_gpu, macro_batch_size in ((1, 2000), (2, 4000), (3, 6000), (4, 8000))
    ]
    total_jobs = len(frost_jobs) + 1

    run_name = args.run_name or (
        f"lastfm_tgn_dygformer_scaling_w{args.warmup_batches}_m{args.measure_batches}_"
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
        f"frost/{job.model}/{job.n_gpu}gpu/mb{job.macro_batch_size}"
        for job in frost_jobs
    ] + ["dyglib/dygformer/1gpu/mb2000"]
    render_progress(
        progress_path,
        run_dir=run_dir,
        total_jobs=total_jobs,
        completed_rows=completed_rows,
        pending_jobs=pending_jobs,
    )

    for idx, job in enumerate(frost_jobs):
        log_path = (
            log_dir / f"frost_{job.model}_{job.n_gpu}gpu_mb{job.macro_batch_size}.log"
        )
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
            prepared = FROST_ADAPTER.prepare(
                ExperimentSpec(
                    model=job.model,
                    dataset=DATASET,
                    seed=args.seed,
                    n_gpus=job.n_gpu,
                    batch_size=job.macro_batch_size,
                    batch_size_scope=BatchSizeScope.GLOBAL,
                    epochs=frost_epochs,
                    extra_args=("tqdm=true",),
                ),
                gpu_ids=",".join(gpu_list[: job.n_gpu]),
                warmup_batches=args.warmup_batches,
                measure_batches=args.measure_batches,
                master_port=args.master_port_base + idx,
                omp_threads=args.omp_threads,
                mkl_threads=args.mkl_threads,
            )
            run_prepared(prepared, log_path)
            throughput_by_rank = parse_frost_rank_throughput(read_text(log_path))
            bottleneck_rank, bottleneck_eps, measured_batches = select_bottleneck_rank(
                throughput_by_rank
            )
            row.update(
                {
                    "status": "ok",
                    "epochs": frost_epochs,
                    "events_per_sec_per_gpu": bottleneck_eps,
                    "events_per_sec_mean_per_gpu": mean_events_per_sec(
                        throughput_by_rank
                    ),
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

    dyglib_log_path = log_dir / "dyglib_dygformer_1gpu_mb2000.log"
    dyglib_row: dict[str, object] = {
        "system": "dyglib",
        "model": "dygformer",
        "n_gpu": 1,
        "macro_batch_size": 2000,
        "status": "failed",
        "log_path": str(dyglib_log_path),
    }
    try:
        max_train_batches = compute_dyglib_max_train_batches(
            DATASET, 2000, args.warmup_batches, args.measure_batches
        )
        prepared = DYGLIB_ADAPTER.prepare(
            ExperimentSpec(
                model="dygformer",
                dataset=DATASET,
                seed=args.seed,
                n_gpus=1,
                batch_size=2000,
                batch_size_scope=BatchSizeScope.GLOBAL,
                epochs=1000,
            ),
            gpu_ids=gpu_list[0],
            max_train_batches=max_train_batches,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
        )
        run_prepared(prepared, dyglib_log_path)
        throughput = parse_dyglib_summary(read_text(dyglib_log_path))
        dyglib_row.update(
            {
                "status": "ok",
                "events_per_sec_per_gpu": float(throughput["events_per_sec_per_gpu"]),
                "events_per_sec_mean_per_gpu": float(
                    throughput["events_per_sec_per_gpu"]
                ),
                "events_per_sec_total": float(throughput["events_per_sec_per_gpu"]),
                "measured_batches": int(throughput["measured_batches"]),
                "max_train_batches": max_train_batches,
            }
        )
    except Exception as exc:
        dyglib_row["error"] = str(exc)
    completed_rows.append(dyglib_row)
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
