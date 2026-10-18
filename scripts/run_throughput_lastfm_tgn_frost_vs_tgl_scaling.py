#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from frost_bench.adapters import FrostAdapter, TGLAdapter
from frost_bench.io import (
    ensure_dir,
    read_text,
    run_prepared,
    utc_timestamp,
    write_rows_tsv,
)
from frost_bench.metrics import (
    mean_events_per_sec,
    parse_frost_rank_throughput,
    parse_tgl_rank_throughput,
    select_bottleneck_rank,
)
from frost_bench.paths import TGL_ROOT, artifact_root
from frost_bench.spec import BatchSizeScope, ExperimentSpec
from frost_bench.workload import (
    compute_frost_epochs,
    compute_tgl_max_train_steps,
    intervals_per_epoch,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional in some envs
    plt = None  # type: ignore[assignment]


OUTPUT_ROOT = artifact_root() / "throughput" / "frost_vs_tgl_scaling"
TGL_CONFIG_SRC = TGL_ROOT / "config" / "frost_parity" / "TGN.yml"
DATASET = "LASTFM"
MODEL = "tgn"
FROST_ADAPTER = FrostAdapter()
TGL_ADAPTER = TGLAdapter()


@dataclass(frozen=True)
class Job:
    n_gpu: int
    macro_batch_size: int


def render_progress(
    progress_path: Path,
    *,
    run_dir: Path,
    total_jobs: int,
    completed_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# TGL train_dist.py vs FROST TGN Throughput",
        "",
        f"- Run dir: {run_dir}",
        f"- Completed: {len(completed_rows)}/{total_jobs}",
        "",
        "## Completed Jobs",
        "",
    ]
    if completed_rows:
        for row in completed_rows:
            lines.append(
                "- "
                f"{row['n_gpu']} GPU / mb={row['macro_batch_size']}: "
                f"FROST={row['frost_bottleneck_events_per_sec_per_gpu']:.6f} ev/s/gpu, "
                f"TGL={row['tgl_bottleneck_events_per_sec_per_gpu']:.6f} ev/s/gpu, "
                f"speedup={row['speedup_frost_over_tgl']:.6f}x"
            )
    else:
        lines.append("- none yet")
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(summary_rows: list[dict[str, object]], run_dir: Path) -> list[Path]:
    if plt is None or not summary_rows:
        return []

    ordered_rows = sorted(
        summary_rows,
        key=lambda row: (int(row["n_gpu"]), int(row["macro_batch_size"])),
    )
    gpu_counts = sorted({int(row["n_gpu"]) for row in ordered_rows})
    systems = [
        ("frost_bottleneck_events_per_sec_per_gpu", "FROST", "#1b9e77"),
        ("tgl_bottleneck_events_per_sec_per_gpu", "TGL", "#d95f02"),
    ]

    fig, axes = plt.subplots(
        1, len(gpu_counts), figsize=(5.0 * len(gpu_counts), 4.0), sharey=True
    )
    if len(gpu_counts) == 1:
        axes = [axes]
    for ax, n_gpu in zip(axes, gpu_counts):
        rows = [row for row in ordered_rows if int(row["n_gpu"]) == n_gpu]
        x_values = [int(row["macro_batch_size"]) for row in rows]
        for key, label, color in systems:
            y = [float(row[key]) for row in rows]
            ax.plot(x_values, y, marker="o", linewidth=2.0, color=color, label=label)
        ax.set_title(f"{n_gpu} GPU")
        ax.set_xlabel("Macro batch size")
        ax.set_ylabel("Events / sec / GPU")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x_values)
        ax.legend()
    fig.suptitle("LASTFM + TGN: bottleneck-rank throughput per GPU")
    fig.tight_layout()
    per_gpu_path = run_dir / "throughput_per_gpu.png"
    fig.savefig(per_gpu_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(gpu_counts), figsize=(5.0 * len(gpu_counts), 4.0), sharey=True
    )
    if len(gpu_counts) == 1:
        axes = [axes]
    for ax, n_gpu in zip(axes, gpu_counts):
        rows = [row for row in ordered_rows if int(row["n_gpu"]) == n_gpu]
        x_values = [int(row["macro_batch_size"]) for row in rows]
        for key, label, color in [
            ("frost_bottleneck_events_per_sec_total", "FROST", "#1b9e77"),
            ("tgl_bottleneck_events_per_sec_total", "TGL", "#d95f02"),
        ]:
            y = [float(row[key]) for row in rows]
            ax.plot(x_values, y, marker="o", linewidth=2.0, color=color, label=label)
        ax.set_title(f"{n_gpu} GPU")
        ax.set_xlabel("Macro batch size")
        ax.set_ylabel("Aggregate events / sec")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x_values)
        ax.legend()
    fig.suptitle("LASTFM + TGN: aggregate throughput from bottleneck per-GPU rate")
    fig.tight_layout()
    total_path = run_dir / "throughput_total.png"
    fig.savefig(total_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for n_gpu, color in zip(gpu_counts, ["#377eb8", "#4daf4a", "#984ea3"]):
        rows = [row for row in ordered_rows if int(row["n_gpu"]) == n_gpu]
        x_values = [int(row["macro_batch_size"]) for row in rows]
        y = [float(row["speedup_frost_over_tgl"]) for row in rows]
        ax.plot(
            x_values, y, marker="o", linewidth=2.0, color=color, label=f"{n_gpu} GPU"
        )
    ax.set_title("FROST / TGL speedup on LASTFM + TGN")
    ax.set_xlabel("Macro batch size")
    ax.set_ylabel("Speedup")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(sorted({int(row["macro_batch_size"]) for row in ordered_rows}))
    ax.legend()
    fig.tight_layout()
    speedup_path = run_dir / "speedup_vs_batch.png"
    fig.savefig(speedup_path, dpi=180)
    plt.close(fig)

    return [per_gpu_path, total_path, speedup_path]


def write_report(
    report_path: Path,
    *,
    run_dir: Path,
    summary_rows: list[dict[str, object]],
    plot_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    actual_gpu_counts = sorted({int(row["n_gpu"]) for row in summary_rows})
    actual_batch_sizes = sorted({int(row["macro_batch_size"]) for row in summary_rows})
    lines = [
        "# FROST vs TGL train_dist.py Throughput",
        "",
        f"- Dataset: {DATASET}",
        f"- Model: {MODEL}",
        f"- GPU counts: {' '.join(str(v) for v in actual_gpu_counts)}",
        f"- Macro batch sizes: {' '.join(str(v) for v in actual_batch_sizes)}",
        f"- Warmup/measure: {args.warmup_batches}/{args.measure_batches}",
        f"- OMP_NUM_THREADS={args.omp_threads}, MKL_NUM_THREADS={args.mkl_threads}",
        f"- Seed: {args.seed}",
        f"- Run dir: {run_dir}",
        "",
        "## Summary",
        "",
        "| n_gpu | macro_batch_size | FROST ev/s/gpu | TGL ev/s/gpu | FROST total ev/s | TGL total ev/s | speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        summary_rows,
        key=lambda item: (int(item["n_gpu"]), int(item["macro_batch_size"])),
    ):
        lines.append(
            f"| {row['n_gpu']} | {row['macro_batch_size']} | "
            f"{float(row['frost_bottleneck_events_per_sec_per_gpu']):.6f} | "
            f"{float(row['tgl_bottleneck_events_per_sec_per_gpu']):.6f} | "
            f"{float(row['frost_bottleneck_events_per_sec_total']):.6f} | "
            f"{float(row['tgl_bottleneck_events_per_sec_total']):.6f} | "
            f"{float(row['speedup_frost_over_tgl']):.6f}x |"
        )
    if plot_paths:
        lines.extend(
            [
                "",
                "## Plots",
                "",
            ]
        )
        for path in plot_paths:
            lines.append(f"- {path.name}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serial LASTFM/TGN throughput benchmark for FROST vs TGL train_dist.py."
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Explicit job in the form n_gpu:macro_batch_size. Repeatable.",
    )
    parser.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[600, 1200, 2000, 4000]
    )
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=200)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--mkl-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=34900)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-plots", action="store_true")
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
    if not gpu_list:
        raise ValueError("--gpu-ids must contain at least one GPU id")

    jobs: list[Job] = []
    if args.job:
        for job_spec in args.job:
            if ":" not in job_spec:
                raise ValueError(
                    f"Invalid --job '{job_spec}'. Expected n_gpu:macro_batch_size."
                )
            n_gpu_str, batch_size_str = job_spec.split(":", 1)
            n_gpu = int(n_gpu_str)
            batch_size = int(batch_size_str)
            jobs.append(Job(n_gpu=n_gpu, macro_batch_size=batch_size))
    else:
        jobs = [
            Job(n_gpu=n_gpu, macro_batch_size=macro_batch_size)
            for n_gpu in args.gpu_counts
            for macro_batch_size in args.batch_sizes
        ]

    for job in jobs:
        if job.n_gpu < 1:
            raise ValueError(f"Invalid n_gpu={job.n_gpu}")
        if job.n_gpu > len(gpu_list):
            raise ValueError(
                f"Requested n_gpu={job.n_gpu}, but only {len(gpu_list)} ids were provided in --gpu-ids"
            )
        if job.macro_batch_size < 1:
            raise ValueError(f"Invalid batch size {job.macro_batch_size}")
        if job.macro_batch_size % job.n_gpu != 0:
            raise ValueError(
                f"macro_batch_size={job.macro_batch_size} is not divisible by n_gpu={job.n_gpu}"
            )
        epoch_intervals = intervals_per_epoch(DATASET, job.macro_batch_size)
        if args.warmup_batches + args.measure_batches > epoch_intervals:
            raise ValueError(
                "Requested single-epoch measurement window does not fit: "
                f"macro_batch_size={job.macro_batch_size}, n_gpu={job.n_gpu}, "
                f"intervals_per_epoch={epoch_intervals}, "
                f"warmup+measure={args.warmup_batches + args.measure_batches}"
            )

    run_name = args.run_name or f"lastfm_tgn_scaling_{utc_timestamp()}"
    run_dir = args.output_root / run_name
    cfg_dir = run_dir / "configs"
    log_dir = run_dir / "logs"
    ensure_dir(cfg_dir)
    ensure_dir(log_dir)

    results_tsv = run_dir / "results.tsv"
    summary_tsv = run_dir / "summary.tsv"
    report_md = run_dir / "report.md"
    progress_md = run_dir / "progress.md"

    completed_rows: list[dict[str, object]] = []
    render_progress(
        progress_md,
        run_dir=run_dir,
        total_jobs=len(jobs),
        completed_rows=completed_rows,
    )

    print(f"Writing outputs to {run_dir}")
    print(f"GPU ids: {args.gpu_ids}")
    if args.job:
        print(f"Jobs: {[f'{job.n_gpu}:{job.macro_batch_size}' for job in jobs]}")
    else:
        print(f"GPU counts: {args.gpu_counts}")
        print(f"Batch sizes: {args.batch_sizes}")
    print(
        f"Warmup/measure: {args.warmup_batches}/{args.measure_batches}, "
        f"OMP={args.omp_threads}, MKL={args.mkl_threads}"
    )

    for idx, job in enumerate(jobs):
        gpu_ids = ",".join(gpu_list[: job.n_gpu])
        per_gpu_batch_size = job.macro_batch_size // job.n_gpu
        frost_epochs = compute_frost_epochs(
            DATASET, job.macro_batch_size, args.warmup_batches, args.measure_batches
        )
        tgl_max_train_steps = compute_tgl_max_train_steps(
            DATASET,
            per_gpu_batch_size,
            job.n_gpu,
            args.warmup_batches,
            args.measure_batches,
        )
        master_port = args.master_port_base + idx
        tgl_cfg = (
            cfg_dir / f"tgl_tgn_lastfm_ngpu{job.n_gpu}_mb{job.macro_batch_size}.yml"
        )
        frost_log = (
            log_dir / f"frost_tgn_lastfm_ngpu{job.n_gpu}_mb{job.macro_batch_size}.log"
        )
        tgl_log = (
            log_dir / f"tgl_tgn_lastfm_ngpu{job.n_gpu}_mb{job.macro_batch_size}.log"
        )

        TGL_ADAPTER.write_config(
            source=TGL_CONFIG_SRC,
            destination=tgl_cfg,
            epochs=frost_epochs,
            per_gpu_batch_size=per_gpu_batch_size,
        )

        frost_run = FROST_ADAPTER.prepare(
            ExperimentSpec(
                model=MODEL,
                dataset=DATASET,
                seed=args.seed,
                n_gpus=job.n_gpu,
                batch_size=job.macro_batch_size,
                batch_size_scope=BatchSizeScope.GLOBAL,
                epochs=frost_epochs,
                extra_args=("tqdm=true",),
            ),
            gpu_ids=gpu_ids,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            master_port=master_port,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
        )
        tgl_run = TGL_ADAPTER.prepare(
            ExperimentSpec(
                model=MODEL,
                dataset=DATASET,
                seed=args.seed,
                n_gpus=job.n_gpu,
                batch_size=per_gpu_batch_size,
                batch_size_scope=BatchSizeScope.PER_GPU,
                epochs=frost_epochs,
                extra_args=("--tqdm",),
            ),
            gpu_ids=gpu_ids,
            config_path=tgl_cfg,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            max_train_steps=tgl_max_train_steps,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
        )

        print()
        print(
            f"[{idx + 1}/{len(jobs)}] {job.n_gpu} GPU / mb={job.macro_batch_size} "
            f"(per_gpu_batch_size={per_gpu_batch_size}, frost_epochs={frost_epochs}, "
            f"tgl_max_train_steps={tgl_max_train_steps})"
        )
        print(f"[FROST] log -> {frost_log}")
        run_prepared(frost_run, frost_log)
        print(f"[TGL] log -> {tgl_log}")
        run_prepared(tgl_run, tgl_log)

        frost_text = read_text(frost_log)
        tgl_text = read_text(tgl_log)
        frost_by_rank = parse_frost_rank_throughput(frost_text)
        tgl_by_rank = parse_tgl_rank_throughput(tgl_text, n_gpu=job.n_gpu)

        frost_rank, frost_eps, frost_measured = select_bottleneck_rank(frost_by_rank)
        tgl_rank, tgl_eps, tgl_measured = select_bottleneck_rank(tgl_by_rank)

        row: dict[str, object] = {
            "dataset": DATASET,
            "model": MODEL,
            "seed": args.seed,
            "n_gpu": job.n_gpu,
            "macro_batch_size": job.macro_batch_size,
            "per_gpu_batch_size": per_gpu_batch_size,
            "warmup_batches": args.warmup_batches,
            "measure_batches": args.measure_batches,
            "frost_epochs": frost_epochs,
            "tgl_max_train_steps": tgl_max_train_steps,
            "frost_bottleneck_rank": frost_rank,
            "tgl_bottleneck_rank": tgl_rank,
            "frost_bottleneck_events_per_sec_per_gpu": round(frost_eps, 6),
            "tgl_bottleneck_events_per_sec_per_gpu": round(tgl_eps, 6),
            "frost_bottleneck_events_per_sec_total": round(frost_eps * job.n_gpu, 6),
            "tgl_bottleneck_events_per_sec_total": round(tgl_eps * job.n_gpu, 6),
            "frost_mean_events_per_sec_per_gpu": round(
                mean_events_per_sec(frost_by_rank), 6
            ),
            "tgl_mean_events_per_sec_per_gpu": round(
                mean_events_per_sec(tgl_by_rank), 6
            ),
            "speedup_frost_over_tgl": round(frost_eps / tgl_eps, 6),
            "frost_measured_batches": frost_measured,
            "tgl_measured_batches": tgl_measured,
            "frost_log": str(frost_log),
            "tgl_log": str(tgl_log),
        }
        completed_rows.append(row)
        write_rows_tsv(results_tsv, completed_rows)
        write_rows_tsv(summary_tsv, completed_rows)
        render_progress(
            progress_md,
            run_dir=run_dir,
            total_jobs=len(jobs),
            completed_rows=completed_rows,
        )
        print(
            "[SUMMARY] "
            f"FROST={frost_eps:.6f} ev/s/gpu, "
            f"TGL={tgl_eps:.6f} ev/s/gpu, "
            f"speedup={frost_eps / tgl_eps:.6f}x"
        )

    plot_paths = [] if args.skip_plots else plot_results(completed_rows, run_dir)
    write_report(
        report_md,
        run_dir=run_dir,
        summary_rows=completed_rows,
        plot_paths=plot_paths,
        args=args,
    )

    latest_link = args.output_root / "latest"
    if latest_link.is_symlink() or latest_link.is_file():
        latest_link.unlink()
    elif latest_link.is_dir():
        shutil.rmtree(latest_link)
    latest_link.symlink_to(run_dir.name)

    print()
    print(f"Results TSV: {results_tsv}")
    print(f"Summary TSV: {summary_tsv}")
    print(f"Progress MD: {progress_md}")
    print(f"Report MD: {report_md}")
    if plot_paths:
        print("Plots:")
        for path in plot_paths:
            print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
