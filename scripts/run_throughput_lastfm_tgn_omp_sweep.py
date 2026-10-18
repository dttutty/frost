#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

from frost_bench.io import ensure_dir, utc_timestamp

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "scripts" / "run_throughput_lastfm_tgn_frost_vs_tgl_scaling.py"
OUTPUT_ROOT = REPO_ROOT / "experiments" / "com_throughput_with_tgl_train_dist_omp_sweep"


def read_tsv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one result row in {path}, found {len(rows)}")
    return rows[0]


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def render_progress(path: Path, *, run_dir: Path, total_runs: int, rows: list[dict[str, object]]) -> None:
    lines = [
        "# LASTFM/TGN 4GPU OMP/MKL Sweep",
        "",
        f"- Run dir: {run_dir}",
        f"- Completed: {len(rows)}/{total_runs}",
        "",
        "## Completed Runs",
        "",
    ]
    if not rows:
        lines.append("- none yet")
    else:
        for row in rows:
            lines.append(
                "- "
                f"OMP=MKL={row['omp_threads']}: "
                f"FROST={float(row['frost_events_per_sec_per_gpu']):.6f} ev/s/gpu, "
                f"TGL={float(row['tgl_events_per_sec_per_gpu']):.6f} ev/s/gpu, "
                f"speedup={float(row['speedup_frost_over_tgl']):.6f}x"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(rows: list[dict[str, object]], run_dir: Path) -> list[Path]:
    if plt is None or not rows:
        return []

    ordered = sorted(rows, key=lambda row: int(row["omp_threads"]))
    x = [int(row["omp_threads"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        x,
        [float(row["frost_events_per_sec_per_gpu"]) for row in ordered],
        marker="o",
        linewidth=2.0,
        color="#1b9e77",
        label="FROST",
    )
    ax.plot(
        x,
        [float(row["tgl_events_per_sec_per_gpu"]) for row in ordered],
        marker="o",
        linewidth=2.0,
        color="#d95f02",
        label="TGL",
    )
    ax.set_title("4 GPU / macro-batch 8000: throughput per GPU")
    ax.set_xlabel("OMP_NUM_THREADS = MKL_NUM_THREADS")
    ax.set_ylabel("Events / sec / GPU")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    per_gpu_path = run_dir / "throughput_per_gpu_vs_omp.png"
    fig.savefig(per_gpu_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        x,
        [float(row["frost_events_per_sec_total"]) for row in ordered],
        marker="o",
        linewidth=2.0,
        color="#1b9e77",
        label="FROST",
    )
    ax.plot(
        x,
        [float(row["tgl_events_per_sec_total"]) for row in ordered],
        marker="o",
        linewidth=2.0,
        color="#d95f02",
        label="TGL",
    )
    ax.set_title("4 GPU / macro-batch 8000: aggregate throughput")
    ax.set_xlabel("OMP_NUM_THREADS = MKL_NUM_THREADS")
    ax.set_ylabel("Aggregate events / sec")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    total_path = run_dir / "throughput_total_vs_omp.png"
    fig.savefig(total_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        x,
        [float(row["speedup_frost_over_tgl"]) for row in ordered],
        marker="o",
        linewidth=2.0,
        color="#377eb8",
    )
    ax.set_title("4 GPU / macro-batch 8000: FROST over TGL speedup")
    ax.set_xlabel("OMP_NUM_THREADS = MKL_NUM_THREADS")
    ax.set_ylabel("Speedup")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    speedup_path = run_dir / "speedup_vs_omp.png"
    fig.savefig(speedup_path, dpi=180)
    plt.close(fig)

    return [per_gpu_path, total_path, speedup_path]


def write_report(path: Path, *, run_dir: Path, rows: list[dict[str, object]], plot_paths: list[Path], args: argparse.Namespace) -> None:
    ordered = sorted(rows, key=lambda row: int(row["omp_threads"]))
    lines = [
        "# LASTFM/TGN 4GPU OMP/MKL Sweep",
        "",
        "- Dataset: LASTFM",
        "- Model: tgn",
        "- GPUs: 4",
        "- Macro batch size: 8000",
        f"- Warmup/measure: {args.warmup_batches}/{args.measure_batches}",
        "- Constraint: single-epoch measurement window only",
        f"- Run dir: {run_dir}",
        "",
        "## Summary",
        "",
        "| OMP=MKL | FROST ev/s/gpu | TGL ev/s/gpu | FROST total ev/s | TGL total ev/s | speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            f"| {row['omp_threads']} | "
            f"{float(row['frost_events_per_sec_per_gpu']):.6f} | "
            f"{float(row['tgl_events_per_sec_per_gpu']):.6f} | "
            f"{float(row['frost_events_per_sec_total']):.6f} | "
            f"{float(row['tgl_events_per_sec_total']):.6f} | "
            f"{float(row['speedup_frost_over_tgl']):.6f}x |"
        )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for plot_path in plot_paths:
            lines.append(f"- {plot_path.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep OMP/MKL thread counts for the LASTFM/TGN 4GPU macro-batch-8000 benchmark."
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--omp-values", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measure-batches", type=int, default=50)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_name = args.run_name or f"lastfm_tgn_4gpu_mb8000_omp_sweep_{utc_timestamp()}"
    run_dir = args.output_root / run_name
    ensure_dir(run_dir)
    progress_md = run_dir / "progress.md"
    results_tsv = run_dir / "results.tsv"
    report_md = run_dir / "report.md"
    launcher_log = run_dir / "launcher.log"

    rows: list[dict[str, object]] = []
    render_progress(progress_md, run_dir=run_dir, total_runs=len(args.omp_values), rows=rows)

    with launcher_log.open("w", encoding="utf-8") as log_handle:
        for omp_threads in args.omp_values:
            sub_run_name = f"{run_name}_omp{omp_threads}"
            cmd = [
                "uv",
                "run",
                "python",
                str(BASE_SCRIPT),
                "--run-name",
                sub_run_name,
                "--gpu-ids",
                args.gpu_ids,
                "--job",
                "4:8000",
                "--warmup-batches",
                str(args.warmup_batches),
                "--measure-batches",
                str(args.measure_batches),
                "--omp-threads",
                str(omp_threads),
                "--mkl-threads",
                str(omp_threads),
            ]
            print(f"[RUN] OMP=MKL={omp_threads}", file=log_handle, flush=True)
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            sub_run_dir = REPO_ROOT / "experiments" / "com_throughput_with_tgl_train_dist_scaling" / sub_run_name
            row = read_tsv_row(sub_run_dir / "results.tsv")
            rows.append(
                {
                    "omp_threads": omp_threads,
                    "mkl_threads": omp_threads,
                    "frost_events_per_sec_per_gpu": float(row["frost_bottleneck_events_per_sec_per_gpu"]),
                    "tgl_events_per_sec_per_gpu": float(row["tgl_bottleneck_events_per_sec_per_gpu"]),
                    "frost_events_per_sec_total": float(row["frost_bottleneck_events_per_sec_total"]),
                    "tgl_events_per_sec_total": float(row["tgl_bottleneck_events_per_sec_total"]),
                    "speedup_frost_over_tgl": float(row["speedup_frost_over_tgl"]),
                    "warmup_batches": int(row["warmup_batches"]),
                    "measure_batches": int(row["measure_batches"]),
                    "sub_run_dir": str(sub_run_dir),
                    "sub_report": str(sub_run_dir / "report.md"),
                }
            )
            write_tsv(results_tsv, rows)
            render_progress(progress_md, run_dir=run_dir, total_runs=len(args.omp_values), rows=rows)

    plot_paths = [] if args.skip_plots else plot_results(rows, run_dir)
    write_report(report_md, run_dir=run_dir, rows=rows, plot_paths=plot_paths, args=args)

    latest_link = args.output_root / "latest"
    if latest_link.is_symlink() or latest_link.is_file():
        latest_link.unlink()
    elif latest_link.is_dir():
        shutil.rmtree(latest_link)
    latest_link.symlink_to(run_dir.name)

    print(f"Results TSV: {results_tsv}")
    print(f"Progress MD: {progress_md}")
    print(f"Report MD: {report_md}")
    if plot_paths:
        print("Plots:")
        for plot_path in plot_paths:
            print(f"- {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
