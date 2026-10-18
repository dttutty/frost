#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_PATTERN = re.compile(r"\[Epoch\s+(\d+)\]")
NUM_PATTERN = r"(?:[0-9]+\.[0-9]+|[0-9]+(?!\.))"
VAL_AP_PATTERN = re.compile(rf"Val AP(?:\(New high\))?:\s*({NUM_PATTERN})")
VAL_AUC_PATTERN = re.compile(rf"Val AUC:\s*({NUM_PATTERN})")
TEST_AUC_PATTERN = re.compile(rf"Test AUC:\s*({NUM_PATTERN})")
TEST_AP_PATTERN = re.compile(rf"Test AP:\s*({NUM_PATTERN})")
BEST_PATTERN = re.compile(
    rf"Best Val AP:\s*({NUM_PATTERN})\s*@ Epoch\s+(\d+)\s*\|\s*"
    rf"Val AUC:\s*({NUM_PATTERN})\s*\|\s*"
    rf"Test AP:\s*({NUM_PATTERN})\s*\|\s*"
    rf"Test AUC:\s*({NUM_PATTERN})"
)


@dataclass(frozen=True)
class SeedSummary:
    best_epoch: int
    best_val_ap: float
    best_val_auc: float
    best_test_ap: float
    best_test_auc: float
    epochs_ran: int
    source: str


@dataclass(frozen=True)
class LogMetrics:
    epochs: list[int]
    val_ap: list[float | None]
    val_auc: list[float | None]
    test_ap: list[float | None]
    test_auc: list[float | None]
    seed_summary: SeedSummary | None


def parse_log_file(log_path: Path) -> LogMetrics:
    """Parse one seed log into epoch-wise AP/AUC series and a seed summary."""
    val_ap_by_epoch: dict[int, float] = {}
    val_auc_by_epoch: dict[int, float] = {}
    test_ap_by_epoch: dict[int, float] = {}
    test_auc_by_epoch: dict[int, float] = {}
    final_summary: SeedSummary | None = None

    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    epoch_marks = list(EPOCH_PATTERN.finditer(text))
    for idx, mark in enumerate(epoch_marks):
        epoch = int(mark.group(1))
        end = epoch_marks[idx + 1].start() if idx + 1 < len(epoch_marks) else len(text)
        chunk = text[mark.start() : end]

        val_ap_matches = VAL_AP_PATTERN.findall(chunk)
        if val_ap_matches:
            val_ap_by_epoch[epoch] = float(val_ap_matches[-1])

        val_auc_matches = VAL_AUC_PATTERN.findall(chunk)
        if val_auc_matches:
            val_auc_by_epoch[epoch] = float(val_auc_matches[-1])

        test_auc_matches = TEST_AUC_PATTERN.findall(chunk)
        if test_auc_matches:
            test_auc_by_epoch[epoch] = float(test_auc_matches[-1])

        test_ap_matches = TEST_AP_PATTERN.findall(chunk)
        if test_ap_matches:
            test_ap_by_epoch[epoch] = float(test_ap_matches[-1])

    for line in text.splitlines():
        best_match = BEST_PATTERN.search(line)
        if best_match:
            final_summary = SeedSummary(
                best_val_ap=float(best_match.group(1)),
                best_epoch=int(best_match.group(2)),
                best_val_auc=float(best_match.group(3)),
                best_test_ap=float(best_match.group(4)),
                best_test_auc=float(best_match.group(5)),
                epochs_ran=0,
                source="final_summary",
            )

    epochs = sorted(
        set(val_ap_by_epoch)
        | set(val_auc_by_epoch)
        | set(test_ap_by_epoch)
        | set(test_auc_by_epoch)
    )
    if not epochs:
        return LogMetrics(
            epochs=[],
            val_ap=[],
            val_auc=[],
            test_ap=[],
            test_auc=[],
            seed_summary=None,
        )

    val_ap = [val_ap_by_epoch.get(e) for e in epochs]
    val_auc = [val_auc_by_epoch.get(e) for e in epochs]
    test_ap = [test_ap_by_epoch.get(e) for e in epochs]
    test_auc = [test_auc_by_epoch.get(e) for e in epochs]
    epochs_ran = epochs[-1]

    if final_summary is None:
        candidate_indices = [
            idx
            for idx in range(len(epochs))
            if val_ap[idx] is not None and test_ap[idx] is not None and test_auc[idx] is not None
        ]
        if candidate_indices:
            best_idx = max(candidate_indices, key=lambda idx: (val_ap[idx], -epochs[idx]))
            seed_summary = SeedSummary(
                best_epoch=epochs[best_idx],
                best_val_ap=val_ap[best_idx],
                best_val_auc=val_auc[best_idx] if val_auc[best_idx] is not None else math.nan,
                best_test_ap=test_ap[best_idx],
                best_test_auc=test_auc[best_idx],
                epochs_ran=epochs_ran,
                source="derived_from_epoch_lines",
            )
        else:
            seed_summary = None
    else:
        seed_summary = SeedSummary(
            best_epoch=final_summary.best_epoch,
            best_val_ap=final_summary.best_val_ap,
            best_val_auc=final_summary.best_val_auc,
            best_test_ap=final_summary.best_test_ap,
            best_test_auc=final_summary.best_test_auc,
            epochs_ran=epochs_ran,
            source=final_summary.source,
        )

    return LogMetrics(
        epochs=epochs,
        val_ap=val_ap,
        val_auc=val_auc,
        test_ap=test_ap,
        test_auc=test_auc,
        seed_summary=seed_summary,
    )


def _mark_best_epoch(ax: plt.Axes, epoch: int, value: float, color: str, label: str) -> None:
    ax.axvline(
        epoch,
        color=color,
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label=label,
    )
    ax.scatter([epoch], [value], color=color, s=34, zorder=3)


def plot_one_log(
    log_path: Path,
    out_path: Path,
    dpi: int,
    mark_val_max: bool,
) -> tuple[bool, str, LogMetrics]:
    metrics = parse_log_file(log_path)
    if not metrics.epochs:
        return False, f"[WARN] Skip {log_path.name}: no metrics parsed", metrics

    fig, (ax_ap, ax_auc) = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.0),
        sharex=True,
    )

    val_ap_points = [
        (epoch, value) for epoch, value in zip(metrics.epochs, metrics.val_ap) if value is not None
    ]
    test_ap_points = [
        (epoch, value) for epoch, value in zip(metrics.epochs, metrics.test_ap) if value is not None
    ]
    val_auc_points = [
        (epoch, value) for epoch, value in zip(metrics.epochs, metrics.val_auc) if value is not None
    ]
    test_auc_points = [
        (epoch, value) for epoch, value in zip(metrics.epochs, metrics.test_auc) if value is not None
    ]

    if val_ap_points:
        e_val_ap, y_val_ap = zip(*val_ap_points)
        ax_ap.plot(e_val_ap, y_val_ap, label="Val AP", color="#1f77b4", linewidth=1.9)
    if test_ap_points:
        e_test_ap, y_test_ap = zip(*test_ap_points)
        ax_ap.plot(
            e_test_ap,
            y_test_ap,
            label="Test AP",
            color="#ff7f0e",
            linewidth=1.9,
        )
    ax_ap.set_ylabel("AP")
    ax_ap.grid(True, alpha=0.3)
    if val_ap_points or test_ap_points:
        ax_ap.legend()

    if val_auc_points:
        e_val_auc, y_val_auc = zip(*val_auc_points)
        ax_auc.plot(
            e_val_auc,
            y_val_auc,
            label="Val AUC",
            color="#2ca02c",
            linewidth=1.9,
        )
    if test_auc_points:
        e_test_auc, y_test_auc = zip(*test_auc_points)
        ax_auc.plot(
            e_test_auc,
            y_test_auc,
            label="Test AUC",
            color="#d62728",
            linewidth=1.9,
        )
    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("AUC")
    ax_auc.grid(True, alpha=0.3)
    if val_auc_points or test_auc_points:
        ax_auc.legend()

    if mark_val_max and metrics.seed_summary is not None:
        summary = metrics.seed_summary
        _mark_best_epoch(
            ax_ap,
            epoch=summary.best_epoch,
            value=summary.best_val_ap,
            color="#1f77b4",
            label=f"Best Val AP @ epoch {summary.best_epoch}",
        )
        ax_ap.text(
            summary.best_epoch + 0.7,
            summary.best_val_ap + 0.001,
            f"{summary.best_val_ap:.4f}",
            color="#1f77b4",
            fontsize=9,
        )
        if not math.isnan(summary.best_val_auc):
            _mark_best_epoch(
                ax_auc,
                epoch=summary.best_epoch,
                value=summary.best_val_auc,
                color="#2ca02c",
                label=f"Val AUC @ best epoch {summary.best_epoch}",
            )
            ax_auc.text(
                summary.best_epoch + 0.7,
                summary.best_val_auc + 0.001,
                f"{summary.best_val_auc:.4f}",
                color="#2ca02c",
                fontsize=9,
            )

    fig.suptitle(f"{log_path.stem}: Val/Test AP and AUC vs Epoch")
    fig.subplots_adjust(hspace=0.28, top=0.93)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True, f"Saved: {out_path}", metrics


def find_log_directories(input_dir: Path, glob_pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted({path.parent for path in input_dir.rglob(glob_pattern)})
    return [input_dir]


def build_summary_text(input_dir: Path, metrics_by_log: dict[Path, LogMetrics]) -> str:
    lines = [
        f"Directory: {input_dir}",
        f"Log files: {len(metrics_by_log)}",
    ]

    complete = 0
    derived = 0
    missing = 0
    test_ap_values: list[float] = []

    for metrics in metrics_by_log.values():
        if metrics.seed_summary is None:
            missing += 1
            continue
        test_ap_values.append(metrics.seed_summary.best_test_ap)
        if metrics.seed_summary.source == "final_summary":
            complete += 1
        else:
            derived += 1

    lines.append(f"Completed from final summary: {complete}")
    lines.append(f"Derived from epoch lines: {derived}")
    lines.append(f"Missing metrics: {missing}")

    if test_ap_values:
        mean_test_ap = statistics.fmean(test_ap_values)
        std_test_ap = statistics.stdev(test_ap_values) if len(test_ap_values) > 1 else 0.0
        lines.append(f"Mean test AP: {mean_test_ap:.6f}")
        lines.append(f"Std test AP: {std_test_ap:.6f}")
    else:
        lines.append("Mean test AP: n/a")
        lines.append("Std test AP: n/a")

    lines.append("")
    lines.append("Per-seed results:")
    for log_path in sorted(metrics_by_log):
        metrics = metrics_by_log[log_path]
        summary = metrics.seed_summary
        if summary is None:
            lines.append(f"{log_path.name}: no metrics parsed")
            continue
        lines.append(
            f"{log_path.name}: "
            f"test_ap={summary.best_test_ap:.4f}, "
            f"test_auc={summary.best_test_auc:.4f}, "
            f"best_val_ap={summary.best_val_ap:.4f}, "
            f"best_epoch={summary.best_epoch}, "
            f"epochs_ran={summary.epochs_ran}, "
            f"source={summary.source}"
        )

    lines.append("")
    return "\n".join(lines)


def process_directory(args: argparse.Namespace, input_dir: Path) -> tuple[int, int]:
    log_files = sorted(input_dir.glob(args.glob))
    if not log_files:
        print(f"[SKIP] No log files found in {input_dir} with glob '{args.glob}'")
        return 0, 0

    print(f"[DIR] {input_dir}")
    saved = 0
    skipped = 0
    metrics_by_log: dict[Path, LogMetrics] = {}

    for log_path in log_files:
        out_path = input_dir / f"{log_path.stem}{args.out_suffix}"
        metrics = parse_log_file(log_path)
        metrics_by_log[log_path] = metrics

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {out_path}")
            skipped += 1
            continue

        ok, message, metrics = plot_one_log(
            log_path=log_path,
            out_path=out_path,
            dpi=args.dpi,
            mark_val_max=args.mark_val_max,
        )
        metrics_by_log[log_path] = metrics
        print(message)
        if ok:
            saved += 1
        else:
            skipped += 1

    if args.write_summary:
        summary_path = input_dir / args.summary_name
        summary_path.write_text(build_summary_text(input_dir, metrics_by_log))
        print(f"Saved: {summary_path}")

    return saved, skipped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Val/Test AP+AUC curves from seed logs and optionally write per-directory summaries."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing seed log files, or a root directory when --recursive is used.",
    )
    parser.add_argument(
        "--glob",
        default="seed_*.log",
        help="Glob pattern for log files inside each input directory (default: seed_*.log).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process every subdirectory under input_dir that contains matching logs.",
    )
    parser.add_argument(
        "--out-suffix",
        default="_val_test_ap_auc_vs_epoch.png",
        help="Suffix for output image file names.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI (default: 220).",
    )
    parser.add_argument(
        "--mark-val-max",
        action="store_true",
        help="Draw a vertical dashed line at the epoch where Val AP is maximal.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write summary.txt-style statistics for each processed directory.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.txt",
        help="Output file name for per-directory summaries (default: summary.txt).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Invalid input directory: {input_dir}")
        return 1

    target_dirs = find_log_directories(input_dir, args.glob, args.recursive)
    if not target_dirs:
        print(f"[ERROR] No directories found under {input_dir} with glob '{args.glob}'")
        return 1

    total_saved = 0
    total_skipped = 0
    processed_dirs = 0
    for directory in target_dirs:
        log_files = sorted(directory.glob(args.glob))
        if not log_files:
            continue
        processed_dirs += 1
        saved, skipped = process_directory(args, directory)
        total_saved += saved
        total_skipped += skipped

    if processed_dirs == 0:
        print(f"[ERROR] No log files found in {input_dir} with glob '{args.glob}'")
        return 1

    print(
        "Done. "
        f"dirs={processed_dirs}, saved={total_saved}, skipped={total_skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
