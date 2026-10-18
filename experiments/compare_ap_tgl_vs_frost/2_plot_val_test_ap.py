#!/usr/bin/env python3
"""Plot Val AP and Test AP over epochs for compare_ap_tgl_vs_frost logs."""

from __future__ import annotations

import argparse
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

FLOAT_PATTERN = r"[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
EPOCH_LINE_PATTERN = re.compile(r"\[Epoch\s+(?P<epoch>\d+)\].*", flags=re.IGNORECASE)
VAL_AP_PATTERN = re.compile(
    rf"\bVal AP(?:\(New high\))?:\s+(?P<val_ap>{FLOAT_PATTERN})",
    flags=re.IGNORECASE,
)
TEST_AP_PATTERN = re.compile(
    rf"\bTest AP:\s+(?P<test_ap>{FLOAT_PATTERN})",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class EpochMetric:
    epoch: int
    val_ap: float
    test_ap: float


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_log_dir(raw: str) -> Path:
    if raw.lower() == "auto":
        return _script_dir() / "logs"
    return Path(raw).expanduser().resolve()


def _parse_one_log(log_path: Path) -> list[EpochMetric]:
    metrics: list[EpochMetric] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = EPOCH_LINE_PATTERN.search(line)
        if epoch_match is None:
            continue

        val_ap_match = VAL_AP_PATTERN.search(line)
        test_ap_match = TEST_AP_PATTERN.search(line)
        if val_ap_match is None or test_ap_match is None:
            continue

        metrics.append(
            EpochMetric(
                epoch=int(epoch_match.group("epoch")),
                val_ap=float(val_ap_match.group("val_ap")),
                test_ap=float(test_ap_match.group("test_ap")),
            )
        )
    if not metrics:
        raise ValueError(f"no epoch AP lines found in {log_path}")
    return metrics


def _output_paths(log_path: Path, formats: tuple[str, ...]) -> list[Path]:
    base = log_path.with_suffix("")
    return [base.with_name(f"{base.name}__val_test_ap.{fmt}") for fmt in formats]


def _build_title(log_path: Path, metrics: list[EpochMetric]) -> str:
    best = max(metrics, key=lambda item: item.val_ap)
    run_name = log_path.stem.replace("__", " | ")
    return (
        f"{run_name}\n"
        f"Best Val AP {best.val_ap:.4f} @ Epoch {best.epoch} | "
        f"Test AP {best.test_ap:.4f}"
    )


def _plot_one_log(
    log_path: Path,
    metrics: list[EpochMetric],
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    epochs = [item.epoch for item in metrics]
    val_ap = [item.val_ap for item in metrics]
    test_ap = [item.test_ap for item in metrics]
    best = max(metrics, key=lambda item: item.val_ap)

    y_min = max(0.0, min(min(val_ap), min(test_ap)) - 0.03)
    y_max = min(1.0, max(max(val_ap), max(test_ap)) + 0.03)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, val_ap, color="tab:blue", linewidth=2.2, label="Val AP")
    ax.plot(epochs, test_ap, color="tab:orange", linewidth=2.0, label="Test AP")
    ax.scatter(
        [best.epoch],
        [best.val_ap],
        color="tab:blue",
        s=45,
        zorder=3,
        label=f"Best Val AP ({best.epoch}, {best.val_ap:.4f})",
    )
    ax.axvline(best.epoch, color="tab:blue", linestyle="--", linewidth=1, alpha=0.25)
    ax.set_title(_build_title(log_path, metrics))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_paths = _output_paths(log_path, formats)
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"dpi": dpi} if output_path.suffix.lower() == ".png" else {}
        fig.savefig(str(output_path), bbox_inches="tight", **save_kwargs)
    plt.close(fig)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Val AP and Test AP over epochs for FROST logs."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="auto",
        help="Directory containing .log files. Use 'auto' for script-dir logs/",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "svg"),
        help="Output formats for each plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG DPI. Ignored for vector formats such as svg.",
    )
    args = parser.parse_args()

    log_dir = _resolve_log_dir(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log_dir does not exist: {log_dir}")

    formats = tuple(fmt.lower().lstrip(".") for fmt in args.formats)
    if not formats:
        raise SystemExit("at least one output format is required")

    log_files = sorted(path for path in log_dir.glob("*.log") if path.is_file())
    if not log_files:
        raise SystemExit(f"no .log files found in {log_dir}")

    plotted = 0
    skipped: list[tuple[Path, str]] = []
    for log_path in log_files:
        try:
            metrics = _parse_one_log(log_path)
        except ValueError as exc:
            skipped.append((log_path, str(exc)))
            print(f"# skipped log: {log_path}")
            print(f"# reason: {exc}")
            continue

        output_paths = _plot_one_log(log_path, metrics, formats=formats, dpi=args.dpi)
        plotted += 1
        print(f"# plotted log: {log_path}")
        for output_path in output_paths:
            print(f"# wrote plot: {output_path}")

    print(f"# total logs plotted: {plotted}")
    if skipped:
        print(f"# total logs skipped: {len(skipped)}")
        for log_path, reason in skipped:
            print(f"# skipped summary: {log_path} | {reason}")


if __name__ == "__main__":
    main()
