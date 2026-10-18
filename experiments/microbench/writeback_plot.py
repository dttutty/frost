#!/usr/bin/env python3
"""Plot writeback benchmark CSV.

Default behavior reads and writes in the same directory as this script:
- input CSV: writeback.csv
- output PNG: writeback.png
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_csv_path(raw: str) -> Path:
    if raw.lower() == "auto":
        return _script_dir() / "writeback.csv"
    return Path(raw).expanduser().resolve()


def _resolve_plot_path(raw: str, csv_path: Path) -> Path:
    if raw.lower() == "auto":
        return csv_path.with_suffix(".png")
    return Path(raw).expanduser().resolve()


def _read_csv_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "size_bytes": int(float(row["size_bytes"])),
                    "size_mb": float(row["size_mb"]),
                    "avg_ms": float(row["avg_ms"]),
                    "p50_ms": float(row["p50_ms"]),
                    "p95_ms": float(row["p95_ms"]),
                    "gbps": float(row["gbps"]),
                }
            )
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")
    return rows


def _plot_rows(rows: list[dict], plot_path: Path) -> None:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    size_mb = [row["size_mb"] for row in rows]
    avg_ms = [row["avg_ms"] for row in rows]
    p50_ms = [row["p50_ms"] for row in rows]
    p95_ms = [row["p95_ms"] for row in rows]
    gbps = [row["gbps"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(size_mb, avg_ms, marker="o", label="avg_ms")
    axes[0].plot(size_mb, p50_ms, marker="s", label="p50_ms")
    axes[0].plot(size_mb, p95_ms, marker="^", label="p95_ms")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Writeback latency vs payload size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(size_mb, gbps, marker="o", color="tab:green", label="GB/s")
    axes[1].set_xlabel("Payload size (MB)")
    axes[1].set_ylabel("Bandwidth (GB/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plot_path), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot writeback benchmark results from CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="auto",
        help="Input CSV path. Use 'auto' for script-dir default",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="auto",
        help="Output PNG path. Use 'auto' to place beside CSV",
    )
    args = parser.parse_args()

    csv_path = _resolve_csv_path(args.csv)
    plot_path = _resolve_plot_path(args.plot_path, csv_path)

    rows = _read_csv_rows(csv_path)
    _plot_rows(rows, plot_path)

    print(f"# plotted csv: {csv_path}")
    print(f"# wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
