#!/usr/bin/env python3
"""Plot writeback+random-read shm benchmark CSV.

Default behavior reads and writes in the same directory as this script:
- input CSV: writeback_read_random_shm.csv
- output PNG: writeback_read_random_shm.png
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
        return _script_dir() / "writeback_read_random_shm.csv"
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
                    "n_node": int(float(row["n_node"])),
                    "n_dim": int(float(row["n_dim"])),
                    "n_read": int(float(row["n_read"])),
                    "total_mb": float(row["total_mb"]),
                    "write_avg_ms": float(row["write_avg_ms"]),
                    "read_avg_ms": float(row["read_avg_ms"]),
                    "total_avg_ms": float(row["total_avg_ms"]),
                    "total_p50_ms": float(row["total_p50_ms"]),
                    "total_p95_ms": float(row["total_p95_ms"]),
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

    n_node = [row["n_node"] for row in rows]
    total_mb = [row["total_mb"] for row in rows]
    write_avg_ms = [row["write_avg_ms"] for row in rows]
    read_avg_ms = [row["read_avg_ms"] for row in rows]
    total_avg_ms = [row["total_avg_ms"] for row in rows]
    total_p50_ms = [row["total_p50_ms"] for row in rows]
    total_p95_ms = [row["total_p95_ms"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(n_node, total_avg_ms, marker="o", label="total_avg_ms")
    axes[0].plot(n_node, total_p50_ms, marker="s", label="total_p50_ms")
    axes[0].plot(n_node, total_p95_ms, marker="^", label="total_p95_ms")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Writeback + random shm read latency vs n_node")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(n_node, write_avg_ms, marker="o", label="write_avg_ms")
    axes[1].plot(n_node, read_avg_ms, marker="s", label="read_avg_ms")
    axes[1].plot(n_node, total_mb, marker="^", label="total_mb")
    axes[1].set_xlabel("n_node")
    axes[1].set_ylabel("ms / MB")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plot_path), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot writeback+random-read shm benchmark results from CSV"
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
