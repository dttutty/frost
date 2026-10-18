#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from frost_bench.io import ensure_dir

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    Line2D = None  # type: ignore[assignment]


MODEL_ORDER = ["dygformer", "graph_mixer", "tgat", "tcl", "cawn"]
DATASET_ORDER = ["LASTFM", "WIKIPEDIA", "REDDIT", "MOOC"]
BATCH_ORDER = [600, 1200, 2000, 4000]

MODEL_LABELS = {
    "dygformer": "DyGFormer",
    "graph_mixer": "GraphMixer",
    "tgat": "TGAT",
    "tcl": "TCL",
    "cawn": "CAWN",
}
MODEL_COLORS = {
    "dygformer": "#1b9e77",
    "graph_mixer": "#d95f02",
    "tgat": "#7570b3",
    "tcl": "#e7298a",
    "cawn": "#66a61e",
}
DATASET_MARKERS = {
    "LASTFM": "o",
    "WIKIPEDIA": "s",
    "REDDIT": "^",
    "MOOC": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot FROST vs DyGLib throughput sweep results."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_float(text: str) -> float:
    if text is None:
        return float("nan")
    stripped = text.strip()
    if not stripped:
        return float("nan")
    try:
        return float(stripped)
    except ValueError:
        return float("nan")


def parse_int(text: str) -> int | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def read_results(results_tsv: Path) -> list[dict[str, object]]:
    if not results_tsv.exists():
        return []
    with results_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: list[dict[str, object]] = []
        for row in reader:
            model = (row.get("model") or "").strip()
            dataset = (row.get("dataset") or "").strip().upper()
            macro_batch_size = parse_int(row.get("macro_batch_size", ""))
            if not model or not dataset or macro_batch_size is None:
                continue
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seed": (row.get("seed") or "").strip(),
                    "n_gpu": parse_int(row.get("n_gpu", "")),
                    "macro_batch_size": macro_batch_size,
                    "per_gpu_batch_size": parse_int(row.get("per_gpu_batch_size", "")),
                    "frost_status": (row.get("frost_status") or "").strip(),
                    "dyglib_status": (row.get("dyglib_status") or "").strip(),
                    "frost_events_per_sec_per_gpu": parse_float(
                        row.get("frost_events_per_sec_per_gpu", "")
                    ),
                    "dyglib_events_per_sec_per_gpu": parse_float(
                        row.get("dyglib_events_per_sec_per_gpu", "")
                    ),
                    "speedup_frost_over_dyglib": parse_float(
                        row.get("speedup_frost_over_dyglib", "")
                    ),
                    "frost_measured_batches": parse_int(
                        row.get("frost_measured_batches", "")
                    ),
                    "dyglib_measured_batches": parse_int(
                        row.get("dyglib_measured_batches", "")
                    ),
                    "frost_log": (row.get("frost_log") or "").strip(),
                    "dyglib_log": (row.get("dyglib_log") or "").strip(),
                }
            )
    return rows


def is_success(row: dict[str, object]) -> bool:
    return row["frost_status"] == "ok" and row["dyglib_status"] == "ok"


def geometric_mean(values: list[float]) -> float:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def sort_key(summary_row: dict[str, object]) -> tuple[int, int, int]:
    model = str(summary_row["model"])
    dataset = str(summary_row["dataset"])
    batch = int(summary_row["macro_batch_size"])
    return (
        MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
        DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER),
        batch,
    )


def build_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if is_success(row):
            grouped[
                (
                    str(row["model"]),
                    str(row["dataset"]),
                    int(row["macro_batch_size"]),
                )
            ].append(row)

    summary_rows: list[dict[str, object]] = []
    for (model, dataset, macro_batch_size), items in grouped.items():
        frost_values = [float(item["frost_events_per_sec_per_gpu"]) for item in items]
        dyglib_values = [float(item["dyglib_events_per_sec_per_gpu"]) for item in items]
        speedup_values = [float(item["speedup_frost_over_dyglib"]) for item in items]
        frost_mean = statistics.mean(frost_values)
        dyglib_mean = statistics.mean(dyglib_values)
        summary_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "macro_batch_size": macro_batch_size,
                "runs": len(items),
                "frost_events_per_sec_per_gpu_mean": frost_mean,
                "dyglib_events_per_sec_per_gpu_mean": dyglib_mean,
                "speedup_frost_over_dyglib_mean": (
                    frost_mean / dyglib_mean if dyglib_mean > 0 else float("nan")
                ),
                "speedup_frost_over_dyglib_gmean": geometric_mean(speedup_values),
            }
        )
    return sorted(summary_rows, key=sort_key)


def write_summary_tsv(path: Path, summary_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "dataset",
        "macro_batch_size",
        "runs",
        "frost_events_per_sec_per_gpu_mean",
        "dyglib_events_per_sec_per_gpu_mean",
        "speedup_frost_over_dyglib_mean",
        "speedup_frost_over_dyglib_gmean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "macro_batch_size": row["macro_batch_size"],
                    "runs": row["runs"],
                    "frost_events_per_sec_per_gpu_mean": (
                        f"{float(row['frost_events_per_sec_per_gpu_mean']):.6f}"
                    ),
                    "dyglib_events_per_sec_per_gpu_mean": (
                        f"{float(row['dyglib_events_per_sec_per_gpu_mean']):.6f}"
                    ),
                    "speedup_frost_over_dyglib_mean": (
                        f"{float(row['speedup_frost_over_dyglib_mean']):.6f}"
                    ),
                    "speedup_frost_over_dyglib_gmean": (
                        f"{float(row['speedup_frost_over_dyglib_gmean']):.6f}"
                    ),
                }
            )


def write_failures_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    failures = [row for row in rows if not is_success(row)]
    fieldnames = [
        "model",
        "dataset",
        "seed",
        "macro_batch_size",
        "frost_status",
        "dyglib_status",
        "frost_events_per_sec_per_gpu",
        "dyglib_events_per_sec_per_gpu",
        "frost_measured_batches",
        "dyglib_measured_batches",
        "frost_log",
        "dyglib_log",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in failures:
            writer.writerow(
                {
                    name: row.get(name, "")
                    for name in fieldnames
                }
            )


def write_summary_txt(path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "model\tdataset\tmacro_batch_size\truns\tfrost_eps_mean\tdyglib_eps_mean\tspeedup_mean\tspeedup_gmean"
    ]
    for row in summary_rows:
        lines.append(
            "\t".join(
                [
                    str(row["model"]),
                    str(row["dataset"]),
                    str(row["macro_batch_size"]),
                    str(row["runs"]),
                    f"{float(row['frost_events_per_sec_per_gpu_mean']):.6f}",
                    f"{float(row['dyglib_events_per_sec_per_gpu_mean']):.6f}",
                    f"{float(row['speedup_frost_over_dyglib_mean']):.6f}",
                    f"{float(row['speedup_frost_over_dyglib_gmean']):.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_speedup_grid(run_dir: Path, summary_rows: list[dict[str, object]]) -> list[Path]:
    if plt is None:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), sharex=True, sharey=True)
    by_key = {
        (str(row["model"]), str(row["dataset"]), int(row["macro_batch_size"])): row
        for row in summary_rows
    }
    for ax, dataset in zip(axes.flat, DATASET_ORDER):
        plotted = False
        for model in MODEL_ORDER:
            xs: list[int] = []
            ys: list[float] = []
            for batch_size in BATCH_ORDER:
                row = by_key.get((model, dataset, batch_size))
                if row is None:
                    continue
                xs.append(batch_size)
                ys.append(float(row["speedup_frost_over_dyglib_mean"]))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2.0,
                    markersize=6.0,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )
                plotted = True
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(dataset)
        ax.set_xticks(BATCH_ORDER)
        ax.grid(alpha=0.22)
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No completed runs yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#555555",
            )

    axes[0, 0].set_ylabel("Speedup (FROST / DyGLib)")
    axes[1, 0].set_ylabel("Speedup (FROST / DyGLib)")
    axes[1, 0].set_xlabel("Macro batch size")
    axes[1, 1].set_xlabel("Macro batch size")

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=2.0,
            markersize=6.0,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("FROST vs DyGLib Throughput Speedup by Dataset", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = run_dir / "speedup_by_dataset.png"
    pdf_path = run_dir / "speedup_by_dataset.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_throughput_scatter(run_dir: Path, summary_rows: list[dict[str, object]]) -> list[Path]:
    if plt is None or not summary_rows:
        return []

    fig, ax = plt.subplots(figsize=(8.4, 6.7))
    max_value = 0.0
    sorted_rows = sorted(
        summary_rows,
        key=lambda row: float(row["speedup_frost_over_dyglib_mean"]),
        reverse=True,
    )
    for idx, row in enumerate(sorted_rows):
        model = str(row["model"])
        dataset = str(row["dataset"])
        frost_value = float(row["frost_events_per_sec_per_gpu_mean"])
        dyglib_value = float(row["dyglib_events_per_sec_per_gpu_mean"])
        max_value = max(max_value, frost_value, dyglib_value)
        ax.scatter(
            dyglib_value,
            frost_value,
            s=85,
            marker=DATASET_MARKERS[dataset],
            color=MODEL_COLORS[model],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
        )
        if idx < 6:
            ax.annotate(
                f"{MODEL_LABELS[model]}/{dataset}/{int(row['macro_batch_size'])}",
                (dyglib_value, frost_value),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8.5,
            )

    diagonal_max = max_value * 1.08 if max_value > 0 else 1.0
    ax.plot([0, diagonal_max], [0, diagonal_max], linestyle="--", color="#888888", linewidth=1.2)
    ax.set_xlim(0, diagonal_max)
    ax.set_ylim(0, diagonal_max)
    ax.set_xlabel("DyGLib throughput (events/sec/gpu)")
    ax.set_ylabel("FROST throughput (events/sec/gpu)")
    ax.set_title("Per-Combo Throughput: FROST vs DyGLib")
    ax.grid(alpha=0.22)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor=MODEL_COLORS[model],
            markersize=7,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=DATASET_MARKERS[dataset],
            color="#333333",
            linestyle="none",
            markersize=7,
            label=dataset,
        )
        for dataset in DATASET_ORDER
    ]
    legend_models = ax.legend(handles=model_handles, title="Model", loc="upper left", frameon=False)
    ax.add_artist(legend_models)
    ax.legend(handles=dataset_handles, title="Dataset", loc="lower right", frameon=False)

    fig.tight_layout()
    png_path = run_dir / "throughput_scatter.png"
    pdf_path = run_dir / "throughput_scatter.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def build_report(
    run_dir: Path,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    expected_total: int,
) -> str:
    completed = len(rows)
    successful = sum(1 for row in rows if is_success(row))
    failed = completed - successful
    remaining = max(expected_total - completed, 0) if expected_total > 0 else 0
    completion_pct = (completed / expected_total * 100.0) if expected_total > 0 else float("nan")

    lines = [
        f"# Throughput Sweep Report ({utc_now()})",
        "",
        "## Status",
        f"- Run dir: `{run_dir}`",
        f"- Completed combos: `{completed}`"
        + (f" / `{expected_total}` (`{completion_pct:.1f}%`)" if expected_total > 0 else ""),
        f"- Successful combos: `{successful}`",
        f"- Failed combos: `{failed}`",
    ]
    if expected_total > 0:
        lines.append(f"- Remaining combos: `{remaining}`")
    lines.extend(
        [
            "",
            "## Aggregate Speedup",
        ]
    )

    if summary_rows:
        speedups = [float(row["speedup_frost_over_dyglib_mean"]) for row in summary_rows]
        lines.extend(
            [
                f"- Mean speedup across completed cells: `{statistics.mean(speedups):.3f}x`",
                f"- Median speedup across completed cells: `{statistics.median(speedups):.3f}x`",
                f"- Geometric mean speedup across completed cells: `{geometric_mean(speedups):.3f}x`",
            ]
        )

        by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
        by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in summary_rows:
            by_model[str(row["model"])].append(row)
            by_dataset[str(row["dataset"])].append(row)

        lines.extend(["", "## Speedup by Model"])
        for model in MODEL_ORDER:
            model_rows = by_model.get(model, [])
            if not model_rows:
                continue
            model_speedups = [
                float(row["speedup_frost_over_dyglib_mean"]) for row in model_rows
            ]
            lines.append(
                f"- {MODEL_LABELS[model]}: `{geometric_mean(model_speedups):.3f}x` gmean over `{len(model_rows)}` completed cells"
            )

        lines.extend(["", "## Speedup by Dataset"])
        for dataset in DATASET_ORDER:
            dataset_rows = by_dataset.get(dataset, [])
            if not dataset_rows:
                continue
            dataset_speedups = [
                float(row["speedup_frost_over_dyglib_mean"]) for row in dataset_rows
            ]
            lines.append(
                f"- {dataset}: `{geometric_mean(dataset_speedups):.3f}x` gmean over `{len(dataset_rows)}` completed cells"
            )

        best_rows = sorted(
            summary_rows,
            key=lambda row: float(row["speedup_frost_over_dyglib_mean"]),
            reverse=True,
        )[:10]
        lines.extend(["", "## Top Completed Cells"])
        for row in best_rows:
            lines.append(
                "- "
                + f"{MODEL_LABELS[str(row['model'])]} / {row['dataset']} / mb{row['macro_batch_size']}: "
                + f"`{float(row['speedup_frost_over_dyglib_mean']):.3f}x` "
                + f"(FROST `{float(row['frost_events_per_sec_per_gpu_mean']):.1f}`, "
                + f"DyGLib `{float(row['dyglib_events_per_sec_per_gpu_mean']):.1f}`)"
            )
    else:
        lines.append("- No successful cells have completed yet.")

    failures = [row for row in rows if not is_success(row)]
    lines.extend(["", "## Failures"])
    if failures:
        for row in failures:
            lines.append(
                "- "
                + f"{MODEL_LABELS.get(str(row['model']), str(row['model']))} / {row['dataset']} / mb{row['macro_batch_size']} / seed{row['seed']}: "
                + f"FROST=`{row['frost_status']}`, DyGLib=`{row['dyglib_status']}`"
            )
    else:
        lines.append("- No failed cells recorded so far.")

    latest_rows = rows[-10:]
    lines.extend(["", "## Latest Completed Combos"])
    if latest_rows:
        for row in latest_rows:
            speedup = float(row["speedup_frost_over_dyglib"])
            speedup_text = f"{speedup:.3f}x" if math.isfinite(speedup) else "nan"
            lines.append(
                "- "
                + f"{MODEL_LABELS.get(str(row['model']), str(row['model']))} / {row['dataset']} / mb{row['macro_batch_size']} / seed{row['seed']}: "
                + f"FROST=`{row['frost_status']}`, DyGLib=`{row['dyglib_status']}`, speedup=`{speedup_text}`"
            )
    else:
        lines.append("- No combo has finished yet.")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ensure_dir(run_dir)

    results_tsv = run_dir / "results.tsv"
    rows = read_results(results_tsv)
    summary_rows = build_summary_rows(rows)

    write_summary_tsv(run_dir / "summary.tsv", summary_rows)
    write_summary_txt(run_dir / "summary.txt", summary_rows)
    write_failures_tsv(run_dir / "failures.tsv", rows)
    report_text = build_report(run_dir, rows, summary_rows, args.expected_total)
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")

    plot_speedup_grid(run_dir, summary_rows)
    plot_throughput_scatter(run_dir, summary_rows)

    print(f"Updated artifacts under {run_dir}")


if __name__ == "__main__":
    main()
