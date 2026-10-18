from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

from frost_bench.adapters import FrostAdapter, PipeTGLAdapter
from frost_bench.config import (
    load_dataset_default_macro_batch_size,
    normalize_dataset_name,
)
from frost_bench.io import ensure_dir, read_text, run_prepared, write_rows_tsv
from frost_bench.metrics import (
    parse_frost_epoch_summaries,
    parse_pipetgl_epoch_summaries,
)
from frost_bench.spec import ExperimentSpec


SYSTEMS = ("frost", "pipetgl")
DEFAULT_NEGATIVE_FILE = "neg_pre_sampling_1_all_nodes_seed0.csv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and summarize a two-GPU TGN convergence comparison."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--system", choices=SYSTEMS, required=True)
    run.add_argument("--dataset", default="LASTFM")
    run.add_argument("--gpu-ids", required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    run.add_argument("--epochs", type=int, default=100)
    run.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="global batch size; defaults from --dataset",
    )
    run.add_argument("--master-port", type=int, default=29731)
    run.add_argument("--negative-file", default=DEFAULT_NEGATIVE_FILE)
    run.add_argument("--omp-threads", type=int, default=32)
    run.add_argument("--mkl-threads", type=int, default=32)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--dataset", default="LASTFM")
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    summarize.add_argument("--epochs", type=int, default=100)
    return parser


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _prepared_run(args: argparse.Namespace, seed: int):
    dataset = normalize_dataset_name(args.dataset)
    batch_size = args.batch_size or load_dataset_default_macro_batch_size(
        dataset, "tgn"
    )
    spec = ExperimentSpec(
        model="tgn",
        dataset=dataset,
        seed=seed,
        n_gpus=2,
        batch_size=batch_size,
        epochs=args.epochs,
        extra_args=(),
    )
    if args.system == "frost":
        spec = replace(
            spec,
            extra_args=(
                "model=tgn",
                "lr=0.0001",
                "d_time=100",
                "dropout=0.2",
                "model.embedding_dim=100",
                "model.neighborhood.fanout=[10]",
                "neg_sampler=rnd_dst_then_fixed_all",
                "neg_sampler.seed=0",
                f"neg_sampler.eval.neg_file={args.negative_file}",
                f"patience={args.epochs + 1}",
            ),
        )
        prepared = FrostAdapter().prepare(
            spec,
            master_port=args.master_port + seed,
            warmup_batches=0,
            measure_batches=0,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
            gpu_ids=args.gpu_ids,
            stable_mode=True,
            epoch_wall=True,
        )
    else:
        spec = replace(
            spec,
            extra_args=(
                "--evaluation-negative-file",
                args.negative_file,
            ),
        )
        prepared = PipeTGLAdapter(data_root=args.data_root).prepare(
            spec,
            warmup_batches=0,
            measure_batches=0,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
            gpu_ids=args.gpu_ids,
        )
    return prepared


def run_system(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    dataset = normalize_dataset_name(args.dataset)
    batch_size = args.batch_size or load_dataset_default_macro_batch_size(
        dataset, "tgn"
    )
    dataset_dir = data_root / dataset
    negative_path = dataset_dir / args.negative_file
    if not negative_path.is_file():
        raise FileNotFoundError(f"Missing all-node negatives: {negative_path}")
    os.environ["FROST_DATA_ROOT"] = str(data_root)
    os.environ["FROST_OPTIMIZED_SPEED"] = "0"
    ensure_dir(args.output)

    status_path = args.output / f"{args.system}_status.json"
    for ordinal, seed in enumerate(args.seeds, start=1):
        run_dir = args.output / args.system / f"seed_{seed}"
        ensure_dir(run_dir)
        prepared = _prepared_run(args, seed)
        metadata = {
            "system": args.system,
            "dataset": dataset,
            "seed": seed,
            "epochs": args.epochs,
            "global_batch_size": batch_size,
            "gpu_ids": args.gpu_ids,
            "data_root": str(data_root),
            "negative_file": args.negative_file,
            "negative_candidate_mode": "all",
            "frost_optimized_speed": "0",
            "frost_commit": _git_revision(),
            "command": list(prepared.command),
            "environment": dict(prepared.env),
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status_path.write_text(
            json.dumps(
                {
                    "state": "running",
                    "system": args.system,
                    "seed": seed,
                    "seed_ordinal": ordinal,
                    "seed_count": len(args.seeds),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run_prepared(prepared, run_dir / "train.log")

    status_path.write_text(
        json.dumps(
            {
                "state": "complete",
                "system": args.system,
                "seeds": args.seeds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _load_curve(output: Path, system: str, seed: int) -> list[dict]:
    text = read_text(output / system / f"seed_{seed}" / "train.log")
    if system == "frost":
        return parse_frost_epoch_summaries(text)
    return parse_pipetgl_epoch_summaries(text)


def _plot(
    curves: dict[tuple[str, int], list[dict]], output: Path, dataset: str
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    colors = {"frost": "#D55E00", "pipetgl": "#0072B2"}
    for system in SYSTEMS:
        system_curves = [
            rows for (name, _), rows in curves.items() if name == system
        ]
        epochs = [int(row["epoch"]) for row in system_curves[0]]
        values = [
            [float(row["test_ap"]) for row in rows] for rows in system_curves
        ]
        means = [statistics.fmean(column) for column in zip(*values)]
        stds = [
            statistics.stdev(column) if len(column) > 1 else 0.0
            for column in zip(*values)
        ]
        for curve in values:
            axis.plot(epochs, curve, color=colors[system], alpha=0.14, linewidth=0.8)
        axis.plot(epochs, means, color=colors[system], label=system, linewidth=2.0)
        axis.fill_between(
            epochs,
            [mean - std for mean, std in zip(means, stds)],
            [mean + std for mean, std in zip(means, stds)],
            color=colors[system],
            alpha=0.18,
        )
    axis.set(
        xlabel="Epoch",
        ylabel="Test AP",
        title=f"{dataset} test AP convergence",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "test_ap_convergence.png", dpi=200)
    figure.savefig(output / "test_ap_convergence.pdf")
    plt.close(figure)


def summarize(args: argparse.Namespace) -> int:
    dataset = normalize_dataset_name(args.dataset)
    curves: dict[tuple[str, int], list[dict]] = {}
    convergence_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for seed in args.seeds:
            rows = _load_curve(args.output, system, seed)
            if len(rows) != args.epochs:
                raise RuntimeError(
                    f"{system} seed {seed} produced {len(rows)} epochs, "
                    f"expected {args.epochs}"
                )
            curves[(system, seed)] = rows
            convergence_rows.extend({**row, "seed": seed} for row in rows)
            final = rows[-1]
            best = max(rows, key=lambda row: float(row["val_ap"]))
            final_rows.append(
                {
                    "system": system,
                    "seed": seed,
                    "final_epoch": int(final["epoch"]),
                    "final_test_ap": float(final["test_ap"]),
                    "best_val_epoch": int(best["epoch"]),
                    "best_val_ap": float(best["val_ap"]),
                    "best_val_test_ap": float(best["test_ap"]),
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        system_rows = [row for row in final_rows if row["system"] == system]
        for metric in ("final_test_ap", "best_val_test_ap"):
            values = [float(cast(float, row[metric])) for row in system_rows]
            aggregate_rows.append(
                {
                    "system": system,
                    "metric": metric,
                    "seeds": len(values),
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )

    write_rows_tsv(args.output / "convergence.tsv", convergence_rows)
    write_rows_tsv(args.output / "final_ap.tsv", final_rows)
    write_rows_tsv(args.output / "aggregate_ap.tsv", aggregate_rows)
    _plot(curves, args.output, dataset)
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        return run_system(args)
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
