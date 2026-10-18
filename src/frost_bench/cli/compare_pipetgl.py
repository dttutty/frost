from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from frost_bench.adapters import FrostAdapter, PipeTGLAdapter
from frost_bench.config import load_dataset_default_macro_batch_size
from frost_bench.io import (
    ensure_dir,
    read_text,
    run_prepared,
    utc_timestamp,
    write_rows_tsv,
)
from frost_bench.metrics import (
    mean_events_per_sec,
    parse_frost_epoch_summaries,
    parse_frost_epoch_wall_summaries,
    parse_frost_rank_throughput,
    parse_pipetgl_epoch_summaries,
    parse_pipetgl_rank_throughput,
    parse_pipetgl_run_summary,
    select_bottleneck_rank,
)
from frost_bench.paths import artifact_root
from frost_bench.spec import ExperimentSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a matched FROST vs PipeTGL TGN speed/convergence comparison."
    )
    parser.add_argument("--dataset", default="WIKIPEDIA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-gpus", type=int, default=2)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="global batch size; defaults from --dataset",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument(
        "--measure-start-epoch",
        type=int,
        default=0,
        help="zero-based epoch at which throughput measurement begins",
    )
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--mkl-threads", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29621)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _plot_convergence(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    metric_key = "val_mrr" if any("val_mrr" in row for row in rows) else "val_ap"
    for system in ("frost", "pipetgl"):
        system_rows = [
            row for row in rows if row["system"] == system and metric_key in row
        ]
        if not system_rows:
            continue
        epochs = [int(row["epoch_ordinal"]) for row in system_rows]
        axes[0].plot(
            epochs, [float(row[metric_key]) for row in system_rows], label=system
        )
        axes[1].plot(
            epochs,
            [float(row["train_loss"]) for row in system_rows],
            label=system,
        )
    axes[0].set(
        xlabel="Epoch", ylabel=f"Validation {metric_key[4:].upper()}", title="Convergence"
    )
    axes[1].set(xlabel="Epoch", ylabel="Train loss", title="Optimization")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def resolve_macro_batch_size(dataset: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return load_dataset_default_macro_batch_size(dataset, "tgn")


def attach_frost_epoch_wall_times(
    rows: list[dict[str, float | int | str]], wall_by_epoch: dict[int, float]
) -> None:
    """Attach zero-based runtime timings to one-based printed epoch rows."""
    for row in rows:
        runtime_epoch = int(row["epoch"]) - 1
        if runtime_epoch in wall_by_epoch:
            row["epoch_wall_time_s"] = wall_by_epoch[runtime_epoch]


def main() -> int:
    args = build_parser().parse_args()
    batch_size = resolve_macro_batch_size(args.dataset, args.batch_size)
    output = args.output or (
        artifact_root() / "comparisons" / "frost_vs_pipetgl" / utc_timestamp()
    )

    frost_extra = [
        "model=tgn",
        "lr=0.0001",
        "d_time=100",
        "dropout=0.2",
        "model.embedding_dim=100",
        "model.neighborhood.fanout=[10]",
    ]
    if args.dataset.upper() == "TGBL_WIKI":
        frost_extra.extend(
            [
                "neg_sampler=rnd_dst_then_tgb_wiki",
                "eval_metric=mrr",
                f"eval_macro_batch_size={args.n_gpus}",
            ]
        )
    else:
        frost_extra.append("neg_sampler=rnd_dst_then_fixed")
    frost_extra.extend(
        [
            "neg_sampler.seed=0",
            "neg_sampler.train.candidate_mode=dst_partition",
            f"patience={args.epochs}",
        ]
    )
    frost_spec = ExperimentSpec(
        model="tgn",
        dataset=args.dataset,
        seed=args.seed,
        n_gpus=args.n_gpus,
        batch_size=batch_size,
        epochs=args.epochs,
        extra_args=tuple(frost_extra),
    )
    pipetgl_spec = ExperimentSpec(
        model="tgn",
        dataset=args.dataset,
        seed=args.seed,
        n_gpus=args.n_gpus,
        batch_size=batch_size,
        epochs=args.epochs,
    )
    runs = [
        FrostAdapter().prepare(
            frost_spec,
            master_port=args.master_port,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            measure_start_epoch=args.measure_start_epoch,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
            gpu_ids=args.gpu_ids,
            epoch_wall=True,
        ),
        PipeTGLAdapter().prepare(
            pipetgl_spec,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            measure_start_epoch=args.measure_start_epoch,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
            gpu_ids=args.gpu_ids,
        ),
    ]
    if args.dry_run:
        for run in runs:
            print(f"{run.system}: {shlex.join(run.command)}")
        return 0

    ensure_dir(output)
    for run in runs:
        run_prepared(run, output / f"{run.system}.log")

    frost_log = read_text(output / "frost.log")
    pipetgl_log = read_text(output / "pipetgl.log")
    throughput_rows: list[dict[str, object]] = []
    for system, by_rank in (
        ("frost", parse_frost_rank_throughput(frost_log)),
        ("pipetgl", parse_pipetgl_rank_throughput(pipetgl_log)),
    ):
        rank, bottleneck, measured_batches = select_bottleneck_rank(by_rank)
        throughput_rows.append(
            {
                "system": system,
                "bottleneck_rank": rank,
                "bottleneck_events_per_sec_per_gpu": bottleneck,
                "mean_events_per_sec_per_gpu": mean_events_per_sec(by_rank),
                "measured_batches": measured_batches,
            }
        )
    write_rows_tsv(output / "throughput.tsv", throughput_rows)

    convergence_rows: list[dict[str, float | int | str]] = []
    frost_rows = parse_frost_epoch_summaries(frost_log)
    attach_frost_epoch_wall_times(
        frost_rows, parse_frost_epoch_wall_summaries(frost_log)
    )
    for rows in (frost_rows, parse_pipetgl_epoch_summaries(pipetgl_log)):
        for ordinal, row in enumerate(rows, start=1):
            convergence_rows.append({**row, "epoch_ordinal": ordinal})
    write_rows_tsv(output / "convergence.tsv", convergence_rows)
    _plot_convergence(convergence_rows, output / "convergence.png")

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_gpus": args.n_gpus,
        "global_batch_size": batch_size,
        "epochs": args.epochs,
        "measure_start_epoch": args.measure_start_epoch,
        "pipetgl": parse_pipetgl_run_summary(pipetgl_log),
        "throughput": throughput_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
