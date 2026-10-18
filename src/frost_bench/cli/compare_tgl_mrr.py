from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np

from frost.data.paths import resolve_tgb_raw_dataset_dir
from frost.data.tgb import load_tgbl_wiki_edge_features, load_tgbl_wiki_runtime_graph
from frost_bench.adapters import FrostAdapter, TGLAdapter
from frost_bench.config import load_dataset_default_macro_batch_size
from frost_bench.io import ensure_dir, read_text, run_prepared, utc_timestamp, write_rows_tsv
from frost_bench.metrics import (
    parse_frost_epoch_summaries,
    parse_frost_epoch_wall_summaries,
    parse_tgl_epoch_summaries,
)
from frost_bench.paths import TGL_ROOT, artifact_root
from frost_bench.spec import BatchSizeScope, ExperimentSpec, PreparedRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a matched FROST vs TGL TGN MRR convergence comparison."
    )
    parser.add_argument("--dataset", default="WIKIPEDIA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-gpus", type=int, default=2)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-neg-samples", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--mkl-threads", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29641)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_macro_batch_size(dataset: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return load_dataset_default_macro_batch_size(dataset, "tgn")


def resolve_eval_neg_samples(dataset: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return 999 if dataset.upper() == "TGBL_WIKI" else 20


def materialize_tgbl_wiki_for_tgl(output_root: Path) -> Path:
    """Materialize TGL's generic artifact view from official TGB files."""
    dataset_dir = output_root / "tgbl_wiki"
    edges_path = dataset_dir / "edges.csv"
    graph_path = dataset_dir / "full_graph_with_reverse_edges.npz"
    edge_features_path = dataset_dir / "edge_features.npy"
    if edges_path.is_file() and graph_path.is_file() and edge_features_path.is_file():
        return output_root

    raw_dir = resolve_tgb_raw_dataset_dir("TGBL_WIKI")
    runtime_graph = load_tgbl_wiki_runtime_graph(raw_dir)
    edge_features = load_tgbl_wiki_edge_features(raw_dir)
    if edge_features.shape[0] != len(runtime_graph.frame):
        raise ValueError(
            "official tgbl-wiki edge features do not match the edge table: "
            f"{edge_features.shape[0]} != {len(runtime_graph.frame)}"
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    runtime_graph.frame.to_csv(edges_path, index=False)
    np.savez(
        graph_path,
        indptr=runtime_graph.arrays["indptr"],
        indices=runtime_graph.arrays["indices"],
        eid=runtime_graph.arrays["eid"],
        ts=runtime_graph.arrays["ts"],
    )
    np.save(edge_features_path, edge_features.astype(np.float32, copy=False))
    return output_root


def _attach_frost_epoch_wall_times(
    rows: list[dict[str, float | int | str]], wall_by_epoch: dict[int, float]
) -> None:
    for ordinal, row in enumerate(rows):
        if ordinal in wall_by_epoch:
            row["epoch_wall_time_s"] = wall_by_epoch[ordinal]


def prepare_runs(
    args: argparse.Namespace, output: Path
) -> tuple[PreparedRun, PreparedRun]:
    eval_neg_samples = resolve_eval_neg_samples(args.dataset, args.eval_neg_samples)
    if eval_neg_samples <= 0:
        raise ValueError("--eval-neg-samples must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    batch_size = resolve_macro_batch_size(args.dataset, args.batch_size)
    if batch_size % args.n_gpus:
        raise ValueError("global batch size must be divisible by --n-gpus")

    official_tgb = args.dataset.upper() == "TGBL_WIKI"
    if official_tgb and eval_neg_samples != 999:
        raise ValueError(
            "official TGBL_WIKI evaluation requires capacity 999; omit "
            "--eval-neg-samples or set it to 999"
        )
    frost_negative_args = (
        (
            "neg_sampler=rnd_dst_then_tgb_wiki",
            f"eval_macro_batch_size={args.eval_batch_size}",
        )
        if official_tgb
        else (
            "neg_sampler=rnd_dst_then_random",
            f"neg_sampler.seed={args.seed}",
            f"neg_sampler.eval.n_neg={eval_neg_samples}",
            "reset_eval_neg_sampler_each_epoch=true",
        )
    )
    frost_spec = ExperimentSpec(
        model="tgn",
        dataset=args.dataset,
        seed=args.seed,
        n_gpus=args.n_gpus,
        batch_size=batch_size,
        epochs=args.epochs,
        extra_args=(
            "model=tgn",
            *frost_negative_args,
            "eval_metric=mrr",
            f"patience={args.epochs}",
        ),
    )
    tgl_config = output / "config" / "tgl_tgn.yml"
    TGLAdapter().write_config(
        source=TGL_ROOT / "config" / "frost_parity" / "TGN.yml",
        destination=tgl_config,
        epochs=args.epochs,
        per_gpu_batch_size=batch_size // args.n_gpus,
    )
    tgl_eval_args = (
        (
            "--eval-negative-dir",
            str(resolve_tgb_raw_dataset_dir(args.dataset)),
            "--eval-batch-size",
            str(args.eval_batch_size),
        )
        if official_tgb
        else ()
    )
    tgl_spec = ExperimentSpec(
        model="tgn",
        dataset=args.dataset,
        seed=args.seed,
        n_gpus=args.n_gpus,
        batch_size=batch_size // args.n_gpus,
        batch_size_scope=BatchSizeScope.PER_GPU,
        epochs=args.epochs,
        extra_args=(
            "--eval-metric",
            "mrr",
            "--eval-neg-samples",
            str(eval_neg_samples),
            *tgl_eval_args,
        ),
    )
    return (
        FrostAdapter().prepare(
            frost_spec,
            master_port=args.master_port,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
            gpu_ids=args.gpu_ids,
            epoch_wall=True,
        ),
        TGLAdapter(data_root=output / "tgl_data").prepare(
            tgl_spec,
            config_path=tgl_config,
            max_train_steps=0,
            warmup_batches=0,
            measure_batches=0,
            omp_threads=args.omp_threads,
            mkl_threads=args.mkl_threads,
            gpu_ids=args.gpu_ids,
            appendix_train_e2e=False,
        ),
    )


def _plot_mrr(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for system in ("frost", "tgl"):
        system_rows = [row for row in rows if row["system"] == system]
        epochs = [int(row["epoch_ordinal"]) for row in system_rows]
        axes[0].plot(
            epochs, [float(row["val_mrr"]) for row in system_rows], label=system
        )
        axes[1].plot(
            epochs, [float(row["test_mrr"]) for row in system_rows], label=system
        )
    axes[0].set(xlabel="Epoch", ylabel="Validation MRR", title="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Test MRR", title="Test")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _system_summary(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    best_row = max(rows, key=lambda row: float(row["val_mrr"]))
    return {
        "best_epoch": int(best_row["epoch_ordinal"]),
        "best_val_mrr": float(best_row["val_mrr"]),
        "corresponding_test_mrr": float(best_row["test_mrr"]),
        "final_val_mrr": float(rows[-1]["val_mrr"]),
        "final_test_mrr": float(rows[-1]["test_mrr"]),
    }


def main() -> int:
    args = build_parser().parse_args()
    output = args.output or (
        artifact_root() / "comparisons" / "frost_vs_tgl_mrr" / utc_timestamp()
    )
    ensure_dir(output)
    runs = prepare_runs(args, output)
    if args.dry_run:
        for run in runs:
            print(f"{run.system}: {shlex.join(run.command)}")
        return 0

    if args.dataset.upper() == "TGBL_WIKI":
        materialize_tgbl_wiki_for_tgl(output / "tgl_data")

    for run in runs:
        run_prepared(run, output / f"{run.system}.log")

    frost_rows = parse_frost_epoch_summaries(read_text(output / "frost.log"))
    _attach_frost_epoch_wall_times(
        frost_rows,
        parse_frost_epoch_wall_summaries(read_text(output / "frost.log")),
    )
    tgl_rows = parse_tgl_epoch_summaries(read_text(output / "tgl.log"))
    if len(frost_rows) != args.epochs or len(tgl_rows) != args.epochs:
        raise RuntimeError(
            f"incomplete curves: frost={len(frost_rows)}, tgl={len(tgl_rows)}, "
            f"expected={args.epochs}"
        )

    convergence_rows: list[dict[str, float | int | str]] = []
    for rows in (frost_rows, tgl_rows):
        for ordinal, row in enumerate(rows, start=1):
            convergence_rows.append({**row, "epoch_ordinal": ordinal})
    write_rows_tsv(output / "convergence.tsv", convergence_rows)
    _plot_mrr(convergence_rows, output / "convergence.png")

    normalized = {
        system: [row for row in convergence_rows if row["system"] == system]
        for system in ("frost", "tgl")
    }
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_gpus": args.n_gpus,
        "global_batch_size": resolve_macro_batch_size(
            args.dataset, args.batch_size
        ),
        "epochs": args.epochs,
        "evaluation_metric": "mrr",
        "eval_neg_samples": resolve_eval_neg_samples(
            args.dataset, args.eval_neg_samples
        ),
        "negative_candidate_mode": (
            "official_tgb" if args.dataset.upper() == "TGBL_WIKI" else "dst_partition"
        ),
        "systems": {
            system: _system_summary(rows) for system, rows in normalized.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
