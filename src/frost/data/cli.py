"""Command-line interface for FROST dataset preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from frost.data.catalog import load_dataset_catalog, write_dataset_catalog
from frost.data.graph_policy import graph_filename
from frost.data.dgb import import_dgb_snapshot
from frost.data.preparation import (
    convert_tensor_to_numpy,
    validate_dataset,
    write_dataset_manifest,
    write_temporal_csr,
    write_temporal_splits,
)
from frost.data.tgb_download import (
    fetch_tgb_link_datasets,
    select_tgb_link_datasets,
    tgb_link_dataset_catalog,
)
from frost.neg_sampler import generate_precomputed_negative_file


def _add_graph_direction_args(
    parser: argparse.ArgumentParser, *, required: bool
) -> None:
    direction = parser.add_mutually_exclusive_group(required=required)
    direction.add_argument(
        "--directed",
        dest="graph_direction",
        action="store_const",
        const="directed",
        help="keep each input edge in its original direction",
    )
    direction.add_argument(
        "--bidirectional",
        dest="graph_direction",
        action="store_const",
        const="bidirectional",
        help="add a reverse adjacency for every input edge",
    )
    parser.set_defaults(graph_direction="auto")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frost-data",
        description="Build and validate dataset artifacts consumed by FROST.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    graph = subparsers.add_parser(
        "build-graph", help="build a temporal CSR .npz from edges.csv"
    )
    graph.add_argument("edges", type=Path)
    graph.add_argument("--output", type=Path)
    _add_graph_direction_args(graph, required=True)
    graph.add_argument("--force", action="store_true")

    features = subparsers.add_parser(
        "convert-features", help="convert a plain PyTorch tensor to .npy"
    )
    features.add_argument("source", type=Path)
    features.add_argument("--output", type=Path)
    features.add_argument("--force", action="store_true")

    split = subparsers.add_parser(
        "split", help="regenerate default_split with an explicit temporal policy"
    )
    split.add_argument("edges", type=Path)
    split.add_argument("--output", type=Path)
    split.add_argument(
        "--strategy", choices=["timestamp", "event_order"], default="timestamp"
    )
    split.add_argument("--train-ratio", type=float, default=0.70)
    split.add_argument("--val-ratio", type=float, default=0.15)
    split.add_argument("--force", action="store_true")

    negatives = subparsers.add_parser(
        "negatives",
        help="generate eid-keyed fixed evaluation negatives",
    )
    negatives.add_argument("edges", type=Path)
    negatives.add_argument("--output", type=Path)
    negatives.add_argument("--seed", type=int, default=0)
    negatives.add_argument(
        "--candidate-mode",
        choices=["all", "dst_partition"],
        default="dst_partition",
    )
    negatives.add_argument("--force", action="store_true")

    dgb = subparsers.add_parser(
        "import-dgb", help="import one extracted DGB Zenodo ml_* directory"
    )
    dgb.add_argument("source_dir", type=Path)
    dgb.add_argument("output_dir", type=Path)
    dgb.add_argument("--dataset", required=True)
    _add_graph_direction_args(dgb, required=True)
    dgb.add_argument(
        "--split-strategy",
        choices=["timestamp", "event_order"],
        default="timestamp",
    )
    dgb.add_argument("--train-ratio", type=float, default=0.70)
    dgb.add_argument("--val-ratio", type=float, default=0.15)
    dgb.add_argument("--negative-seed", type=int, default=0)
    dgb.add_argument("--force", action="store_true")

    tgb = subparsers.add_parser(
        "fetch-tgb",
        help="download official TGB dynamic link prediction datasets",
    )
    tgb.add_argument(
        "datasets",
        nargs="*",
        help="dataset names such as tgbl-wiki; use --all for the full catalog",
    )
    tgb.add_argument("--all", action="store_true", dest="all_datasets")
    tgb.add_argument(
        "--list",
        action="store_true",
        help="print the pinned catalog without downloading",
    )
    tgb.add_argument(
        "--root",
        type=Path,
        help="override TGB_DATA_ROOT for this command",
    )
    tgb.add_argument(
        "--download-only",
        action="store_true",
        help="verify and retain archives without extracting them",
    )

    manifest = subparsers.add_parser(
        "manifest", help="write provenance and protocol metadata for a dataset"
    )
    manifest.add_argument("dataset_dir", type=Path)
    manifest.add_argument("--upstream-project", required=True)
    manifest.add_argument("--upstream-dataset", required=True)
    manifest.add_argument("--source-url", required=True)
    manifest.add_argument(
        "--upstream-license", default="not_specified_by_upstream_record"
    )
    manifest.add_argument("--event-direction", required=True)
    manifest.add_argument(
        "--split-strategy", choices=["timestamp", "event_order"], required=True
    )
    manifest.add_argument("--negative-seed", type=int, default=0)
    manifest.add_argument("--force", action="store_true")

    catalog = subparsers.add_parser(
        "catalog", help="summarize a dataset bundle from its manifests"
    )
    catalog.add_argument("root", type=Path)
    catalog.add_argument(
        "--markdown-output",
        type=Path,
        help="write a generated Markdown table instead of returning JSON only",
    )
    catalog.add_argument("--force", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="validate one published dataset directory"
    )
    validate.add_argument("dataset_dir", type=Path)
    _add_graph_direction_args(validate, required=False)
    validate.add_argument("--allow-tied-split-timestamps", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result: object
    if args.command == "build-graph":
        add_reverse = args.graph_direction == "bidirectional"
        default_name = graph_filename(args.graph_direction)
        output = args.output or args.edges.parent / default_name
        result = write_temporal_csr(
            args.edges,
            output,
            add_reverse=add_reverse,
            force=args.force,
        )
    elif args.command == "convert-features":
        output = args.output or args.source.with_suffix(".npy")
        result = convert_tensor_to_numpy(args.source, output, force=args.force)
    elif args.command == "split":
        output = args.output or args.edges
        result = write_temporal_splits(
            args.edges,
            output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            strategy=args.strategy,
            force=args.force,
        )
    elif args.command == "negatives":
        output = args.output or args.edges.with_name("neg_pre_sampling_1_result.csv")
        path = generate_precomputed_negative_file(
            args.edges,
            output_path=output,
            seed=args.seed,
            force=args.force,
            candidate_mode=args.candidate_mode,
        )
        result = {
            "output": str(path.resolve()),
            "seed": args.seed,
            "candidate_mode": args.candidate_mode,
        }
    elif args.command == "import-dgb":
        result = import_dgb_snapshot(
            args.source_dir,
            args.output_dir,
            dataset=args.dataset,
            graph_direction=args.graph_direction,
            split_strategy=args.split_strategy,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            negative_seed=args.negative_seed,
            force=args.force,
        )
    elif args.command == "fetch-tgb":
        if args.list:
            if args.datasets or args.all_datasets:
                parser.error(
                    "fetch-tgb --list cannot be combined with dataset selectors"
                )
            result = {"datasets": tgb_link_dataset_catalog()}
        else:
            try:
                specs = select_tgb_link_datasets(
                    args.datasets, all_datasets=args.all_datasets
                )
            except ValueError as exc:
                parser.error(str(exc))
            print(
                "Selected TGB datasets: " + ", ".join(spec.name for spec in specs),
                file=sys.stderr,
            )
            result = {
                "datasets": fetch_tgb_link_datasets(
                    specs,
                    root=args.root,
                    download_only=args.download_only,
                )
            }
    elif args.command == "manifest":
        result = write_dataset_manifest(
            args.dataset_dir,
            upstream_project=args.upstream_project,
            upstream_dataset=args.upstream_dataset,
            source_url=args.source_url,
            upstream_license=args.upstream_license,
            event_direction=args.event_direction,
            split_strategy=args.split_strategy,
            negative_seed=args.negative_seed,
            force=args.force,
        )
    elif args.command == "catalog":
        if args.markdown_output is None:
            result = {"datasets": load_dataset_catalog(args.root)}
        else:
            result = write_dataset_catalog(
                args.root, args.markdown_output, force=args.force
            )
    else:
        result = validate_dataset(
            args.dataset_dir,
            graph_direction=args.graph_direction,
            allow_tied_split_timestamps=args.allow_tied_split_timestamps,
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
