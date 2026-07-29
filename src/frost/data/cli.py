"""Command-line interface for FROST dataset preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from frost.data.graph_policy import graph_filename
from frost.data.preparation import (
    convert_tensor_to_numpy,
    generate_node_roles,
    validate_dataset,
    write_temporal_csr,
)


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

    roles = subparsers.add_parser(
        "node-roles", help="build node_role.npy for a bipartite edge table"
    )
    roles.add_argument("edges", type=Path)
    roles.add_argument("--output", type=Path)
    roles.add_argument("--force", action="store_true")

    features = subparsers.add_parser(
        "convert-features", help="convert a plain PyTorch tensor to .npy"
    )
    features.add_argument("source", type=Path)
    features.add_argument("--output", type=Path)
    features.add_argument("--force", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="validate one published dataset directory"
    )
    validate.add_argument("dataset_dir", type=Path)
    _add_graph_direction_args(validate, required=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
    elif args.command == "node-roles":
        output = args.output or args.edges.parent / "node_role.npy"
        result = generate_node_roles(args.edges, output, force=args.force)
    elif args.command == "convert-features":
        output = args.output or args.source.with_suffix(".npy")
        result = convert_tensor_to_numpy(args.source, output, force=args.force)
    else:
        result = validate_dataset(
            args.dataset_dir, graph_direction=args.graph_direction
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
