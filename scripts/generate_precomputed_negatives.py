from __future__ import annotations

import argparse
from pathlib import Path

from frost.neg_sampler import (
    PRECOMPUTED_NEGATIVE_FILENAME,
    generate_precomputed_negative_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one precomputed negative destination per edge row."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("DATA"),
        help="Root directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used for deterministic negative generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges_paths = sorted(args.data_root.glob("*/edges.csv"))
    if not edges_paths:
        raise SystemExit(f"No edges.csv files found under {args.data_root}")

    for edges_path in edges_paths:
        output_path = edges_path.with_name(PRECOMPUTED_NEGATIVE_FILENAME)
        generate_precomputed_negative_file(
            edges_path=edges_path,
            output_path=output_path,
            seed=args.seed,
        )
        print(output_path)


if __name__ == "__main__":
    main()
