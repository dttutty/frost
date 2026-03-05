#!/usr/bin/env python3
import argparse
import logging
from collections import Counter

import numpy as np


logger = logging.getLogger(__name__)


def load_edges(npz_path: str) -> Counter:
    """
    Load indptr and indices from an .npz CSR file and
    return a Counter of directed edges (u, v).
    """
    data = np.load(npz_path)
    indptr = data["indptr"]
    indices = data["indices"]
    edges = Counter()
    num_nodes = len(indptr) - 1

    for u in range(num_nodes):
        start, end = indptr[u], indptr[u + 1]
        for v in indices[start:end]:
            edges[(u, int(v))] += 1
    return edges


def check_symmetry(edges: Counter) -> bool:
    """
    Return True if for every (u, v), count == count of (v, u).
    """
    return all(edges.get((v, u), 0) == cnt for (u, v), cnt in edges.items())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check if a CSR .npz was built with add_reverse=True"
    )
    p.add_argument(
        "npz_file", help="Path to the .npz file (must contain 'indptr' and 'indices')"
    )
    args = p.parse_args()

    logger.debug(f"Loading edges from {args.npz_file}...")
    edges = load_edges(args.npz_file)
    logger.debug(f"Total directed edges loaded: {sum(edges.values())}")

    logger.debug("Checking symmetry of the edge set...")
    if check_symmetry(edges):
        logger.debug(
            "✔ Every edge has a matching reverse: likely generated with add_reverse=True"
        )
    else:
        logger.debug(
            "✘ Not every edge is paired: likely generated with add_reverse=False"
        )


if __name__ == "__main__":
    main()
