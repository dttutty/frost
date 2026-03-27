from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "DATA"
ROLE_CONFIG = {
    "wikipedia": {
        "num_nodes": 9228,
        "user_range": (1, 8228),
        "item_range": (8228, 9228),
    },
    "reddit": {
        "num_nodes": 10985,
        "user_range": (1, 10001),
        "item_range": (10001, 10985),
    },
    "mooc": {
        "num_nodes": 7145,
        "user_range": (1, 7048),
        "item_range": (7048, 7145),
    },
    "lastfm": {
        "num_nodes": 1981,
        "user_range": (1, 981),
        "item_range": (981, 1981),
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def dataset_csv_path(dataset_name: str) -> Path:
    csv_path = DATA_DIR / dataset_name / "edges.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing edges.csv for {dataset_name}: {csv_path}")
    return csv_path


def validate_config(dataset_name: str, config: dict[str, tuple[int, int] | int]) -> None:
    csv_path = dataset_csv_path(dataset_name)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        umin = umax = imin = imax = None
        uset: set[int] = set()
        iset: set[int] = set()
        n_edges = 0
        for row in reader:
            u = int(float(row["src"]))
            i = int(float(row["dst"]))
            umin = u if umin is None else min(umin, u)
            umax = u if umax is None else max(umax, u)
            imin = i if imin is None else min(imin, i)
            imax = i if imax is None else max(imax, i)
            uset.add(u)
            iset.add(i)
            n_edges += 1

    expected_user_lo, expected_user_hi = config["user_range"]
    expected_item_lo, expected_item_hi = config["item_range"]
    expected_num_nodes = int(config["num_nodes"])
    inferred_num_nodes = max(int(umax), int(imax)) + 1

    if (umin, umax + 1) != (expected_user_lo, expected_user_hi):
        raise ValueError(
            f"{dataset_name}: user id range mismatch, got [{umin}->{umax + 1}), "
            f"expected [{expected_user_lo}->{expected_user_hi})"
        )
    if (imin, imax + 1) != (expected_item_lo, expected_item_hi):
        raise ValueError(
            f"{dataset_name}: item id range mismatch, got [{imin}->{imax + 1}), "
            f"expected [{expected_item_lo}->{expected_item_hi})"
        )
    if inferred_num_nodes != expected_num_nodes:
        raise ValueError(
            f"{dataset_name}: num_nodes mismatch, got {inferred_num_nodes}, expected {expected_num_nodes}"
        )
    if len(uset) != expected_user_hi - expected_user_lo:
        raise ValueError(f"{dataset_name}: unexpected user unique count {len(uset)}")
    if len(iset) != expected_item_hi - expected_item_lo:
        raise ValueError(f"{dataset_name}: unexpected item unique count {len(iset)}")

    log(
        f"validated {dataset_name}: edges={n_edges}, "
        f"user_range=[{expected_user_lo}->{expected_user_hi}), "
        f"item_range=[{expected_item_lo}->{expected_item_hi}), "
        f"num_nodes={expected_num_nodes}"
    )


def generate_node_role(dataset_name: str, config: dict[str, tuple[int, int] | int]) -> Path:
    validate_config(dataset_name, config)

    num_nodes = int(config["num_nodes"])
    item_lo, item_hi = config["item_range"]
    arr = np.zeros((num_nodes, 1), dtype=np.bool_)
    arr[item_lo:item_hi, 0] = True

    dataset_dir = DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dataset_dir / "node_role.npy"
    np.save(dst_path, arr)
    log(
        f"wrote {dst_path} shape={arr.shape} dtype={arr.dtype} "
        f"user=False for [0->{item_lo}), item=True for [{item_lo}->{item_hi})"
    )
    return dst_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bool node_role.npy for bipartite datasets in DATA/<dataset>/."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(ROLE_CONFIG),
        help="dataset name to generate; repeatable. Defaults to all supported datasets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = args.dataset or list(ROLE_CONFIG)
    for dataset_name in datasets:
        generate_node_role(dataset_name, ROLE_CONFIG[dataset_name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
