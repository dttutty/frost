#!/usr/bin/env python3
import argparse
import logging
import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

# We can use the resolve_dataset_dir from our codebase
from frost.neg_sampler._utils import resolve_dataset_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_negatives(
    dataset_name: str,
    strategy: str = "hist_rnd",
    num_neg: int = 100,
    seed: int = 123,
    data_root: str | None = None,
):
    """
    Generate negative samples for validation and test splits using TGB strategies.

    Args:
        dataset_name: Name of the dataset (e.g., LASTFM).
        strategy: 'rnd' or 'hist_rnd'.
        num_neg: Number of negatives per positive edge.
        seed: Random seed.
        data_root: Root directory for datasets.
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset_dir = resolve_dataset_dir(dataset_name, data_root=data_root)
    edges_path = dataset_dir / "edges.csv"

    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found at {edges_path}")

    logger.info(f"Loading edges from {edges_path}...")
    df = pd.read_csv(edges_path)

    # Requirement: columns 'src', 'dst', 'time', 'default_split'
    required_cols = {"src", "dst", "time", "default_split"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain {required_cols}. Found: {df.columns.tolist()}"
        )

    # Determine node universe
    all_nodes = np.union1d(df["src"].unique(), df["dst"].unique())
    first_node = int(all_nodes.min())
    last_node = int(all_nodes.max())
    all_dst_pool = np.arange(first_node, last_node + 1)

    logger.info(
        f"Node universe: [{first_node}, {last_node}], total {len(all_dst_pool)} nodes."
    )

    train_df = df[df["default_split"] == "train"]
    val_df = df[df["default_split"] == "val"]
    test_df = df[df["default_split"] == "test"]

    logger.info(
        f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    # Pre-calculate historical destinations for hist_rnd
    hist_per_src = {}
    if strategy == "hist_rnd":
        logger.info("Building historical edge map from training data...")
        for src, dst in zip(train_df["src"], train_df["dst"]):
            if src not in hist_per_src:
                hist_per_src[src] = {dst}
            else:
                hist_per_src[src].add(dst)

        # Convert to numpy arrays for faster set operations
        for src in hist_per_src:
            hist_per_src[src] = np.array(list(hist_per_src[src]))

    def sample_for_split(split_df, split_name):
        filename = dataset_dir / f"{dataset_name.lower()}_{split_name}_ns.pkl"
        if filename.exists():
            logger.info(
                f"File {filename} already exists. Skipping generation for {split_name}."
            )
            return

        logger.info(
            f"Generating negatives for {split_name} split using strategy='{strategy}'..."
        )

        # Group by (ts, src) to identify all positive destinations at the same time
        # This is for the TGB dynamic collision check (same-timestamp-same-source)
        ts_src_pos = split_df.groupby(["time", "src"])["dst"].apply(set).to_dict()

        evaluation_set = {}
        pos_edges = list(zip(split_df["src"], split_df["dst"], split_df["time"]))

        for src, dst, ts in tqdm(pos_edges, desc=f"Sampling {split_name}"):
            # TGB Collision: destination must not be a positive friend at the SAME timestamp
            pos_friends_at_ts = np.array(list(ts_src_pos[(ts, src)]))

            neg_dsts = []

            if strategy == "hist_rnd":
                # 50% historical, 50% random
                max_hist = num_neg // 2
                hist_pool = hist_per_src.get(src, np.array([]))

                # Filter out current positives from historical pool
                valid_hist = np.setdiff1d(hist_pool, pos_friends_at_ts)

                if len(valid_hist) > 0:
                    n_hist = min(len(valid_hist), max_hist)
                    neg_hist = np.random.choice(valid_hist, n_hist, replace=False)
                    neg_dsts.extend(neg_hist.tolist())

                # Sample random to reach num_neg
                needed = num_neg - len(neg_dsts)
                # Filter out current positives AND already selected historical negatives from the full pool
                forbidden = np.union1d(pos_friends_at_ts, np.array(neg_dsts))
                # For strictness, TGB often also filters hist_pool from the random part in hist_rnd
                forbidden = np.union1d(forbidden, hist_pool)

                valid_rnd = np.setdiff1d(all_dst_pool, forbidden)
                if len(valid_rnd) < needed:
                    # If we don't have enough strictly valid nodes, just relax the 'hist_pool' exclusion
                    forbidden_relaxed = pos_friends_at_ts
                    valid_rnd = np.setdiff1d(all_dst_pool, forbidden_relaxed)

                if len(valid_rnd) >= needed:
                    neg_rnd = np.random.choice(valid_rnd, needed, replace=False)
                    neg_dsts.extend(neg_rnd.tolist())
                else:
                    # Extreme fallback: allow replacement
                    neg_rnd = np.random.choice(valid_rnd, needed, replace=True)
                    neg_dsts.extend(neg_rnd.tolist())

            else:  # strategy == 'rnd'
                forbidden = pos_friends_at_ts
                valid_rnd = np.setdiff1d(all_dst_pool, forbidden)

                if len(valid_rnd) >= num_neg:
                    neg_dsts = np.random.choice(
                        valid_rnd, num_neg, replace=False
                    ).tolist()
                else:
                    neg_dsts = np.random.choice(
                        valid_rnd, num_neg, replace=True
                    ).tolist()

            evaluation_set[(int(src), int(dst), float(ts))] = np.array(
                neg_dsts, dtype=np.int64
            )

        # Save to disk
        logger.info(f"Saving {len(evaluation_set)} records to {filename}")
        with open(filename, "wb") as f:
            pickle.dump(evaluation_set, f)

    sample_for_split(val_df, "val")
    sample_for_split(test_df, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TGB negative samples (.pkl)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--strategy", type=str, default="hist_rnd", choices=["rnd", "hist_rnd"]
    )
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data_root", type=str, default=None)

    args = parser.parse_args()

    generate_negatives(
        dataset_name=args.dataset,
        strategy=args.strategy,
        num_neg=args.num_neg,
        seed=args.seed,
        data_root=args.data_root,
    )
