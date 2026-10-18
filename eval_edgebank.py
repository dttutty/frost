"""
EdgeBank standalone evaluation script for FROST datasets.

EdgeBank is a heuristic baseline with no trainable parameters:
  - Maintains a set of (src, dst) pairs seen during training.
  - Predicts 1.0 if the queried edge is in memory, 0.0 otherwise.

Three memory modes (matching the reference paper):
  - unlimited_memory    : store every training edge
  - time_window_memory  : store edges in the most recent time window
  - repeat_threshold    : store only edges that appear >= mean frequency

Usage:
    python eval_edgebank.py --dataset LASTFM
    python eval_edgebank.py --dataset LASTFM --memory_mode time_window_memory
    python eval_edgebank.py --dataset LASTFM --memory_mode repeat_threshold
    python eval_edgebank.py --dataset LASTFM --split int_roll  # inductive split

The script reads DATA/{dataset}/edges.csv and evaluates on val and test sets,
reporting Average Precision (AP) and AUC-ROC for each, matching the metrics
used by the rest of the FROST framework.
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Edge memory builders
# ---------------------------------------------------------------------------


def build_unlimited_memory(src: np.ndarray, dst: np.ndarray) -> set[tuple[int, int]]:
    """Store every (src, dst) pair seen in history."""
    return set(zip(src.tolist(), dst.tolist()))


def build_time_window_memory(
    src: np.ndarray,
    dst: np.ndarray,
    ts: np.ndarray,
    time_window_mode: str = "fixed_proportion",
    time_window_proportion: float = 0.15,
) -> set[tuple[int, int]]:
    """
    Store only edges whose timestamp falls within a recent time window.

    time_window_mode:
      'fixed_proportion'  — window covers the last `time_window_proportion`
                            fraction of the training time span.
      'repeat_interval'   — window width = average re-occurrence interval
                            across all repeated edges.
    """
    if time_window_mode == "fixed_proportion":
        start = float(np.quantile(ts, 1.0 - time_window_proportion))
        end = float(ts.max())
    elif time_window_mode == "repeat_interval":
        edge_times: dict[tuple[int, int], list[float]] = defaultdict(list)
        for s, d, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
            edge_times[(s, d)].append(t)

        total_interval = 0.0
        count = 0
        for times in edge_times.values():
            if len(times) > 1:
                times_sorted = sorted(times)
                total_interval += float(
                    np.mean(
                        [
                            times_sorted[i + 1] - times_sorted[i]
                            for i in range(len(times_sorted) - 1)
                        ]
                    )
                )
                count += 1

        avg_interval = total_interval / max(count, 1)
        end = float(ts.max())
        start = end - avg_interval
    else:
        raise ValueError(f"Unknown time_window_mode: {time_window_mode!r}")

    mask = (ts >= start) & (ts <= end)
    return build_unlimited_memory(src[mask], dst[mask])


def build_repeat_threshold_memory(
    src: np.ndarray, dst: np.ndarray
) -> set[tuple[int, int]]:
    """Store only edges whose frequency >= mean frequency across all edges."""
    freq: dict[tuple[int, int], int] = defaultdict(int)
    for s, d in zip(src.tolist(), dst.tolist()):
        freq[(s, d)] += 1

    threshold = float(np.mean(list(freq.values())))
    return {edge for edge, cnt in freq.items() if cnt >= threshold}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    memory: set[tuple[int, int]],
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """Return 1.0 if (src, dst) is in memory, else 0.0."""
    return np.array(
        [
            1.0 if (int(s), int(d)) in memory else 0.0
            for s, d in zip(src.tolist(), dst.tolist())
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Negative sampling  (uniform random destination, matching FROST's sampler)
# ---------------------------------------------------------------------------


def sample_negatives(
    src: np.ndarray,
    n_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one random negative destination per source."""
    return rng.integers(0, n_nodes, size=len(src)).astype(np.int64)


# ---------------------------------------------------------------------------
# Evaluation for one split
# ---------------------------------------------------------------------------


def evaluate_split(
    memory: set[tuple[int, int]],
    src: np.ndarray,
    dst: np.ndarray,
    n_nodes: int,
    rng: np.random.Generator,
    n_neg: int = 1,
) -> dict[str, float]:
    pos_scores = predict(memory, src, dst)

    neg_scores_list = []
    for _ in range(n_neg):
        neg_dst = sample_negatives(src, n_nodes, rng)
        neg_scores_list.append(predict(memory, src, neg_dst))
    neg_scores = np.concatenate(neg_scores_list)

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate(
        [
            np.ones(len(pos_scores), dtype=np.float32),
            np.zeros(len(neg_scores), dtype=np.float32),
        ]
    )

    ap = float(average_precision_score(all_labels, all_scores))
    auc = float(roc_auc_score(all_labels, all_scores))
    return {"ap": ap, "auc": auc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="EdgeBank evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="LASTFM",
        help="Dataset name (must exist under DATA/)",
    )
    parser.add_argument(
        "--memory_mode",
        type=str,
        default="unlimited_memory",
        choices=["unlimited_memory", "time_window_memory", "repeat_threshold"],
        help="EdgeBank memory mode",
    )
    parser.add_argument(
        "--time_window_mode",
        type=str,
        default="fixed_proportion",
        choices=["fixed_proportion", "repeat_interval"],
        help="Sub-mode for time_window_memory",
    )
    parser.add_argument(
        "--time_window_proportion",
        type=float,
        default=0.15,
        help="Fraction of training time span used as the window (fixed_proportion mode)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="default_split",
        choices=["default_split", "int_roll"],
        help="Column used for train/val/test split (0=train, 1=val, 2=test)",
    )
    parser.add_argument(
        "--n_neg",
        type=int,
        default=1,
        help="Number of negative samples per positive edge",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df = pd.read_csv(f"DATA/{args.dataset}/edges.csv")

    src_col = df["src"].to_numpy(dtype=np.int64)
    dst_col = df["dst"].to_numpy(dtype=np.int64)
    ts_col = df["time"].to_numpy(dtype=np.float64)
    roll = df[args.split].to_numpy(dtype=np.int64)

    n_nodes = int(max(src_col.max(), dst_col.max())) + 1

    train_mask = roll == 0
    val_mask = roll == 1
    test_mask = roll == 2

    print(f"Dataset : {args.dataset}  ({args.split} split)")
    print(f"Nodes   : {n_nodes:,}")
    print(
        f"Train   : {train_mask.sum():,}  Val: {val_mask.sum():,}  Test: {test_mask.sum():,}"
    )
    print(f"Memory  : {args.memory_mode}")

    # -----------------------------------------------------------------------
    # Build edge memory from training set
    # -----------------------------------------------------------------------
    tr_src, tr_dst, tr_ts = src_col[train_mask], dst_col[train_mask], ts_col[train_mask]

    if args.memory_mode == "unlimited_memory":
        memory = build_unlimited_memory(tr_src, tr_dst)
    elif args.memory_mode == "time_window_memory":
        memory = build_time_window_memory(
            tr_src,
            tr_dst,
            tr_ts,
            time_window_mode=args.time_window_mode,
            time_window_proportion=args.time_window_proportion,
        )
    else:  # repeat_threshold
        memory = build_repeat_threshold_memory(tr_src, tr_dst)

    print(f"Memory size: {len(memory):,} unique edges\n")

    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    val_metrics = evaluate_split(
        memory,
        src_col[val_mask],
        dst_col[val_mask],
        n_nodes,
        rng,
        n_neg=args.n_neg,
    )
    test_metrics = evaluate_split(
        memory,
        src_col[test_mask],
        dst_col[test_mask],
        n_nodes,
        rng,
        n_neg=args.n_neg,
    )

    print(f"Val   AP={val_metrics['ap']:.4f}  AUC={val_metrics['auc']:.4f}")
    print(f"Test  AP={test_metrics['ap']:.4f}  AUC={test_metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
