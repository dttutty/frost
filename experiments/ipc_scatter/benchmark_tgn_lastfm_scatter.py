"""TGN/LASTFM IPC scatter microbenchmark.

Run from the repo root with the TGL environment because the comparison payload
uses DGL blocks from ``third_party/tgl``:

    uv run --project third_party/tgl python -m torch.distributed.run \
        --nproc_per_node=3 --master_port=29637 \
        experiments/ipc_scatter/benchmark_tgn_lastfm_scatter.py

The benchmark times only the host-side ``dist.scatter_object_list`` calls. It
does not include FROST shared-memory reads, H2D transfer, or model execution.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Queue
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TGL_ROOT = REPO_ROOT / "third_party" / "tgl"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TGL_ROOT) not in sys.path:
    sys.path.insert(0, str(TGL_ROOT))

from df_split import make_balance_plan  # noqa: E402
from sampler import ParallelSampler  # noqa: E402
from utils import to_dgl_blocks  # noqa: E402

from frost.datatypes.meta import Meta  # noqa: E402
from frost.datatypes.plan import PlanSet  # noqa: E402
from frost.runtime.host.p2p_planner._dynamic_planner import DynamicPlanner  # noqa: E402
from frost.utils.tensor_set import TensorSet  # noqa: E402


@dataclass
class BatchPayload:
    metas: list[Meta]
    needs: list[TensorSet]
    updates: list[TensorSet]
    raw_need_counts: list[int]
    tgl_mfgs: list[object]
    tgl_roots: list[np.ndarray]
    tgl_ts: list[np.ndarray]
    tgl_eids: list[np.ndarray]


class DstPartitionNegSampler:
    """TGL/FROST parity negative sampler: uniform over observed dst nodes."""

    def __init__(self, df: pd.DataFrame, *, seed: int | None) -> None:
        candidates = np.unique(df["dst"].to_numpy(dtype=np.int64, copy=False))
        if candidates.size == 0:
            raise ValueError("Negative sampler needs at least one dst candidate.")
        self.candidates = candidates
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> np.ndarray:
        return self.rng.choice(self.candidates, size=n, replace=True).astype(np.int64)


class FrostTgnPlanLookahead:
    """Build the PlanSet payload that FROST scatters for TGN memory models."""

    def __init__(self, n_gpu: int) -> None:
        gpu_dist = np.ones((n_gpu, n_gpu), dtype=np.int32)
        np.fill_diagonal(gpu_dist, 0)
        self.n_gpu = n_gpu
        self._stop_ev = threading.Event()
        self.mem = DynamicPlanner(
            gpus_distance=gpu_dist,
            in_q=Queue(),
            out_q=Queue(),
            stop_ev=self._stop_ev,
            if_carry_over_route=False,
            plan_kind="mem",
        )
        self.mail = DynamicPlanner(
            gpus_distance=gpu_dist,
            in_q=Queue(),
            out_q=Queue(),
            stop_ev=self._stop_ev,
            if_carry_over_route=False,
            plan_kind="mail",
        )
        self.pending: BatchPayload | None = None

    def push(self, current: BatchPayload) -> BatchPayload | None:
        if self.pending is None:
            self.pending = current
            self.mem.fetch = current.needs
            self.mail.fetch = current.needs
            return None

        planned = self._plan_pending(next_batch=current)
        self.pending = current
        return planned

    def finish(self) -> BatchPayload | None:
        if self.pending is None:
            return None

        ref = (
            self.pending.needs[0].data
            if self.pending.needs
            else torch.empty(0, dtype=torch.long)
        )
        empty = [
            TensorSet(torch.empty(0, dtype=ref.dtype, device=ref.device))
            for _ in range(self.n_gpu)
        ]
        planned = self._plan_pending(
            next_needs=empty,
            next_updates=empty,
            next_batch=None,
        )
        self.pending = None
        return planned

    def _plan_pending(
        self,
        *,
        next_batch: BatchPayload | None = None,
        next_needs: list[TensorSet] | None = None,
        next_updates: list[TensorSet] | None = None,
    ) -> BatchPayload:
        assert self.pending is not None
        if next_batch is not None:
            next_needs = next_batch.needs
            next_updates = next_batch.updates
        assert next_needs is not None
        assert next_updates is not None

        mem_plans = self.mem._plan(
            self.pending.needs,
            self.pending.updates,
            next_needs,
            next_updates,
            self.pending.metas,
            raw_need_counts=self.pending.raw_need_counts,
        )
        mail_plans = self.mail._plan(
            self.pending.needs,
            self.pending.updates,
            next_needs,
            next_updates,
            self.pending.metas,
            raw_need_counts=self.pending.raw_need_counts,
        )

        planned_metas = []
        for meta, mem_plan, mail_plan in zip(
            self.pending.metas, mem_plans, mail_plans, strict=True
        ):
            planned_metas.append(
                replace(
                    meta,
                    plan_set=PlanSet(
                        mem_plan=mem_plan,
                        box_plan=mail_plan,
                        nfeat_plan=None,
                    ),
                )
            )

        return replace(self.pending, metas=planned_metas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare host->worker scatter_object_list time for FROST TGN "
            "metadata payloads vs third_party/tgl DGL graph payloads."
        )
    )
    parser.add_argument("--dataset", default="LASTFM")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument(
        "--macro-batch-sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 4000],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Splits included in the one-epoch scatter profile.",
    )
    parser.add_argument("--neighbor", type=int, default=10)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--num-thread", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-macro-batches",
        type=int,
        default=0,
        help="Optional smoke-test limit across the selected splits. 0 means full epoch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "ipc_scatter",
    )
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument(
        "--tqdm", action="store_true", help="Show host-side progress bars."
    )
    return parser.parse_args()


def load_edges(dataset: str) -> pd.DataFrame:
    path = REPO_ROOT / "DATA" / dataset / "edges.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing edge CSV: {path}")

    df = pd.read_csv(path)
    if "time" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "time"})
    required = {"eid", "src", "dst", "time", "default_split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    for column in required:
        df[column] = df[column].astype(np.int64)
    return df.reset_index(drop=True)


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    split = df["default_split"].to_numpy()
    val_mask = split > 0
    test_mask = split > 1
    if not val_mask.any() or not test_mask.any():
        raise ValueError("Expected default_split boundaries for train/val/test.")
    train_end = int(np.flatnonzero(val_mask)[0])
    val_end = int(np.flatnonzero(test_mask)[0])
    return {
        "train": df.iloc[:train_end].reset_index(drop=True),
        "val": df.iloc[train_end:val_end].reset_index(drop=True),
        "test": df.iloc[val_end:].reset_index(drop=True),
    }


def load_tgl_graph(dataset: str) -> dict[str, np.ndarray]:
    path = REPO_ROOT / "DATA" / dataset / "full_graph_with_reverse_edges.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing temporal graph NPZ: {path}")
    with np.load(path, allow_pickle=False) as npz:
        return {
            "indptr": np.ascontiguousarray(npz["indptr"], dtype=np.int64),
            "indices": np.ascontiguousarray(npz["indices"], dtype=np.int64),
            "eid": np.ascontiguousarray(npz["eid"], dtype=np.int64),
            "ts": np.ascontiguousarray(npz["ts"], dtype=np.int64),
        }


def count_epoch_batches(
    frames: dict[str, pd.DataFrame],
    *,
    splits: Iterable[str],
    per_gpu_batch_size: int,
    n_gpu: int,
    limit_macro_batches: int,
) -> int:
    total = 0
    for split in splits:
        _, step_ids, _ = make_balance_plan(frames[split], per_gpu_batch_size, n_gpu)
        total += len(set(step_ids))
    if limit_macro_batches > 0:
        total = min(total, limit_macro_batches)
    return total


def build_one_batch(
    *,
    step_df: pd.DataFrame,
    sampler: ParallelSampler,
    neg_sampler: DstPartitionNegSampler,
    n_gpu: int,
    neighbor: int,
    history: int,
    split: str,
    cur_epoch: int,
    cur_macro_of_split: int,
    n_macro_of_split: int,
    cur_macro_of_epoch: int,
    mini_batch_idx_start: int,
) -> BatchPayload:
    metas: list[Meta] = []
    needs: list[TensorSet] = []
    updates: list[TensorSet] = []
    raw_need_counts: list[int] = []
    tgl_mfgs: list[object] = []
    tgl_roots: list[np.ndarray] = []
    tgl_ts: list[np.ndarray] = []
    tgl_eids: list[np.ndarray] = []

    for gpu in range(n_gpu):
        rows = step_df[step_df["_gpu_id"] == gpu]
        if rows.empty:
            raise ValueError(
                f"Balanced step {cur_macro_of_split} unexpectedly has no rows for GPU {gpu}."
            )

        batch_src = rows["src"].to_numpy(dtype=np.int64, copy=False)
        batch_dst = rows["dst"].to_numpy(dtype=np.int64, copy=False)
        batch_eid = rows["eid"].to_numpy(dtype=np.int64, copy=False)
        batch_ts = rows["time"].to_numpy(dtype=np.int64, copy=False)
        neg_dst = neg_sampler.sample(len(rows))

        root_nodes = np.concatenate([batch_src, batch_dst, neg_dst]).astype(np.int32)
        root_ts = np.concatenate([batch_ts, batch_ts, batch_ts]).astype(np.int64)

        sampler.sample(root_nodes, root_ts)
        ret = sampler.get_ret()
        mfgs = to_dgl_blocks(ret, history, cuda=False)

        tgl_mfgs.append(mfgs)
        tgl_roots.append(root_nodes)
        tgl_ts.append(root_ts)
        tgl_eids.append(batch_eid.astype(np.int64, copy=True))

        ret_nodes = np.asarray(ret[0].nodes(), dtype=np.int64)
        valid_nodes = torch.from_numpy(ret_nodes[ret_nodes != -1].copy()).to(torch.long)
        positive_roots = torch.from_numpy(root_nodes[: 2 * len(rows)].astype(np.int64))

        needs.append(TensorSet(valid_nodes))
        updates.append(TensorSet(positive_roots))
        raw_need_counts.append(int(valid_nodes.numel()))

        n_src = int(len(rows))
        buf_len = int((2 + 1) * n_src * (1 + neighbor))
        metas.append(
            Meta(
                cur_epoch=cur_epoch,
                cur_macro_of_split=int(cur_macro_of_split),
                n_macro_of_split=int(n_macro_of_split),
                cur_macro_of_epoch=int(cur_macro_of_epoch),
                split=split,  # type: ignore[arg-type]
                n_src=n_src,
                mini_batch_idx=mini_batch_idx_start + gpu,
                flat_graph=None,
                buf_len=buf_len,
                n_neg=1,
                fanout=[neighbor],
            )
        )

    return BatchPayload(
        metas=metas,
        needs=needs,
        updates=updates,
        raw_need_counts=raw_need_counts,
        tgl_mfgs=tgl_mfgs,
        tgl_roots=tgl_roots,
        tgl_ts=tgl_ts,
        tgl_eids=tgl_eids,
    )


def iter_epoch_payloads(
    *,
    frames: dict[str, pd.DataFrame],
    splits: list[str],
    per_gpu_batch_size: int,
    n_gpu: int,
    sampler: ParallelSampler,
    neg_sampler: DstPartitionNegSampler,
    neighbor: int,
    history: int,
    limit_macro_batches: int,
) -> Iterable[BatchPayload]:
    cur_macro_of_epoch = 0
    mini_batch_idx = 0

    for split in splits:
        frame = frames[split]
        df_out, step_ids, gpu_ids = make_balance_plan(frame, per_gpu_batch_size, n_gpu)
        df_out = df_out.copy()
        df_out["_step_id"] = step_ids
        df_out["_gpu_id"] = gpu_ids
        n_macro_of_split = int(df_out["_step_id"].nunique())

        for cur_macro_of_split, step_df in df_out.groupby("_step_id", sort=False):
            if limit_macro_batches > 0 and cur_macro_of_epoch >= limit_macro_batches:
                return
            payload = build_one_batch(
                step_df=step_df,
                sampler=sampler,
                neg_sampler=neg_sampler,
                n_gpu=n_gpu,
                neighbor=neighbor,
                history=history,
                split=split,
                cur_epoch=0,
                cur_macro_of_split=int(cur_macro_of_split),
                n_macro_of_split=n_macro_of_split,
                cur_macro_of_epoch=cur_macro_of_epoch,
                mini_batch_idx_start=mini_batch_idx,
            )
            yield payload
            cur_macro_of_epoch += 1
            mini_batch_idx += n_gpu


def scatter_host(per_worker: list[object], *, world_size: int, host_rank: int) -> None:
    scatter_input: list[object | None] = [None] * world_size
    for rank, obj in enumerate(per_worker):
        scatter_input[rank] = obj
    output = [None]
    dist.scatter_object_list(
        output, scatter_object_input_list=scatter_input, src=host_rank
    )


def timed_scatter_host(
    label: str,
    per_worker: list[object],
    *,
    world_size: int,
    host_rank: int,
    times_ms: dict[str, list[float]],
) -> None:
    start = time.perf_counter()
    scatter_host(per_worker, world_size=world_size, host_rank=host_rank)
    times_ms.setdefault(label, []).append((time.perf_counter() - start) * 1000.0)


def recv_worker(*, world_size: int, host_rank: int) -> object:
    output = [None]
    dist.scatter_object_list(
        output,
        scatter_object_input_list=[None] * world_size,
        src=host_rank,
    )
    return output[0]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def summarize_times(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "total_ms": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
        }
    return {
        "n": len(values),
        "total_ms": float(sum(values)),
        "avg_ms": float(statistics.fmean(values)),
        "p50_ms": float(percentile(values, 0.50)),
        "p90_ms": float(percentile(values, 0.90)),
        "p99_ms": float(percentile(values, 0.99)),
    }


def add_savings(
    row: dict[str, float | int], *, base: str, frost: str, prefix: str
) -> None:
    base_total = float(row[f"{base}_total_ms"])
    frost_total = float(row[f"{frost}_total_ms"])
    saved = base_total - frost_total
    row[f"{prefix}_saved_ms"] = saved
    row[f"{prefix}_saved_pct"] = (saved / base_total * 100.0) if base_total > 0 else 0.0


def build_summary_row(
    *,
    macro_batch_size: int,
    per_gpu_batch_size: int,
    n_batches: int,
    times_ms: dict[str, list[float]],
) -> dict[str, float | int]:
    labels = [
        "frost_meta_only",
        "frost_meta_plan",
        "tgl_mfgs",
        "tgl_root",
        "tgl_ts",
        "tgl_eid",
    ]
    row: dict[str, float | int] = {
        "macro_batch_size": macro_batch_size,
        "per_gpu_batch_size": per_gpu_batch_size,
        "n_macro_batches": n_batches,
    }
    for label in labels:
        stats = summarize_times(times_ms.get(label, []))
        for key, value in stats.items():
            row[f"{label}_{key}"] = value

    aux_total = (
        float(row["tgl_root_total_ms"])
        + float(row["tgl_ts_total_ms"])
        + float(row["tgl_eid_total_ms"])
    )
    aux_avg = aux_total / n_batches if n_batches > 0 else 0.0
    payload_total = float(row["tgl_mfgs_total_ms"]) + aux_total
    row["tgl_tgn_aux_total_ms"] = aux_total
    row["tgl_tgn_aux_avg_ms"] = aux_avg
    row["tgl_tgn_payload_total_ms"] = payload_total
    row["tgl_tgn_payload_avg_ms"] = payload_total / n_batches if n_batches > 0 else 0.0

    add_savings(
        row,
        base="tgl_mfgs",
        frost="frost_meta_only",
        prefix="tgl_mfgs_vs_frost_meta_only",
    )
    add_savings(
        row,
        base="tgl_mfgs",
        frost="frost_meta_plan",
        prefix="tgl_mfgs_vs_frost_meta_plan",
    )
    base_total = float(row["tgl_tgn_payload_total_ms"])
    for frost in ("frost_meta_only", "frost_meta_plan"):
        frost_total = float(row[f"{frost}_total_ms"])
        saved = base_total - frost_total
        row[f"tgl_tgn_payload_vs_{frost}_saved_ms"] = saved
        row[f"tgl_tgn_payload_vs_{frost}_saved_pct"] = (
            saved / base_total * 100.0 if base_total > 0 else 0.0
        )
    return row


def print_summary(rows: list[dict[str, float | int]]) -> None:
    columns = [
        "macro_batch_size",
        "per_gpu_batch_size",
        "n_macro_batches",
        "frost_meta_plan_total_ms",
        "tgl_mfgs_total_ms",
        "tgl_mfgs_vs_frost_meta_plan_saved_ms",
        "tgl_mfgs_vs_frost_meta_plan_saved_pct",
        "tgl_tgn_payload_total_ms",
        "tgl_tgn_payload_vs_frost_meta_plan_saved_ms",
        "tgl_tgn_payload_vs_frost_meta_plan_saved_pct",
    ]
    print("\nIPC scatter summary (host-side scatter_object_list time only)")
    print(",".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        print(",".join(values))


def write_outputs(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, float | int]],
    raw: dict[int, dict[str, list[float]]],
) -> None:
    run_dir = args.output_dir / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "scatter_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path = run_dir / "scatter_summary.json"
    with json_path.open("w", encoding="ascii") as f:
        json.dump(
            {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "summary": rows,
            },
            f,
            indent=2,
        )

    if args.save_raw:
        raw_path = run_dir / "scatter_raw.csv"
        with raw_path.open("w", newline="", encoding="ascii") as f:
            writer = csv.writer(f)
            writer.writerow(["macro_batch_size", "label", "batch_idx", "ms"])
            for macro_batch_size, by_label in raw.items():
                for label, values in by_label.items():
                    for batch_idx, value in enumerate(values):
                        writer.writerow([macro_batch_size, label, batch_idx, value])

    print(f"\nWrote summary to {summary_path}")


def run_host(args: argparse.Namespace, *, world_size: int, host_rank: int) -> None:
    frames = split_frames(load_edges(args.dataset))
    graph = load_tgl_graph(args.dataset)
    neg_source_df = pd.concat(
        [frames[split] for split in args.splits], ignore_index=True
    )

    all_rows: list[dict[str, float | int]] = []
    raw_by_macro: dict[int, dict[str, list[float]]] = {}

    for macro_batch_size in args.macro_batch_sizes:
        if macro_batch_size % args.num_gpus != 0:
            raise ValueError(
                f"macro_batch_size={macro_batch_size} must be divisible by "
                f"num_gpus={args.num_gpus}."
            )
        per_gpu_batch_size = macro_batch_size // args.num_gpus
        n_batches = count_epoch_batches(
            frames,
            splits=args.splits,
            per_gpu_batch_size=per_gpu_batch_size,
            n_gpu=args.num_gpus,
            limit_macro_batches=args.limit_macro_batches,
        )
        scatter_host(
            [
                {
                    "kind": "begin",
                    "macro_batch_size": macro_batch_size,
                    "per_gpu_batch_size": per_gpu_batch_size,
                    "n_batches": n_batches,
                }
                for _ in range(args.num_gpus)
            ],
            world_size=world_size,
            host_rank=host_rank,
        )

        times_ms: dict[str, list[float]] = {}
        planner = FrostTgnPlanLookahead(args.num_gpus)
        neg_sampler = DstPartitionNegSampler(neg_source_df, seed=args.seed)
        sampler = ParallelSampler(
            graph["indptr"],
            graph["indices"],
            graph["eid"],
            graph["ts"],
            args.num_thread,
            1,
            1,
            [args.neighbor],
            True,
            False,
            args.history,
            0,
        )
        payload_iter = iter_epoch_payloads(
            frames=frames,
            splits=args.splits,
            per_gpu_batch_size=per_gpu_batch_size,
            n_gpu=args.num_gpus,
            sampler=sampler,
            neg_sampler=neg_sampler,
            neighbor=args.neighbor,
            history=args.history,
            limit_macro_batches=args.limit_macro_batches,
        )
        if args.tqdm:
            from tqdm import tqdm

            payload_iter = tqdm(
                payload_iter,
                total=n_batches,
                desc=f"macro_batch_size={macro_batch_size}",
            )

        processed = 0
        for current in payload_iter:
            planned = planner.push(current)
            if planned is None:
                continue
            scatter_payloads(
                planned, times_ms, world_size=world_size, host_rank=host_rank
            )
            processed += 1

        final = planner.finish()
        if final is not None:
            scatter_payloads(
                final, times_ms, world_size=world_size, host_rank=host_rank
            )
            processed += 1

        if processed != n_batches:
            raise RuntimeError(
                f"Expected to scatter {n_batches} macro-batches for macro_batch_size="
                f"{macro_batch_size}, scattered {processed}."
            )

        row = build_summary_row(
            macro_batch_size=macro_batch_size,
            per_gpu_batch_size=per_gpu_batch_size,
            n_batches=n_batches,
            times_ms=times_ms,
        )
        all_rows.append(row)
        raw_by_macro[macro_batch_size] = times_ms
        print_summary([row])

    scatter_host(
        [{"kind": "stop"} for _ in range(args.num_gpus)],
        world_size=world_size,
        host_rank=host_rank,
    )
    print_summary(all_rows)
    write_outputs(args=args, rows=all_rows, raw=raw_by_macro)


def scatter_payloads(
    payload: BatchPayload,
    times_ms: dict[str, list[float]],
    *,
    world_size: int,
    host_rank: int,
) -> None:
    meta_only = [replace(meta, plan_set=None) for meta in payload.metas]
    timed_scatter_host(
        "frost_meta_only",
        meta_only,
        world_size=world_size,
        host_rank=host_rank,
        times_ms=times_ms,
    )
    timed_scatter_host(
        "frost_meta_plan",
        payload.metas,
        world_size=world_size,
        host_rank=host_rank,
        times_ms=times_ms,
    )
    timed_scatter_host(
        "tgl_mfgs",
        payload.tgl_mfgs,
        world_size=world_size,
        host_rank=host_rank,
        times_ms=times_ms,
    )
    timed_scatter_host(
        "tgl_root",
        payload.tgl_roots,
        world_size=world_size,
        host_rank=host_rank,
        times_ms=times_ms,
    )
    timed_scatter_host(
        "tgl_ts",
        payload.tgl_ts,
        world_size=world_size,
        host_rank=host_rank,
        times_ms=times_ms,
    )
    timed_scatter_host(
        "tgl_eid",
        payload.tgl_eids,
        world_size=world_size,
        host_rank=host_rank,
        times_ms=times_ms,
    )


def run_worker(args: argparse.Namespace, *, world_size: int, host_rank: int) -> None:
    del args
    while True:
        control = recv_worker(world_size=world_size, host_rank=host_rank)
        if not isinstance(control, dict):
            raise TypeError(f"Expected control dict, got {type(control)!r}")
        kind = control.get("kind")
        if kind == "stop":
            return
        if kind != "begin":
            raise ValueError(f"Unexpected control message: {control!r}")

        n_batches = int(control["n_batches"])
        for _ in range(n_batches):
            recv_worker(world_size=world_size, host_rank=host_rank)  # frost_meta_only
            recv_worker(world_size=world_size, host_rank=host_rank)  # frost_meta_plan
            recv_worker(world_size=world_size, host_rank=host_rank)  # tgl_mfgs
            recv_worker(world_size=world_size, host_rank=host_rank)  # tgl_root
            recv_worker(world_size=world_size, host_rank=host_rank)  # tgl_ts
            recv_worker(world_size=world_size, host_rank=host_rank)  # tgl_eid


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    expected_world_size = args.num_gpus + 1
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Expected WORLD_SIZE={expected_world_size} for num_gpus={args.num_gpus}; "
            f"launch with nproc_per_node={expected_world_size}. Got WORLD_SIZE={world_size}."
        )

    host_rank = args.num_gpus
    dist.init_process_group(backend="gloo", timeout=dt.timedelta(hours=1))
    try:
        if rank == host_rank:
            run_host(args, world_size=world_size, host_rank=host_rank)
        else:
            run_worker(args, world_size=world_size, host_rank=host_rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
