#!/usr/bin/env python3
"""Benchmark NVLink P2P all_to_all_single latency vs payload size.

Run with torchrun, e.g.:
  torchrun --standalone --nproc_per_node=2 experiments/microbench/nvlink_p2p_all_to_all.py

Each rank sends the same total payload size per iteration, split evenly to all
peers with ``torch.distributed.all_to_all_single``.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_csv_path(raw: str) -> Path | None:
    if raw.lower() == "none":
        return None
    if raw.lower() == "auto":
        return _script_dir() / "nvlink_p2p_all_to_all.csv"
    return Path(raw).expanduser().resolve()


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "requested_size_bytes",
                "actual_size_bytes",
                "size_mb",
                "avg_ms",
                "p50_ms",
                "p95_ms",
                "gbps",
                "rank_avg_ms_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_sizes_mb(raw: str) -> list[int]:
    sizes = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        mb = float(x)
        if mb <= 0:
            raise ValueError(f"Invalid size {x}; all sizes must be > 0 MB")
        sizes.append(int(mb * 1024 * 1024))
    if not sizes:
        raise ValueError("No valid sizes parsed")
    return sizes


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def _size_to_elems(
    size_bytes: int, dtype: torch.dtype, world_size: int
) -> tuple[int, int]:
    elem_size = torch.empty((), dtype=dtype).element_size()
    n_elem = size_bytes // elem_size
    # all_to_all_single with split sizes works best with equal chunks; round down.
    n_elem = (n_elem // world_size) * world_size
    if n_elem < world_size:
        raise ValueError(
            f"Payload {size_bytes} bytes too small for world_size={world_size} and dtype={dtype}"
        )
    actual_bytes = n_elem * elem_size
    return n_elem, actual_bytes


def _bench_one_size(
    size_bytes: int,
    repeats: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict:
    n_elem, actual_bytes = _size_to_elems(size_bytes, dtype, world_size)
    chunk = n_elem // world_size

    send_t = torch.full((n_elem,), float(rank), dtype=dtype, device=device)
    recv_t = torch.empty_like(send_t)
    splits = [chunk] * world_size

    # Align all ranks before timing.
    dist.barrier()

    for _ in range(warmup):
        work = dist.all_to_all_single(
            output=recv_t,
            input=send_t,
            output_split_sizes=splits,
            input_split_sizes=splits,
            async_op=True,
        )
        assert work is not None
        work.wait()
        torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        work = dist.all_to_all_single(
            output=recv_t,
            input=send_t,
            output_split_sizes=splits,
            input_split_sizes=splits,
            async_op=True,
        )
        assert work is not None
        work.wait()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    # Keep compiler/runtime from eliding recv_t usage in future optimizations.
    _ = recv_t[0].item()

    avg_ms = statistics.mean(samples_ms)
    p50_ms = statistics.median(samples_ms)
    p95_ms = sorted(samples_ms)[max(0, int(len(samples_ms) * 0.95) - 1)]
    gbps = (actual_bytes / (1024**3)) / (avg_ms / 1000.0)

    return {
        "requested_size_bytes": size_bytes,
        "actual_size_bytes": actual_bytes,
        "size_mb": actual_bytes / (1024**2),
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "gbps": gbps,
    }


def _init_dist() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NVLink P2P all_to_all_single time vs data size"
    )
    parser.add_argument(
        "--sizes-mb",
        type=str,
        default="1,2,4,8,16,32,64,128,256",
        help="Comma-separated payload sizes in MB (per-rank total send size)",
    )
    parser.add_argument("--repeats", type=int, default=30, help="Timed repeats")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup iterations")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data dtype: float16|float32|float64|int32|int64|uint8",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="auto",
        help="CSV output path. Use 'auto' to save beside this script, or 'none' to disable",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available; this benchmark requires GPU")
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be > 0 and warmup must be >= 0")

    rank, world_size, local_rank = _init_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dtype = _resolve_dtype(args.dtype)
    sizes = _parse_sizes_mb(args.sizes_mb)
    csv_path = _resolve_csv_path(args.csv)

    if rank == 0:
        print("# NVLink P2P all_to_all_single benchmark")
        print(
            f"# world_size={world_size} dtype={dtype} repeats={args.repeats} warmup={args.warmup}"
        )
        if world_size < 2:
            print("# warning: world_size < 2, this is not a real NVLink multi-GPU run")
        print("size_mb,avg_ms,p50_ms,p95_ms,gbps,rank_avg_ms_std")

    rows = []
    for size_bytes in sizes:
        local = _bench_one_size(
            size_bytes=size_bytes,
            repeats=args.repeats,
            warmup=args.warmup,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)

        if rank == 0:
            gathered_dicts: list[dict] = []
            for item in gathered:
                assert item is not None
                gathered_dicts.append(item)
            rank_avg_ms = [x["avg_ms"] for x in gathered_dicts]
            agg = {
                "requested_size_bytes": local["requested_size_bytes"],
                "actual_size_bytes": local["actual_size_bytes"],
                "size_mb": local["size_mb"],
                "avg_ms": statistics.mean(rank_avg_ms),
                "p50_ms": statistics.median([x["p50_ms"] for x in gathered_dicts]),
                "p95_ms": max(x["p95_ms"] for x in gathered_dicts),
                "gbps": statistics.mean([x["gbps"] for x in gathered_dicts]),
                "rank_avg_ms_std": statistics.pstdev(rank_avg_ms)
                if len(rank_avg_ms) > 1
                else 0.0,
            }
            rows.append(agg)
            print(
                f"{agg['size_mb']:.2f},"
                f"{agg['avg_ms']:.4f},"
                f"{agg['p50_ms']:.4f},"
                f"{agg['p95_ms']:.4f},"
                f"{agg['gbps']:.3f},"
                f"{agg['rank_avg_ms_std']:.4f}"
            )

    if rank == 0 and csv_path is not None:
        _write_csv(rows, csv_path)
        print(f"# wrote csv: {csv_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
