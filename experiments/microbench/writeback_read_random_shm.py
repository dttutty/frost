#!/usr/bin/env python3
"""Benchmark writeback + random read total time on shared memory.

Flow per iteration:
1) Copy a CUDA tensor with shape [n_node, n_dim] into shared memory.
2) Read rows from shared memory using random node positions.

This measures the end-to-end latency of "writeback + read from shm".
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_import_path()

from frost.shm._core import create_shared_mem_array, set_as_host  # noqa: E402


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_csv_path(raw: str) -> Path | None:
    if raw.lower() == "none":
        return None
    if raw.lower() == "auto":
        return _script_dir() / "writeback_read_random_shm.csv"
    return Path(raw).expanduser().resolve()


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


def _parse_n_nodes(raw: str) -> list[int]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if v <= 0:
            raise ValueError(f"Invalid n_node={x}; all n_node must be > 0")
        vals.append(v)
    if not vals:
        raise ValueError("No valid n_node parsed")

    return vals


def _bench_one_n_node(
    n_node: int,
    n_dim: int,
    n_read: int,
    repeats: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict:
    shm_name = f"wb_rd_rand_{n_node}_{n_dim}_{dtype}"
    shm_t = create_shared_mem_array(name=shm_name, shape=(n_node, n_dim), dtype=dtype)

    gpu_src = torch.randn((n_node, n_dim), dtype=dtype, device=device)

    # Fixed random index set per size to keep runs comparable.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + n_node)
    idx = torch.randint(0, n_node, (n_read,), generator=g, dtype=torch.int64)

    # Warmup
    for _ in range(warmup):
        shm_t.copy_(gpu_src, non_blocking=False)
        _ = shm_t[idx]
        torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    write_ms: list[float] = []
    read_ms: list[float] = []

    for _ in range(repeats):
        # Measure writeback and random-read separately and also as total.
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        shm_t.copy_(gpu_src, non_blocking=False)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        _ = shm_t[idx]
        t2 = time.perf_counter()

        w_ms = (t1 - t0) * 1000.0
        r_ms = (t2 - t1) * 1000.0
        write_ms.append(w_ms)
        read_ms.append(r_ms)
        samples_ms.append((t2 - t0) * 1000.0)

    # Prevent future optimization from dropping read path.
    sink = shm_t[idx[: min(8, n_read)]].sum().item()

    elem_size = torch.empty((), dtype=dtype).element_size()
    total_bytes = n_node * n_dim * elem_size

    return {
        "n_node": n_node,
        "n_dim": n_dim,
        "n_read": n_read,
        "dtype": str(dtype),
        "total_mb": total_bytes / (1024**2),
        "write_avg_ms": statistics.mean(write_ms),
        "read_avg_ms": statistics.mean(read_ms),
        "total_avg_ms": statistics.mean(samples_ms),
        "total_p50_ms": statistics.median(samples_ms),
        "total_p95_ms": sorted(samples_ms)[max(0, int(len(samples_ms) * 0.95) - 1)],
        "sink": sink,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark writeback + random read total latency on shm"
    )
    parser.add_argument(
        "--n-nodes",
        type=str,
        default="10000,20000,40000,80000",
        help="Comma-separated n_node values to test",
    )
    parser.add_argument("--n-dim", type=int, default=128, help="Feature dim per node")
    parser.add_argument(
        "--n-read",
        type=int,
        default=4096,
        help="Number of random node positions read from shm each iteration",
    )
    parser.add_argument("--repeats", type=int, default=30, help="Timed repeats")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup iterations")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data dtype: float16|float32|float64|int32|int64|uint8",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    if args.n_dim <= 0:
        raise ValueError("n_dim must be > 0")
    if args.n_read <= 0:
        raise ValueError("n_read must be > 0")

    n_nodes = _parse_n_nodes(args.n_nodes)
    dtype = _resolve_dtype(args.dtype)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    # Host process should own and unlink shm at exit.
    set_as_host()

    print("# writeback + random-read from shm benchmark")
    print(
        f"# device={device} dtype={dtype} n_dim={args.n_dim} n_read={args.n_read} repeats={args.repeats} warmup={args.warmup}"
    )
    print(
        "n_node,total_mb,write_avg_ms,read_avg_ms,total_avg_ms,total_p50_ms,total_p95_ms"
    )

    rows = []
    for n_node in n_nodes:
        row = _bench_one_n_node(
            n_node=n_node,
            n_dim=args.n_dim,
            n_read=args.n_read,
            repeats=args.repeats,
            warmup=args.warmup,
            dtype=dtype,
            device=device,
            seed=args.seed,
        )
        rows.append(row)

        print(
            f"{row['n_node']},"
            f"{row['total_mb']:.2f},"
            f"{row['write_avg_ms']:.4f},"
            f"{row['read_avg_ms']:.4f},"
            f"{row['total_avg_ms']:.4f},"
            f"{row['total_p50_ms']:.4f},"
            f"{row['total_p95_ms']:.4f}"
        )

    csv_path = _resolve_csv_path(args.csv)
    if csv_path is not None:
        out_path = csv_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "n_node",
                    "n_dim",
                    "n_read",
                    "dtype",
                    "total_mb",
                    "write_avg_ms",
                    "read_avg_ms",
                    "total_avg_ms",
                    "total_p50_ms",
                    "total_p95_ms",
                ],
            )
            writer.writeheader()
            for r in rows:
                # Exclude sink from CSV summary.
                r2 = {k: v for k, v in r.items() if k != "sink"}
                writer.writerow(r2)
        print(f"# wrote csv: {out_path}")


if __name__ == "__main__":
    main()
