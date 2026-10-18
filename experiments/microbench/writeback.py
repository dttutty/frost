#!/usr/bin/env python3
"""Benchmark GPU-to-shared-memory write latency vs payload size.

This script uses ``create_shared_mem_array`` to allocate CPU shared-memory
buffers, then measures the time to copy data from a CUDA tensor into that
shared-memory tensor. On most systems this transfer goes over PCIe.
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
    """Allow running this file directly from the repository tree."""
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
        return _script_dir() / "writeback.csv"
    return Path(raw).expanduser().resolve()


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


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _make_gpu_source(
    num_bytes: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    elem_size = _dtype_nbytes(dtype)
    n_elem = num_bytes // elem_size
    if n_elem <= 0:
        raise ValueError("Payload too small for selected dtype")
    # Deterministic content avoids measuring random generation overhead repeatedly.
    src = torch.arange(n_elem, device=device, dtype=dtype)
    return src


def _bench_one_size(
    size_bytes: int,
    repeats: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    elem_size = _dtype_nbytes(dtype)
    n_elem = size_bytes // elem_size
    if n_elem <= 0:
        raise ValueError("size_bytes too small")

    shm_name = f"pcie_shm_{size_bytes}"
    shm_tensor = create_shared_mem_array(shm_name, shape=(n_elem,), dtype=dtype)
    src = _make_gpu_source(size_bytes, dtype=dtype, device=device)

    for _ in range(warmup):
        shm_tensor.copy_(src, non_blocking=False)
        torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        shm_tensor.copy_(src, non_blocking=False)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    avg_ms = statistics.mean(samples_ms)
    p50_ms = statistics.median(samples_ms)
    p95_ms = sorted(samples_ms)[max(0, int(len(samples_ms) * 0.95) - 1)]
    gbps = (size_bytes / (1024**3)) / (avg_ms / 1000.0)

    return {
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024**2),
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "gbps": gbps,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU->shared-memory write time vs data size"
    )
    parser.add_argument(
        "--sizes-mb",
        type=str,
        default="1,2,4,8,16,32,64,128,256",
        help="Comma-separated payload sizes in MB",
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

    dtype = _resolve_dtype(args.dtype)
    sizes = _parse_sizes_mb(args.sizes_mb)
    device = torch.device(f"cuda:{args.device}")

    # Host process should own and unlink shm segments at exit.
    set_as_host()
    torch.cuda.set_device(device)

    print("# GPU -> Shared Memory (create_shared_mem_array) benchmark")
    print(
        f"# device={device} dtype={dtype} repeats={args.repeats} warmup={args.warmup}"
    )
    print("size_mb,avg_ms,p50_ms,p95_ms,gbps")

    rows = []
    for size_bytes in sizes:
        row = _bench_one_size(
            size_bytes=size_bytes,
            repeats=args.repeats,
            warmup=args.warmup,
            dtype=dtype,
            device=device,
        )
        rows.append(row)
        print(
            f"{row['size_mb']:.2f},"
            f"{row['avg_ms']:.4f},"
            f"{row['p50_ms']:.4f},"
            f"{row['p95_ms']:.4f},"
            f"{row['gbps']:.3f}"
        )

    csv_path = _resolve_csv_path(args.csv)
    if csv_path is not None:
        out_path = csv_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "size_bytes",
                    "size_mb",
                    "avg_ms",
                    "p50_ms",
                    "p95_ms",
                    "gbps",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"# wrote csv: {out_path}")


if __name__ == "__main__":
    main()
