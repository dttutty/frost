"""Shared external GPU telemetry helpers for benchmark runners."""

from __future__ import annotations

import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from frost_bench.metrics import summarize_distribution


Sample = dict[str, float | int]


def start_nvidia_smi_sampler(
    sample_path: Path,
    *,
    gpu_ids: str,
    loop_ms: int = 50,
    cwd: Path | None = None,
) -> tuple[subprocess.Popen[str], TextIO]:
    handle = sample_path.open("w", encoding="utf-8")
    process = subprocess.Popen(  # noqa: S603
        [
            "nvidia-smi",
            "-i",
            gpu_ids,
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
            f"--loop-ms={loop_ms}",
        ],
        cwd=cwd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, handle


def stop_nvidia_smi_sampler(
    process: subprocess.Popen[str], handle: TextIO
) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    handle.close()


def parse_nvidia_smi_samples(sample_path: Path) -> list[Sample]:
    rows: list[Sample] = []
    for line in sample_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("timestamp"):
            continue
        parts = [piece.strip() for piece in stripped.split(",")]
        if len(parts) != 4:
            continue
        timestamp_s, gpu_index_s, gpu_util_s, memory_util_s = parts
        timestamp = datetime.strptime(
            timestamp_s, "%Y/%m/%d %H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc).timestamp()
        gpu_util = float(gpu_util_s)
        memory_util = float(memory_util_s)
        if not (0.0 <= gpu_util <= 100.0 and 0.0 <= memory_util <= 100.0):
            continue
        rows.append(
            {
                "timestamp_unix_s": timestamp,
                "gpu_index": int(gpu_index_s),
                "gpu_util": gpu_util,
                "mem_util": memory_util,
            }
        )
    if not rows:
        raise RuntimeError(f"No nvidia-smi samples found in {sample_path}")
    return rows


def summarize_utilization(samples: list[Sample]) -> dict[str, float]:
    gpu = summarize_distribution([float(row["gpu_util"]) for row in samples])
    memory = summarize_distribution([float(row["mem_util"]) for row in samples])
    return {
        "gpu_util_mean": gpu["mean"],
        "gpu_util_p50": gpu["p50"],
        "gpu_util_p90": gpu["p90"],
        "gpu_util_max": gpu["max"],
        "mem_util_mean": memory["mean"],
        "mem_util_p50": memory["p50"],
        "mem_util_p90": memory["p90"],
        "mem_util_max": memory["max"],
    }
