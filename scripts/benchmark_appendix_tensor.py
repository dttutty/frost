#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from frost_bench.adapters.tgl import write_tgl_config
from frost_bench.io import read_text, run_command as run_logged_command
from frost_bench.config import PRESETS

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_RE = re.compile(r"^APPENDIX_TRAIN_E2E_SUMMARY\s+(\{.*\})$", re.MULTILINE)
THROUGHPUT_RE = re.compile(
    r"^TRAIN_THROUGHPUT_PER_GPU\s+"
    r"rank=(?P<rank>\d+)\s+"
    r"events_per_sec_per_gpu=(?P<events_per_sec>[0-9.]+).*?"
    r"elapsed_s=(?P<elapsed_s>[0-9.]+).*?"
    r"measured_batches=(?P<measured_batches>\d+)",
    re.MULTILINE,
)
DENSE_ROWS_RE = re.compile(
    r"^APPENDIX_DENSE_VS_SPARSE_ROWS\s+(\[.*\])$",
    re.MULTILINE,
)


FROST_PRESETS = PRESETS
FROST_PRESET_NAMES = tuple(sorted(FROST_PRESETS))

TGL_CONFIG_ROOT = REPO_ROOT / "third_party" / "tgl" / "config" / "frost_parity"
TGL_CONFIGS = {
    "tgn": TGL_CONFIG_ROOT / "TGN.yml",
    "jodie": TGL_CONFIG_ROOT / "JODIE.yml",
    "apan": TGL_CONFIG_ROOT / "APAN.yml",
    "tgat": TGL_CONFIG_ROOT / "TGAT.yml",
}


@dataclass(frozen=True)
class DenseVsSparseRow:
    n: int
    s: int
    dgl_sparse_ms: float
    dgl_sparse_std_ms: float
    dense_eager_ms: float
    dense_eager_std_ms: float
    speedup: float


@dataclass(frozen=True)
class TrainRunResult:
    system: str
    preset: str
    compile_enabled: bool
    status: str
    log_path: Path
    summaries: list[dict[str, Any]]
    avg_wall_ms: float | None = None
    std_wall_ms: float | None = None
    avg_gpu_ms: float | None = None
    num_ranks: int = 0
    returncode: int = 0


@dataclass(frozen=True)
class CompileUpliftRow:
    runtime: str
    preset: str
    eager_ms: float | None
    eager_std_ms: float | None
    compiled_ms: float | None
    compiled_std_ms: float | None
    uplift: float | None
    status: str


def run_command(
    *,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    allow_failure: bool,
    stage_name: str,
) -> tuple[int, str]:
    """Run one benchmark case and return its exit code and captured log."""
    print(f"[{stage_name}] starting", flush=True)
    try:
        run_logged_command(command, env, log_path, REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        returncode = exc.returncode
        if not allow_failure:
            raise
    else:
        returncode = 0
    text = read_text(log_path)
    print(f"[{stage_name}] exit={returncode}", flush=True)
    return returncode, text


def parse_int_list(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_preset_names(raw: str) -> list[str]:
    if raw == "all":
        return list(FROST_PRESET_NAMES)
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one preset name")
    invalid = [value for value in values if value not in FROST_PRESET_NAMES]
    if invalid:
        raise ValueError(
            "unknown preset(s): "
            + ", ".join(invalid)
            + f". Available presets: {', '.join(FROST_PRESET_NAMES)}"
        )
    return values


def parse_summary_lines(text: str) -> list[dict[str, Any]]:
    summaries = [json.loads(match.group(1)) for match in SUMMARY_RE.finditer(text)]
    if summaries:
        return summaries

    # Current FROST telemetry emits an aggregate throughput line instead of the
    # legacy appendix summary. Derive the same mean minibatch latency so older
    # benchmark reports remain usable across the telemetry refactor.
    rows: list[dict[str, Any]] = []
    for match in THROUGHPUT_RE.finditer(text):
        measured_batches = int(match.group("measured_batches"))
        elapsed_s = float(match.group("elapsed_s"))
        if measured_batches == 0:
            continue
        rows.append(
            {
                "system": "frost",
                "rank": int(match.group("rank")),
                "avg_wall_ms": elapsed_s * 1000.0 / measured_batches,
                "events_per_sec_per_gpu": float(match.group("events_per_sec")),
                "measured_batches": measured_batches,
            }
        )
    return rows


def parse_dense_rows(text: str) -> list[dict[str, Any]]:
    match = DENSE_ROWS_RE.search(text)
    if match is None:
        return []
    return json.loads(match.group(1))


def aggregate_train_summaries(
    *,
    system: str,
    preset: str,
    compile_enabled: bool,
    log_path: Path,
    summaries: list[dict[str, Any]],
    returncode: int,
) -> TrainRunResult:
    valid = [row for row in summaries if "avg_wall_ms" in row]
    if returncode != 0:
        return TrainRunResult(
            system=system,
            preset=preset,
            compile_enabled=compile_enabled,
            status="failed",
            log_path=log_path,
            summaries=summaries,
            returncode=returncode,
        )
    if not valid:
        return TrainRunResult(
            system=system,
            preset=preset,
            compile_enabled=compile_enabled,
            status="missing_summary",
            log_path=log_path,
            summaries=summaries,
            returncode=returncode,
        )
    avg_wall_ms = sum(float(row["avg_wall_ms"]) for row in valid) / len(valid)
    std_values = [float(row["std_wall_ms"]) for row in valid if "std_wall_ms" in row]
    std_wall_ms = sum(std_values) / len(std_values) if std_values else None
    avg_gpu_ms = (
        sum(float(row["avg_gpu_ms"]) for row in valid if "avg_gpu_ms" in row) / len(valid)
        if any("avg_gpu_ms" in row for row in valid)
        else None
    )
    return TrainRunResult(
        system=system,
        preset=preset,
        compile_enabled=compile_enabled,
        status="ok",
        log_path=log_path,
        summaries=summaries,
        avg_wall_ms=avg_wall_ms,
        std_wall_ms=std_wall_ms,
        avg_gpu_ms=avg_gpu_ms,
        num_ranks=len(valid),
        returncode=returncode,
    )


def build_frost_train_command(
    *,
    preset: str,
    dataset: str,
    num_gpus: int,
    epochs: int,
    compile_enabled: bool,
    macro_batch_size: int | None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "-m", "frost_bench.cli.train",
        preset,
        f"--nproc-per-node={num_gpus + 1}",
        f"dataset={dataset}",
        f"epoch={epochs}",
        f"compile={'true' if compile_enabled else 'false'}",
        "tqdm=false",
    ]
    if macro_batch_size is not None:
        command.append(f"macro_batch_size={macro_batch_size}")
    return command


def write_tgl_config_copy(
    *,
    preset: str,
    epochs: int,
    output_path: Path,
) -> Path:
    return write_tgl_config(TGL_CONFIGS[preset], output_path, epochs=epochs)


def build_tgl_train_command(
    *,
    dataset: str,
    config_path: Path,
    num_gpus: int,
    compile_enabled: bool,
    measure_start_epoch: int,
    warmup_batches: int,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--project",
        "third_party/tgl",
        "python",
        "third_party/tgl/train_dist.py",
        "--dataset",
        dataset,
        "--config",
        str(config_path),
        "--num_gpus",
        str(num_gpus),
        "--appendix-train-e2e",
        "--measure-start-epoch",
        str(measure_start_epoch),
        "--warmup-batches",
        str(warmup_batches),
    ]
    if compile_enabled:
        command.append("--compile")
    return command


def get_visible_cuda_device_count() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def run_frost_train_case(
    *,
    preset: str,
    dataset: str,
    num_gpus: int,
    epochs: int,
    compile_enabled: bool,
    macro_batch_size: int | None,
    measure_start_epoch: int,
    warmup_batches: int,
    log_dir: Path,
    allow_failure: bool,
) -> TrainRunResult:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FROST_APPENDIX_TRAIN_E2E"] = "1"
    env["FROST_APPENDIX_MEASURE_START_EPOCH"] = str(measure_start_epoch)
    env["FROST_APPENDIX_WARMUP_BATCHES"] = str(warmup_batches)
    env["FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH"] = "0"
    env["FROST_THROUGHPUT_WARMUP_BATCHES_TOTAL"] = str(warmup_batches)
    command = build_frost_train_command(
        preset=preset,
        dataset=dataset,
        num_gpus=num_gpus,
        epochs=epochs,
        compile_enabled=compile_enabled,
        macro_batch_size=macro_batch_size,
    )
    log_path = log_dir / (
        f"frost_{preset}_{dataset}_ngpu{num_gpus}_ep{epochs}_"
        f"{'compile' if compile_enabled else 'eager'}.log"
    )
    returncode, text = run_command(
        command=command,
        env=env,
        log_path=log_path,
        allow_failure=allow_failure,
        stage_name=(
            f"FROST {preset} {dataset} "
            f"{'compile' if compile_enabled else 'eager'}"
        ),
    )
    return aggregate_train_summaries(
        system="frost",
        preset=preset,
        compile_enabled=compile_enabled,
        log_path=log_path,
        summaries=parse_summary_lines(text),
        returncode=returncode,
    )


def run_tgl_train_case(
    *,
    preset: str,
    dataset: str,
    num_gpus: int,
    epochs: int,
    compile_enabled: bool,
    measure_start_epoch: int,
    warmup_batches: int,
    config_dir: Path,
    log_dir: Path,
    allow_failure: bool,
) -> TrainRunResult:
    config_path = write_tgl_config_copy(
        preset=preset,
        epochs=epochs,
        output_path=config_dir
        / f"tgl_{preset}_{dataset}_ngpu{num_gpus}_ep{epochs}.yml",
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = build_tgl_train_command(
        dataset=dataset,
        config_path=config_path,
        num_gpus=num_gpus,
        compile_enabled=compile_enabled,
        measure_start_epoch=measure_start_epoch,
        warmup_batches=warmup_batches,
    )
    log_path = log_dir / (
        f"tgl_{preset}_{dataset}_ngpu{num_gpus}_ep{epochs}_"
        f"{'compile' if compile_enabled else 'eager'}.log"
    )
    returncode, text = run_command(
        command=command,
        env=env,
        log_path=log_path,
        allow_failure=allow_failure,
        stage_name=(
            f"TGL {preset} {dataset} "
            f"{'compile' if compile_enabled else 'eager'}"
        ),
    )
    return aggregate_train_summaries(
        system="tgl",
        preset=preset,
        compile_enabled=compile_enabled,
        log_path=log_path,
        summaries=parse_summary_lines(text),
        returncode=returncode,
    )


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def format_dense_vs_sparse_table(rows: list[DenseVsSparseRow]) -> str:
    lines = [
        "| N | S | DGLGraph (ms) | FROST HierGraph (ms) | Speedup |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.n} | "
            f"{row.s} | "
            f"{format_mean_std(row.dgl_sparse_ms, row.dgl_sparse_std_ms)} | "
            f"{format_mean_std(row.dense_eager_ms, row.dense_eager_std_ms)} | "
            f"{row.speedup:.2f}x |"
        )
    return "\n".join(lines)


def format_compile_uplift_table(rows: list[CompileUpliftRow]) -> str:
    lines = [
        "| Runtime | Workload | Eager Train Minibatch (ms) | Compiled Train Minibatch (ms) | Uplift | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        uplift = f"{row.uplift:.2f}x" if row.uplift is not None else "N/A"
        lines.append(
            "| "
            f"{row.runtime} | "
            f"{row.preset} | "
            f"{format_mean_std(row.eager_ms, row.eager_std_ms)} | "
            f"{format_mean_std(row.compiled_ms, row.compiled_std_ms)} | "
            f"{uplift} | "
            f"{row.status} |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_cuda(num_gpus: int) -> None:
    visible = get_visible_cuda_device_count()
    if visible == 0:
        raise RuntimeError("CUDA is required for this benchmark script.")
    if visible < num_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPU(s), but only {visible} visible."
        )


def resolve_effective_num_gpus(requested_num_gpus: int) -> tuple[int, int]:
    visible = get_visible_cuda_device_count()
    if visible == 0:
        raise RuntimeError("CUDA is required for this benchmark script.")
    return min(requested_num_gpus, visible), visible


def bench_step_ms(step_fn, *, warmup_iters: int, measure_iters: int) -> tuple[float, float]:
    import torch

    for _ in range(warmup_iters):
        step_fn()
    torch.cuda.synchronize()

    pairs = []
    for _ in range(measure_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        pairs.append((start, end))

    torch.cuda.synchronize()
    timings = [start.elapsed_time(end) for start, end in pairs]
    mean = sum(timings) / len(timings)
    variance = (
        sum((value - mean) ** 2 for value in timings) / len(timings)
        if len(timings) > 1
        else 0.0
    )
    return mean, variance**0.5


def run_dense_vs_sparse(
    *,
    n_values: list[int],
    s_values: list[int],
    dim_node: int,
    dim_edge: int,
    dim_time: int,
    dim_out: int,
    heads: int,
    dropout: float,
    warmup_iters: int,
    measure_iters: int,
    log_dir: Path,
) -> list[DenseVsSparseRow]:
    command = [
        "uv",
        "run",
        "--project",
        "third_party/tgl",
        "python",
        "third_party/tgl/benchmark_appendix_tensor_helper.py",
        "--n-values",
        ",".join(str(value) for value in n_values),
        "--s-values",
        ",".join(str(value) for value in s_values),
        "--dim-node",
        str(dim_node),
        "--dim-edge",
        str(dim_edge),
        "--dim-time",
        str(dim_time),
        "--dim-out",
        str(dim_out),
        "--heads",
        str(heads),
        "--dropout",
        str(dropout),
        "--warmup-iters",
        str(warmup_iters),
        "--measure-iters",
        str(measure_iters),
    ]
    log_path = log_dir / "dense_vs_sparse_helper.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    returncode, text = run_command(
        command=command,
        env=env,
        log_path=log_path,
        allow_failure=False,
        stage_name="DGLGraph vs FROST HierGraph microbenchmark",
    )
    if returncode != 0:
        raise RuntimeError(f"dense-vs-sparse helper failed; see {log_path}")

    parsed_rows = parse_dense_rows(text)
    if not parsed_rows:
        raise RuntimeError(
            f"dense-vs-sparse helper did not emit APPENDIX_DENSE_VS_SPARSE_ROWS; see {log_path}"
        )
    return [DenseVsSparseRow(**row) for row in parsed_rows]


def combine_compile_results(
    *,
    preset: str,
    eager: TrainRunResult,
    compiled: TrainRunResult,
) -> CompileUpliftRow:
    runtime = "FROST HierGraph" if eager.system == "frost" else "TGL DGLGraph"
    if eager.status != "ok":
        return CompileUpliftRow(
            runtime=runtime,
            preset=preset,
            eager_ms=None,
            eager_std_ms=None,
            compiled_ms=None,
            compiled_std_ms=None,
            uplift=None,
            status=f"eager_{eager.status}",
        )
    if compiled.status != "ok":
        return CompileUpliftRow(
            runtime=runtime,
            preset=preset,
            eager_ms=eager.avg_wall_ms,
            eager_std_ms=eager.std_wall_ms,
            compiled_ms=None,
            compiled_std_ms=None,
            uplift=None,
            status=f"compiled_{compiled.status}",
        )
    assert eager.avg_wall_ms is not None
    assert compiled.avg_wall_ms is not None
    uplift = eager.avg_wall_ms / compiled.avg_wall_ms
    return CompileUpliftRow(
        runtime=runtime,
        preset=preset,
        eager_ms=eager.avg_wall_ms,
        eager_std_ms=eager.std_wall_ms,
        compiled_ms=compiled.avg_wall_ms,
        compiled_std_ms=compiled.std_wall_ms,
        uplift=uplift,
        status="ok",
    )


def run_compile_uplift(
    *,
    presets: list[str],
    dataset: str,
    num_gpus: int,
    epochs: int,
    measure_start_epoch: int,
    warmup_batches: int,
    output_dir: Path,
    include_tgl: bool,
    macro_batch_size: int | None,
) -> tuple[list[CompileUpliftRow], list[TrainRunResult]]:
    config_dir = output_dir / "configs"
    log_dir = output_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[CompileUpliftRow] = []
    results: list[TrainRunResult] = []

    for preset in presets:
        frost_eager = run_frost_train_case(
            preset=preset,
            dataset=dataset,
            num_gpus=num_gpus,
            epochs=epochs,
            compile_enabled=False,
            macro_batch_size=macro_batch_size,
            measure_start_epoch=measure_start_epoch,
            warmup_batches=warmup_batches,
            log_dir=log_dir,
            allow_failure=True,
        )
        results.append(frost_eager)
        frost_compiled = run_frost_train_case(
            preset=preset,
            dataset=dataset,
            num_gpus=num_gpus,
            epochs=epochs,
            compile_enabled=True,
            macro_batch_size=macro_batch_size,
            measure_start_epoch=measure_start_epoch,
            warmup_batches=warmup_batches,
            log_dir=log_dir,
            allow_failure=True,
        )
        results.append(frost_compiled)
        rows.append(
            combine_compile_results(
                preset=preset,
                eager=frost_eager,
                compiled=frost_compiled,
            )
        )

        if include_tgl:
            if preset not in TGL_CONFIGS:
                rows.append(
                    CompileUpliftRow(
                        runtime="TGL DGLGraph",
                        preset=preset,
                        eager_ms=None,
                        eager_std_ms=None,
                        compiled_ms=None,
                        compiled_std_ms=None,
                        uplift=None,
                        status="unsupported_preset",
                    )
                )
                continue
            tgl_eager = run_tgl_train_case(
                preset=preset,
                dataset=dataset,
                num_gpus=num_gpus,
                epochs=epochs,
                compile_enabled=False,
                measure_start_epoch=measure_start_epoch,
                warmup_batches=warmup_batches,
                config_dir=config_dir,
                log_dir=log_dir,
                allow_failure=True,
            )
            results.append(tgl_eager)
            tgl_compiled = run_tgl_train_case(
                preset=preset,
                dataset=dataset,
                num_gpus=num_gpus,
                epochs=epochs,
                compile_enabled=True,
                measure_start_epoch=measure_start_epoch,
                warmup_batches=warmup_batches,
                config_dir=config_dir,
                log_dir=log_dir,
                allow_failure=True,
            )
            results.append(tgl_compiled)
            rows.append(
                combine_compile_results(
                    preset=preset,
                    eager=tgl_eager,
                    compiled=tgl_compiled,
                )
            )
    return rows, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run appendix tensor benchmarks and emit paste-ready tables."
    )
    parser.add_argument(
        "--mode",
        choices=("dense-vs-sparse", "compile-uplift", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "appendix_tensor",
    )
    parser.add_argument(
        "--train-preset",
        default="tgn",
        help="Preset name, comma-separated preset list, or 'all' to use every FROST preset.",
    )
    parser.add_argument("--dataset", default="LASTFM")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--measure-start-epoch", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument(
        "--macro-batch-size",
        type=int,
        default=None,
        help="Optional FROST macro_batch_size override applied to every preset.",
    )
    parser.add_argument(
        "--compile-runtimes",
        choices=("frost", "all"),
        default="frost",
        help="Which e2e training runtimes to include in the compile-uplift table.",
    )
    parser.add_argument("--n-values", default="200,600,1000")
    parser.add_argument("--s-values", default="10,20,30")
    parser.add_argument("--micro-warmup-iters", type=int, default=50)
    parser.add_argument("--micro-measure-iters", type=int, default=100)
    parser.add_argument("--dim-node", type=int, default=172)
    parser.add_argument("--dim-edge", type=int, default=172)
    parser.add_argument("--dim-time", type=int, default=100)
    parser.add_argument("--dim-out", type=int, default=100)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.measure_start_epoch < 0:
        parser.error("--measure-start-epoch must be >= 0")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be >= 0")
    if args.num_gpus < 1:
        parser.error("--num-gpus must be >= 1")
    if args.macro_batch_size is not None and args.macro_batch_size < 1:
        parser.error("--macro-batch-size must be >= 1")

    try:
        train_presets = parse_preset_names(args.train_preset)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ran_any = False

    if args.mode in {"dense-vs-sparse", "all"}:
        ensure_cuda(1)
        dense_rows = run_dense_vs_sparse(
            n_values=parse_int_list(args.n_values),
            s_values=parse_int_list(args.s_values),
            dim_node=args.dim_node,
            dim_edge=args.dim_edge,
            dim_time=args.dim_time,
            dim_out=args.dim_out,
            heads=args.heads,
            dropout=args.dropout,
            warmup_iters=args.micro_warmup_iters,
            measure_iters=args.micro_measure_iters,
            log_dir=log_dir,
        )
        dense_table = format_dense_vs_sparse_table(dense_rows)
        (output_dir / "dense_vs_sparse.md").write_text(dense_table + "\n")
        write_csv(
            output_dir / "dense_vs_sparse.csv",
            [asdict(row) for row in dense_rows],
        )
        print("DGLGraph vs. FROST HierGraph")
        print(dense_table)
        print()
        ran_any = True

    if args.mode in {"compile-uplift", "all"}:
        effective_num_gpus, _ = resolve_effective_num_gpus(args.num_gpus)
        if effective_num_gpus < args.num_gpus:
            print(
                "Compiler uplift will use "
                f"{effective_num_gpus} visible GPU(s) instead of the requested "
                f"{args.num_gpus}.",
                file=sys.stderr,
            )
        compile_rows, raw_results = run_compile_uplift(
            presets=train_presets,
            dataset=args.dataset,
            num_gpus=effective_num_gpus,
            epochs=args.epochs,
            measure_start_epoch=args.measure_start_epoch,
            warmup_batches=args.warmup_batches,
            output_dir=output_dir,
            include_tgl=args.compile_runtimes == "all",
            macro_batch_size=args.macro_batch_size,
        )
        compile_table = format_compile_uplift_table(compile_rows)
        (output_dir / "compile_uplift.md").write_text(compile_table + "\n")
        write_csv(
            output_dir / "compile_uplift.csv",
            [asdict(row) for row in compile_rows],
        )
        (output_dir / "compile_uplift_raw.json").write_text(
            json.dumps(
                [
                    {
                        **asdict(result),
                        "log_path": str(result.log_path),
                    }
                    for result in raw_results
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print("Compiler Uplift: Eager vs. torch.compile")
        print(compile_table)
        print()
        ran_any = True

    if not ran_any:
        parser.error("no benchmark mode selected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
