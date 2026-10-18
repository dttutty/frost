"""Generic serial runner for maintained throughput matrices."""

from __future__ import annotations

import csv
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from frost.data.paths import frost_data_root
from frost.launch.affinity import resolve_common_worker_threads
from frost_bench.adapters.frost import FrostAdapter
from frost_bench.adapters.dyglib import DyGLibAdapter
from frost_bench.adapters.tgl import TGLAdapter
from frost_bench.config import load_dataset_default_macro_batch_size
from frost_bench.io import (
    cleanup_prepared,
    ensure_dir,
    read_text,
    run_command,
    utc_timestamp,
    write_rows_tsv,
)
from frost_bench.metrics import (
    ThroughputByRank,
    mean_events_per_sec,
    parse_frost_rank_throughput,
    parse_dyglib_summary,
    parse_tgl_rank_throughput,
    select_bottleneck_rank,
)
from frost_bench.paths import TGL_ROOT, artifact_root
from frost_bench.spec import BatchSizeScope, ExperimentSpec
from frost_bench.suite import ArgumentValue
from frost_bench.workload import (
    compute_frost_epochs,
    compute_tgl_epochs,
    compute_tgl_max_train_steps,
    train_edge_end,
)


TGL_CONFIGS = {
    model: TGL_ROOT / "config" / "frost_parity" / f"{model.upper()}.yml"
    for model in ("tgn", "jodie", "apan", "tgat")
}


@dataclass(frozen=True)
class ThroughputJob:
    model: str
    dataset: str
    macro_batch_size: int
    dimensions: Mapping[str, object] = field(default_factory=dict)
    frost_overrides: tuple[str, ...] = ()
    frost_env: Mapping[str, str] = field(default_factory=dict)
    compare_tgl: bool = False
    compare_dyglib: bool = False
    tgl_overrides: Mapping[str, object] = field(default_factory=dict)
    tgl_args: tuple[str, ...] = ()
    node_feature_dim: int | None = None
    compile: bool = False

    @property
    def slug(self) -> str:
        parts = [self.model, self.dataset.lower(), f"mb{self.macro_batch_size}"]
        parts.extend(f"{key}{value}" for key, value in self.dimensions.items())
        return "_".join(parts)


def _values(arguments: Mapping[str, ArgumentValue], name: str) -> tuple:
    value = arguments.get(name)
    if isinstance(value, tuple):
        return value
    if value is None:
        return ()
    return (value,)


def _integer(arguments: Mapping[str, ArgumentValue], name: str, default: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, tuple) or value is None:
        raise TypeError(f"{name} must be a scalar integer")
    return int(value)


def _thread_setting(
    arguments: Mapping[str, ArgumentValue], name: str
) -> int | None:
    value = arguments.get(name, "auto")
    if isinstance(value, tuple) or value is None:
        raise TypeError(f"{name} must be 'auto' or a scalar positive integer")
    if str(value).strip().lower() == "auto":
        return None
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be positive")
    return resolved


def _boolean(arguments: Mapping[str, ArgumentValue], name: str, default: bool) -> bool:
    value = arguments.get(name, default)
    if isinstance(value, tuple) or value is None or not isinstance(value, bool):
        raise TypeError(f"{name} must be a scalar boolean")
    return value


def _systems_for_job(job: ThroughputJob, requested: tuple[str, ...]) -> tuple[str, ...]:
    available = (
        ("frost", "tgl")
        if job.compare_tgl
        else (("frost", "dyglib") if job.compare_dyglib else ("frost",))
    )
    if not requested:
        return available
    selected = tuple(system for system in available if system in requested)
    if not selected:
        raise ValueError(
            f"Requested systems {requested} do not apply to {job.model}/{job.dataset}; "
            f"available systems are {available}"
        )
    return selected


def build_throughput_jobs(
    profile: str, arguments: Mapping[str, ArgumentValue]
) -> list[ThroughputJob]:
    """Expand one maintained suite profile into normalized jobs."""
    if profile == "memory_models":
        jobs = []
        models = _values(arguments, "models")
        datasets = _values(arguments, "datasets")
        explicit_batches = _values(arguments, "batch_sizes")
        combinations = (
            product(models, datasets, explicit_batches)
            if explicit_batches
            else (
                (
                    raw_model,
                    dataset,
                    load_dataset_default_macro_batch_size(
                        str(dataset), str(raw_model)
                    ),
                )
                for raw_model, dataset in product(models, datasets)
            )
        )
        for raw_model, dataset, batch in combinations:
            model = str(raw_model)
            tgl_overrides: dict[str, object] = {}
            tgl_args: tuple[str, ...] = ()
            if model == "tgat":
                # Match the maintained FROST TGAT preset instead of TGL's
                # older 10-neighbor, 128-dimensional-time defaults.
                tgl_overrides = {
                    "sampling.0.neighbor": [20, 20],
                    "sampling.0.strategy": "recent",
                    "gnn.0.dim_time": 100,
                }
            jobs.append(
                ThroughputJob(
                    model,
                    str(dataset),
                    int(batch),
                    compare_tgl=True,
                    tgl_overrides=tgl_overrides,
                    tgl_args=tgl_args,
                )
            )
        return jobs
    if profile == "tgn_memdim":
        batch = _integer(arguments, "macro_batch_size", 4000)
        return [
            ThroughputJob(
                "tgn",
                "LASTFM",
                batch,
                dimensions={"mem_dim": int(dim)},
                frost_overrides=(f"model.embedding_dim={int(dim)}",),
                compare_tgl=True,
                tgl_overrides={
                    "memory.0.dim_out": int(dim),
                    "gnn.0.dim_out": int(dim),
                },
            )
            for dim in _values(arguments, "mem_dims")
        ]
    if profile == "tgn_nodefeatdim":
        batch = _integer(arguments, "macro_batch_size", 4000)
        return [
            ThroughputJob(
                "tgn",
                "LASTFM",
                batch,
                dimensions={"node_feat_dim": int(dim)},
                compare_tgl=True,
                node_feature_dim=int(dim),
            )
            for dim in _values(arguments, "node_feat_dims")
        ]
    if profile == "tgn_mem_mail_ablation":
        jobs = []
        for batch, raw_case in product(
            _values(arguments, "macro_batch_sizes"),
            _values(arguments, "ablation_cases"),
        ):
            ablation, carry = str(raw_case).split(":", 1)
            jobs.append(
                ThroughputJob(
                    "tgn",
                    "LASTFM",
                    int(batch),
                    dimensions={"ablation": ablation, "carry": carry},
                    frost_overrides=(
                        f"dynamic_state_ablation={ablation}",
                        f"if_carry_over_route={carry}",
                    ),
                )
            )
        return jobs
    if profile == "tgn_worker_queue":
        batch = _integer(arguments, "macro_batch_size", 4000)
        return [
            ThroughputJob(
                "tgn",
                "LASTFM",
                batch,
                dimensions={"mem_dim": int(dim), "worker_queue": int(queue)},
                frost_overrides=(f"model.embedding_dim={int(dim)}",),
                frost_env={"FROST_MEMORY_WORKER_QUEUE_SIZE": str(queue)},
            )
            for queue, dim in product(
                _values(arguments, "worker_queue_sizes"),
                _values(arguments, "mem_dims"),
            )
        ]
    if profile == "tgn_carry_over_route":
        batch = _integer(arguments, "macro_batch_size", 4000)
        return [
            ThroughputJob(
                "tgn",
                str(dataset),
                batch,
                dimensions={"carry": str(carry).lower()},
                frost_overrides=(f"if_carry_over_route={str(carry).lower()}",),
            )
            for dataset, carry in product(
                _values(arguments, "datasets"),
                _values(arguments, "carry_over_route"),
            )
        ]
    if profile == "large_tgn":
        jobs = []
        disable_features = _boolean(arguments, "disable_features", False)
        profile_stages = _boolean(arguments, "profile_stages", False)
        compile_values = _values(arguments, "compile") or (False,)
        if any(not isinstance(value, bool) for value in compile_values):
            raise TypeError("compile must be a boolean or a list of booleans")
        expose_compile_dimension = "compile" in arguments
        compile_modes = _values(arguments, "compile_mode") or ("default",)
        supported_compile_modes = {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }
        if any(mode not in supported_compile_modes for mode in compile_modes):
            raise ValueError("compile_mode contains an unsupported mode")
        expose_compile_mode_dimension = "compile_mode" in arguments
        attention_layouts = _values(arguments, "attention_layout") or ("dense",)
        if any(
            layout not in {"auto", "dense", "packed"}
            for layout in attention_layouts
        ):
            raise ValueError(
                "attention_layout must contain only 'auto', 'dense', or 'packed'"
            )
        expose_attention_dimension = "attention_layout" in arguments
        dynamic_state_ablations = _values(
            arguments, "dynamic_state_ablation"
        ) or ("baseline",)
        if any(not isinstance(value, str) for value in dynamic_state_ablations):
            raise TypeError("dynamic_state_ablation must contain only strings")
        expose_dynamic_state_dimension = "dynamic_state_ablation" in arguments
        for (
            dataset,
            batch,
            compile_enabled,
            compile_mode,
            attention_layout,
            dynamic_state_ablation,
        ) in product(
            _values(arguments, "datasets"),
            _values(arguments, "batch_sizes"),
            compile_values,
            compile_modes,
            attention_layouts,
            dynamic_state_ablations,
        ):
            jobs.append(
                ThroughputJob(
                    "tgn",
                    str(dataset).upper(),
                    int(batch),
                    dimensions={
                        "full_epoch": _boolean(arguments, "full_epoch", False),
                        "features": "none" if disable_features else "disk",
                        **(
                            {"compile": compile_enabled}
                            if expose_compile_dimension
                            else {}
                        ),
                        **(
                            {"attention_layout": attention_layout}
                            if expose_attention_dimension
                            else {}
                        ),
                        **(
                            {"compile_mode": compile_mode}
                            if expose_compile_mode_dimension
                            else {}
                        ),
                        **(
                            {"dynamic_state_ablation": dynamic_state_ablation}
                            if expose_dynamic_state_dimension
                            else {}
                        ),
                    },
                    frost_overrides=(
                        (
                            "ignore_nfeat_on_disk=true",
                            "ignore_efeat_on_disk=true",
                        )
                        if disable_features
                        else ()
                    )
                    + (
                        (f"model.encoder.attention_layout={attention_layout}",)
                        if expose_attention_dimension
                        else ()
                    )
                    + (
                        (f"compile_mode={compile_mode}",)
                        if expose_compile_mode_dimension
                        else ()
                    )
                    + (
                        (f"dynamic_state_ablation={dynamic_state_ablation}",)
                        if expose_dynamic_state_dimension
                        else ()
                    ),
                    frost_env=(
                        {
                            "FROST_HOST_TIMING_BENCH": "1",
                            "FROST_PIPELINE_TIMING_BENCH": "1",
                        }
                        if profile_stages
                        else {}
                    ),
                    compare_tgl=True,
                    tgl_args=(
                        (
                            "--ignore-node-features-on-disk",
                            "--ignore-edge-features-on-disk",
                        )
                        if disable_features
                        else ()
                    )
                    + (("--stage-timing",) if profile_stages else ()),
                    compile=compile_enabled,
                )
            )
        return jobs
    if profile == "large_dygformer":
        return [
            ThroughputJob(
                "dygformer",
                str(dataset).upper(),
                int(batch),
                dimensions={"full_epoch": _boolean(arguments, "full_epoch", False)},
                compare_dyglib=True,
            )
            for dataset, batch in product(
                _values(arguments, "datasets"),
                _values(arguments, "batch_sizes"),
            )
        ]
    raise ValueError(f"Unknown throughput profile {profile!r}")


def _infer_num_nodes(dataset: str) -> int:
    path = frost_data_root() / dataset / "edges.csv"
    maximum = -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            maximum = max(maximum, int(row["src"]), int(row["dst"]))
    if maximum < 0:
        raise RuntimeError(f"Could not infer node count from {path}")
    return maximum + 1


@contextmanager
def _temporary_node_features(
    dataset: str, feature_dim: int | None, seed: int
) -> Iterator[None]:
    if feature_dim is None:
        yield
        return
    path = frost_data_root() / dataset / "node_features.npy"
    existed = path.exists() or path.is_symlink()
    symlink_target = os.readlink(path) if path.is_symlink() else None
    original = path.read_bytes() if existed and not path.is_symlink() else None
    try:
        if existed:
            path.unlink()
        if feature_dim > 0:
            values = np.random.default_rng(seed).standard_normal(
                (_infer_num_nodes(dataset), feature_dim)
            )
            np.save(path, values.astype(np.float32), allow_pickle=False)
        yield
    finally:
        if path.exists() or path.is_symlink():
            path.unlink()
        if symlink_target is not None:
            path.symlink_to(symlink_target)
        elif original is not None:
            path.write_bytes(original)


def _normalized_row(
    *,
    job: ThroughputJob,
    system: str,
    log_path: Path,
    n_gpu: int,
    measured: ThroughputByRank,
    train_events: int,
) -> dict[str, object]:
    rank, throughput, batches = select_bottleneck_rank(measured)
    return {
        **job.dimensions,
        "model": job.model,
        "dataset": job.dataset,
        "macro_batch_size": job.macro_batch_size,
        "system": system,
        "status": "ok",
        "n_gpu": n_gpu,
        "bottleneck_rank": rank,
        "events_per_sec_per_gpu": round(throughput, 6),
        "events_per_sec_total": round(throughput * n_gpu, 6),
        "mean_events_per_sec_per_gpu": round(mean_events_per_sec(measured), 6),
        "measured_batches": batches,
        "train_events": train_events,
        "estimated_train_time_s": round(train_events / (throughput * n_gpu), 6),
        "log": str(log_path),
    }


def _write_report(path: Path, suite_name: str, rows: list[dict[str, object]]) -> None:
    successes = sum(row.get("status") == "ok" for row in rows)
    failures = len(rows) - successes
    path.write_text(
        "\n".join(
            [
                f"# Throughput suite: {suite_name}",
                "",
                f"- Runs: {len(rows)}",
                f"- Successful: {successes}",
                f"- Failed: {failures}",
                "- Raw and normalized data: `logs/`, `results.tsv`, `summary.tsv`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_throughput_suite(
    *,
    suite_name: str,
    profile: str,
    arguments: Mapping[str, ArgumentValue],
) -> Path:
    jobs = build_throughput_jobs(profile, arguments)
    gpu_ids = str(arguments.get("gpu_ids", "0,1"))
    n_gpu = len([value for value in gpu_ids.split(",") if value.strip()])
    if n_gpu < 1:
        raise ValueError("gpu_ids must name at least one GPU")
    warmup = _integer(arguments, "warmup_batches", 5)
    measure = _integer(arguments, "measure_batches", 100)
    omp = _thread_setting(arguments, "omp_threads")
    mkl = _thread_setting(arguments, "mkl_threads")
    seed = _integer(arguments, "seed", 0)
    port_base = _integer(arguments, "master_port_base", 35600)
    run_name = str(arguments.get("run_name") or utc_timestamp())
    configured_root = arguments.get("output_root")
    output_root = (
        Path(str(configured_root)).expanduser()
        if configured_root
        else artifact_root() / "throughput" / suite_name
    )
    run_dir = output_root / run_name
    log_dir = run_dir / "logs"
    config_dir = run_dir / "configs"
    ensure_dir(log_dir)
    ensure_dir(config_dir)
    results: list[dict[str, object]] = []
    frost_adapter = FrostAdapter()
    tgl_adapter = TGLAdapter()
    dyglib_adapter = DyGLibAdapter()
    full_epoch = _boolean(arguments, "full_epoch", False)
    epochs = _integer(arguments, "epochs", 1)
    measure_start_epoch = _integer(arguments, "measure_start_epoch", 0)
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if measure_start_epoch >= epochs:
        raise ValueError("measure_start_epoch must be smaller than epochs")
    stable_mode = _boolean(arguments, "stable_mode", True)
    single_gpu_resident = _boolean(arguments, "single_gpu_resident", False)
    if single_gpu_resident and n_gpu != 1:
        raise ValueError("single_gpu_resident=true requires exactly one GPU")
    auto_threads = resolve_common_worker_threads(
        n_gpu=n_gpu,
        cuda_visible_devices=gpu_ids,
    )
    single_gpu_auto_threads = resolve_common_worker_threads(
        n_gpu=1,
        cuda_visible_devices=gpu_ids.split(",", 1)[0],
    )
    requested_systems = tuple(
        str(system).lower() for system in _values(arguments, "systems")
    )
    unknown_systems = set(requested_systems).difference({"frost", "tgl", "dyglib"})
    if unknown_systems:
        raise ValueError(f"Unknown systems requested: {sorted(unknown_systems)}")
    total_runs = sum(len(_systems_for_job(job, requested_systems)) for job in jobs)

    for job_index, job in enumerate(jobs):
        if job.macro_batch_size % n_gpu:
            raise ValueError(
                f"macro_batch_size={job.macro_batch_size} is not divisible by {n_gpu} GPUs"
            )
        systems = _systems_for_job(job, requested_systems)
        train_events = train_edge_end(job.dataset)
        for system_index, system in enumerate(systems):
            log_path = log_dir / f"{system}_{job.slug}.log"
            base_row: dict[str, object] = {
                **job.dimensions,
                "model": job.model,
                "dataset": job.dataset,
                "macro_batch_size": job.macro_batch_size,
                "system": system,
                "status": "failed",
                "log": str(log_path),
            }
            try:
                frost_epochs = (
                    epochs
                    if full_epoch
                    else compute_frost_epochs(
                        job.dataset, job.macro_batch_size, warmup, measure
                    )
                )
                port = port_base + job_index * 2 + system_index
                with _temporary_node_features(
                    job.dataset,
                    job.node_feature_dim,
                    seed + job_index,
                ):
                    if system == "frost":
                        prepared = frost_adapter.prepare(
                            ExperimentSpec(
                                model=job.model,
                                dataset=job.dataset,
                                seed=seed,
                                n_gpus=n_gpu,
                                batch_size=job.macro_batch_size,
                                epochs=frost_epochs,
                                compile=job.compile,
                                single_gpu_resident=single_gpu_resident,
                                extra_args=(
                                    "tqdm=true",
                                    "+benchmark.train_only=true",
                                    *job.frost_overrides,
                                ),
                            ),
                            master_port=port,
                            warmup_batches=warmup,
                            measure_start_epoch=measure_start_epoch,
                            measure_batches=0 if full_epoch else measure,
                            omp_threads=omp,
                            mkl_threads=mkl,
                            gpu_ids=gpu_ids,
                            stable_mode=stable_mode,
                        )
                        environment = {**prepared.env, **job.frost_env}
                        if not full_epoch:
                            environment["FROST_DEBUG_MAX_BATCHES_PER_SPLIT"] = str(
                                warmup + measure + 2
                            )
                        run_n_gpu = n_gpu
                    elif system == "tgl":
                        per_gpu = job.macro_batch_size // n_gpu
                        max_steps = (
                            0
                            if full_epoch
                            else compute_tgl_max_train_steps(
                                job.dataset, per_gpu, n_gpu, warmup, measure
                            )
                        )
                        tgl_epochs = (
                            epochs
                            if full_epoch
                            else compute_tgl_epochs(
                                job.dataset, per_gpu, n_gpu, max_steps
                            )
                        )
                        source = TGL_CONFIGS[job.model]
                        config_path = config_dir / f"{job.slug}.yml"
                        tgl_adapter.write_config(
                            source=source,
                            destination=config_path,
                            epochs=tgl_epochs,
                            per_gpu_batch_size=per_gpu,
                            overrides=job.tgl_overrides,
                        )
                        prepared = tgl_adapter.prepare(
                            ExperimentSpec(
                                model=job.model,
                                dataset=job.dataset,
                                seed=seed,
                                n_gpus=n_gpu,
                                batch_size=per_gpu,
                                batch_size_scope=BatchSizeScope.PER_GPU,
                                epochs=tgl_epochs,
                                single_gpu_resident=single_gpu_resident,
                                extra_args=("--tqdm", *job.tgl_args),
                            ),
                            config_path=config_path,
                            max_train_steps=max_steps,
                            warmup_batches=warmup,
                            measure_start_epoch=measure_start_epoch,
                            measure_batches=0 if full_epoch else measure,
                            omp_threads=omp or auto_threads,
                            mkl_threads=mkl or auto_threads,
                            gpu_ids=gpu_ids,
                        )
                        environment = dict(prepared.env)
                        run_n_gpu = n_gpu
                    else:
                        prepared = dyglib_adapter.prepare(
                            ExperimentSpec(
                                model=job.model,
                                dataset=job.dataset,
                                seed=seed,
                                n_gpus=1,
                                batch_size=job.macro_batch_size,
                                epochs=1,
                            ),
                            max_train_batches=(
                                0 if full_epoch else warmup + measure + 2
                            ),
                            warmup_batches=warmup,
                            measure_batches=0 if full_epoch else measure,
                            omp_threads=omp or single_gpu_auto_threads,
                            mkl_threads=mkl or single_gpu_auto_threads,
                            gpu_ids=gpu_ids.split(",", 1)[0],
                        )
                        environment = dict(prepared.env)
                        run_n_gpu = 1
                    wall_start = time.perf_counter()
                    try:
                        run_command(
                            prepared.command, environment, log_path, prepared.cwd
                        )
                    finally:
                        cleanup_prepared(prepared)
                    wall_time_s = time.perf_counter() - wall_start

                text = read_text(log_path)
                if system == "frost":
                    measured = parse_frost_rank_throughput(text)
                elif system == "tgl":
                    measured = parse_tgl_rank_throughput(text, n_gpu=n_gpu)
                else:
                    measured = {0: parse_dyglib_summary(text)}
                base_row = _normalized_row(
                    job=job,
                    system=system,
                    log_path=log_path,
                    n_gpu=run_n_gpu,
                    measured=measured,
                    train_events=train_events,
                )
                base_row["wall_time_s"] = round(wall_time_s, 6)
            except Exception as exc:
                base_row["error"] = str(exc)
                base_row["traceback"] = traceback.format_exc()
            results.append(base_row)
            write_rows_tsv(run_dir / "results.tsv", results)
            print(
                f"[{len(results)}/{total_runs}] "
                f"{system} {job.slug}: {base_row['status']}"
            )

    write_rows_tsv(run_dir / "summary.tsv", results)
    _write_report(run_dir / "report.md", suite_name, results)
    return run_dir
