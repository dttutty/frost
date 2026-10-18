from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from frost_bench.paths import REPO_ROOT
from frost_bench.spec import BatchSizeScope, ExperimentSpec, PreparedRun


class FrostAdapter:
    name = "frost"

    def prepare(
        self,
        spec: ExperimentSpec,
        *,
        master_port: int,
        warmup_batches: int | None = None,
        measure_start_epoch: int = 0,
        measure_batches: int | None = None,
        omp_threads: int | None = None,
        mkl_threads: int | None = None,
        gpu_ids: str | None = None,
        stable_mode: bool = True,
        epoch_wall: bool = False,
    ) -> PreparedRun:
        if spec.batch_size_scope is not BatchSizeScope.GLOBAL:
            raise ValueError("FROST expects a global macro batch size")

        env: dict[str, str] = {}
        if gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        if omp_threads is not None:
            env["OMP_NUM_THREADS"] = str(omp_threads)
        if mkl_threads is not None:
            env["MKL_NUM_THREADS"] = str(mkl_threads)
        if warmup_batches is not None:
            env["FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH"] = "0"
            env["FROST_THROUGHPUT_WARMUP_BATCHES_TOTAL"] = str(warmup_batches)
        env["FROST_THROUGHPUT_MEASURE_START_EPOCH"] = str(measure_start_epoch)
        if measure_batches is not None:
            env["FROST_THROUGHPUT_MEASURE_BATCHES"] = str(measure_batches)
        if epoch_wall:
            env["FROST_EPOCH_WALL_BENCH"] = "1"

        command = (
            "uv",
            "run",
            "python",
            "-m",
            "frost_bench.cli.train",
            spec.model,
            "--nproc-per-node",
            str(spec.n_gpus + 1),
            "--master-port",
            str(master_port),
            f"dataset={spec.dataset}",
            f"epoch={spec.epochs}",
            f"macro_batch_size={spec.global_batch_size}",
            f"stable_mode={str(stable_mode).lower()}",
            f"seed={spec.seed}",
            f"compile={str(spec.compile).lower()}",
            f"compute.tf32={str(spec.tf32).lower()}",
            f"runtime.single_gpu_resident={'on' if spec.single_gpu_resident else 'off'}",
            *spec.extra_args,
        )
        return PreparedRun(system=self.name, command=command, cwd=REPO_ROOT, env=env)


def frost_benchmark_command(
    *,
    model: str,
    dataset: str,
    n_gpus: int,
    macro_batch_size: int,
    epochs: int,
    seed: int,
    master_port: int,
    gpu_ids: str,
    warmup_batches: int,
    measure_batches: int,
    omp_threads: int,
    mkl_threads: int,
    stable_mode: bool = True,
    measure_start_epoch: int = 0,
    overrides: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Compatibility view for standalone benchmark scripts.

    The maintained command and telemetry environment are owned by
    :class:`FrostAdapter`; scripts only supply experiment-specific overrides.
    """
    prepared = FrostAdapter().prepare(
        ExperimentSpec(
            model=model,
            dataset=dataset,
            seed=seed,
            n_gpus=n_gpus,
            batch_size=macro_batch_size,
            epochs=epochs,
            compile=False,
            tf32=False,
            extra_args=("tqdm=true", *overrides),
        ),
        master_port=master_port,
        warmup_batches=warmup_batches,
        measure_start_epoch=measure_start_epoch,
        measure_batches=measure_batches,
        omp_threads=omp_threads,
        mkl_threads=mkl_threads,
        gpu_ids=gpu_ids,
        stable_mode=stable_mode,
    )
    environment = os.environ.copy()
    environment.update(prepared.env)
    environment.update(extra_env or {})
    return list(prepared.command), environment
