from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import yaml

from frost.data.paths import frost_data_root
from frost_bench.paths import REPO_ROOT, TGL_ROOT
from frost_bench.spec import BatchSizeScope, ExperimentSpec, PreparedRun


def write_tgl_config(
    source: Path,
    destination: Path,
    epochs: int | None = None,
    per_gpu_batch_size: int | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> Path:
    """Write one derived TGL config without text-based YAML rewrites."""
    with source.open("r", encoding="utf-8") as handle:
        config = deepcopy(yaml.safe_load(handle))
    if epochs is not None:
        config["train"][0]["epoch"] = epochs
    if per_gpu_batch_size is not None:
        config["train"][0]["batch_size"] = per_gpu_batch_size
    for dotted_path, value in (overrides or {}).items():
        target = config
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        leaf = parts[-1]
        if isinstance(target, list):
            target[int(leaf)] = value
        else:
            target[leaf] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return destination


class TGLAdapter:
    name = "tgl"

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root

    def write_config(
        self,
        *,
        source: Path,
        destination: Path,
        epochs: int,
        per_gpu_batch_size: int,
        overrides: Mapping[str, object] | None = None,
    ) -> Path:
        return write_tgl_config(
            source,
            destination,
            epochs,
            per_gpu_batch_size,
            overrides=overrides,
        )

    def prepare(
        self,
        spec: ExperimentSpec,
        *,
        config_path: Path,
        max_train_steps: int,
        warmup_batches: int,
        measure_batches: int,
        omp_threads: int,
        mkl_threads: int,
        measure_start_epoch: int = 0,
        gpu_ids: str | None = None,
        appendix_train_e2e: bool = True,
    ) -> PreparedRun:
        if spec.batch_size_scope is not BatchSizeScope.PER_GPU:
            raise ValueError("TGL train_dist expects a per-GPU batch size")
        env = {
            "OMP_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(mkl_threads),
            "TGL_DATA_ROOT": str(self.data_root or frost_data_root()),
        }
        if gpu_ids is not None and not spec.single_gpu_resident:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        if spec.single_gpu_resident:
            selected_gpu = (
                gpu_ids.split(",", 1)[0].strip() if gpu_ids is not None else "0"
            )
            serial_extra_args = tuple(
                arg for arg in spec.extra_args if arg != "--tqdm"
            )
            appendix_args = (
                (
                    "--appendix-train-e2e",
                    "--measure-start-epoch",
                    str(measure_start_epoch),
                    "--warmup-batches",
                    str(warmup_batches),
                    "--measure-batches",
                    str(measure_batches),
                    "--max-train-steps",
                    str(max_train_steps),
                )
                if appendix_train_e2e
                else ()
            )
            return PreparedRun(
                system=self.name,
                command=(
                    "uv",
                    "run",
                    "--project",
                    str(TGL_ROOT),
                    "python",
                    str(TGL_ROOT / "train.py"),
                    "--data",
                    spec.dataset.lower(),
                    "--config",
                    str(config_path),
                    "--gpu",
                    selected_gpu,
                    "--seed",
                    str(spec.seed),
                    "--rand_edge_features",
                    "0",
                    "--rand_node_features",
                    "0",
                    "--gpu-resident-buffers",
                    *appendix_args,
                    *serial_extra_args,
                ),
                cwd=REPO_ROOT,
                env=env,
            )

        run_id = f"frost_bench_tgl_{uuid4().hex}"
        shared_memory_names = (
            "node_feats",
            "edge_feats",
            "node_memory",
            "node_memory_ts",
            "mails",
            "mails_ts",
            "next_mail_pos",
            "update_mail_pos",
        )

        appendix_args = (
            (
                "--appendix-train-e2e",
                "--measure-start-epoch",
                str(measure_start_epoch),
                "--warmup-batches",
                str(warmup_batches),
                "--measure-batches",
                str(measure_batches),
                "--max-train-steps",
                str(max_train_steps),
            )
            if appendix_train_e2e
            else ()
        )
        command = (
            "uv",
            "run",
            "--project",
            str(TGL_ROOT),
            "python",
            str(TGL_ROOT / "train_dist.py"),
            "--dataset",
            spec.dataset.lower(),
            "--config",
            str(config_path),
            "--num_gpus",
            str(spec.n_gpus),
            "--seed",
            str(spec.seed),
            "--run_id",
            run_id,
            "--omp_num_threads",
            str(omp_threads),
            "--rnd_edim",
            "0",
            "--rnd_ndim",
            "0",
            *appendix_args,
            *spec.extra_args,
        )
        return PreparedRun(
            system=self.name,
            command=command,
            cwd=REPO_ROOT,
            env=env,
            cleanup_paths=tuple(
                Path("/dev/shm") / f"{run_id}_{name}" for name in shared_memory_names
            ),
        )
