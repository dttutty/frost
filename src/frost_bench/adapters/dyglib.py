from __future__ import annotations

from pathlib import Path

from frost.data.paths import frost_data_root
from frost_bench.paths import DYGLIB_ROOT
from frost_bench.spec import BatchSizeScope, ExperimentSpec, PreparedRun


class DyGLibAdapter:
    name = "dyglib"
    model_names = {
        "dygformer": "DyGFormer",
        "graph_mixer": "GraphMixer",
        "tcl": "TCL",
        "cawn": "CAWN",
        "tgat": "TGAT",
    }

    def __init__(
        self, entrypoint: Path | None = None, data_root: Path | None = None
    ) -> None:
        self.entrypoint = entrypoint or (DYGLIB_ROOT / "train_link_prediction.py")
        self.data_root = data_root

    def prepare(
        self,
        spec: ExperimentSpec,
        *,
        max_train_batches: int,
        warmup_batches: int,
        measure_batches: int,
        omp_threads: int,
        mkl_threads: int,
        gpu_ids: str | None = None,
    ) -> PreparedRun:
        if spec.n_gpus != 1:
            raise ValueError("The DyGLib adapter currently supports one GPU per run")
        if spec.batch_size_scope not in {
            BatchSizeScope.GLOBAL,
            BatchSizeScope.PER_GPU,
        }:
            raise ValueError("DyGLib expects a single-process batch size")
        if not self.entrypoint.is_file():
            raise FileNotFoundError(f"Missing DyGLib entrypoint: {self.entrypoint}")

        try:
            model_name = self.model_names[spec.model]
        except KeyError as exc:
            raise ValueError(f"Unsupported DyGLib model: {spec.model}") from exc

        env = {
            "OMP_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(mkl_threads),
        }
        if gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        command = (
            "uv",
            "run",
            "--project",
            str(DYGLIB_ROOT),
            "python",
            str(self.entrypoint),
            "--seed-start",
            str(spec.seed),
            "--dataset_name",
            spec.dataset.lower(),
            "--model_name",
            model_name,
            "--load_best_configs",
            "--gpu",
            "0",
            "--num_runs",
            "1",
            "--num_epochs",
            str(spec.epochs),
            "--batch_size",
            str(spec.batch_size),
            "--test_interval_epochs",
            "1",
            "--data-root",
            str(self.data_root or frost_data_root()),
            "--use-node-features-on-disk",
            "--features-in-host-memory",
            "--appendix-train-e2e",
            "--warmup-batches",
            str(warmup_batches),
            "--measure-batches",
            str(measure_batches),
            "--max-train-batches",
            str(max_train_batches),
            *spec.extra_args,
        )
        return PreparedRun(system=self.name, command=command, cwd=DYGLIB_ROOT, env=env)
