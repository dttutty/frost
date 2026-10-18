from __future__ import annotations

import os
from pathlib import Path

from frost.data.paths import frost_data_root, tgb_data_root
from frost_bench.paths import PIPETGL_ROOT, REPO_ROOT
from frost_bench.spec import BatchSizeScope, ExperimentSpec, PreparedRun


class PipeTGLAdapter:
    """Prepare PipeTGL's explicit FROST-parity training entrypoint."""

    name = "pipetgl"
    dataset_names = {
        "WIKIPEDIA": "WIKI",
        "REDDIT": "REDDIT",
        "MOOC": "MOOC",
        "LASTFM": "LASTFM",
        "TGBL_WIKI": "TGBL_WIKI",
        "GDELT": "GDELT",
        "MAG": "MAG",
        "ENRON": "ENRON",
        "UCI": "UCI",
        "USLEGIS": "USLEGIS",
        "CANPARL": "CANPARL",
    }

    def __init__(
        self,
        entrypoint: Path | None = None,
        data_root: Path | None = None,
        tgb_root: Path | None = None,
    ) -> None:
        self.entrypoint = entrypoint or PIPETGL_ROOT / "train_frost_parity.py"
        self.data_root = data_root
        self.tgb_root = tgb_root

    def prepare(
        self,
        spec: ExperimentSpec,
        *,
        warmup_batches: int,
        measure_batches: int,
        measure_start_epoch: int = 0,
        omp_threads: int,
        mkl_threads: int,
        gpu_ids: str | None = None,
        max_train_batches: int = 0,
        train_only: bool = False,
        ignore_node_features_on_disk: bool = False,
    ) -> PreparedRun:
        if spec.model != "tgn":
            raise ValueError("PipeTGL parity currently supports only TGN")
        if spec.n_gpus < 2:
            raise ValueError("PipeTGL's pipeline requires at least two GPU ranks")
        if spec.batch_size_scope is not BatchSizeScope.GLOBAL:
            raise ValueError("PipeTGL expects a global temporal mini-batch size")
        if max_train_batches < 0:
            raise ValueError("max_train_batches must be non-negative")
        if not self.entrypoint.is_file():
            raise FileNotFoundError(f"Missing PipeTGL entrypoint: {self.entrypoint}")
        try:
            dataset_name = self.dataset_names[spec.dataset.upper()]
        except KeyError as exc:
            raise ValueError(f"Unsupported PipeTGL dataset: {spec.dataset}") from exc

        env = {
            "OMP_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(mkl_threads),
            "UV_SYSTEM_CERTS": "true",
        }
        if gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        existing_pythonpath = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(REPO_ROOT / "src"), existing_pythonpath) if value
        )
        tgb_args: tuple[str, ...] = ()
        if dataset_name == "TGBL_WIKI":
            raw_root = self.tgb_root or tgb_data_root()
            env["TGB_DATA_ROOT"] = str(raw_root)
            tgb_args = ("--tgb-data-root", str(raw_root))
        benchmark_args = (
            "--max-train-batches",
            str(max_train_batches),
            *(("--train-only",) if train_only else ()),
            *(
                ("--ignore-node-features-on-disk",)
                if ignore_node_features_on_disk
                else ()
            ),
        )

        command = (
            "uv",
            "run",
            "--project",
            str(PIPETGL_ROOT),
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(spec.n_gpus),
            str(self.entrypoint),
            "--data",
            dataset_name,
            "--data-root",
            str(self.data_root or frost_data_root()),
            *tgb_args,
            "--epoch",
            str(spec.epochs),
            "--seed",
            str(spec.seed),
            "--batch-size",
            str(spec.per_gpu_batch_size),
            "--lr",
            "0.0001",
            "--dim-time",
            "100",
            "--dim-embed",
            "100",
            "--dim-memory",
            "100",
            "--dropout",
            "0.2",
            "--fanout",
            "10",
            "--negative-sample-seed",
            "0",
            "--warmup-batches",
            str(warmup_batches),
            "--measure-batches",
            str(measure_batches),
            "--measure-start-epoch",
            str(measure_start_epoch),
            *benchmark_args,
            *spec.extra_args,
        )
        return PreparedRun(
            system=self.name,
            command=command,
            cwd=PIPETGL_ROOT,
            env=env,
        )
