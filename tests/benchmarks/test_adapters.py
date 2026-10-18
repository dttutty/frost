import os
from pathlib import Path

import pytest
import yaml

from frost_bench.adapters import (
    DyGLibAdapter,
    FrostAdapter,
    PipeTGLAdapter,
    TGLAdapter,
)
from frost_bench.adapters.frost import frost_benchmark_command
from frost_bench.spec import BatchSizeScope, ExperimentSpec


def test_experiment_spec_makes_batch_scope_explicit() -> None:
    global_spec = ExperimentSpec(
        model="tgn",
        dataset="LASTFM",
        seed=0,
        n_gpus=2,
        batch_size=4000,
    )
    assert global_spec.per_gpu_batch_size == 2000

    per_gpu_spec = ExperimentSpec(
        model="tgn",
        dataset="LASTFM",
        seed=0,
        n_gpus=2,
        batch_size=2000,
        batch_size_scope=BatchSizeScope.PER_GPU,
    )
    assert per_gpu_spec.global_batch_size == 4000


def test_frost_adapter_reserves_one_host_rank() -> None:
    prepared = FrostAdapter().prepare(
        ExperimentSpec(
            model="tgn",
            dataset="LASTFM",
            seed=4,
            n_gpus=2,
            batch_size=4000,
            epochs=3,
        ),
        master_port=29501,
        measure_start_epoch=1,
    )
    command = prepared.command
    assert command[0:3] == ("uv", "run", "python")
    assert command[command.index("--nproc-per-node") + 1] == "3"
    assert "macro_batch_size=4000" in command
    assert prepared.env["FROST_THROUGHPUT_MEASURE_START_EPOCH"] == "1"


def test_frost_benchmark_command_owns_common_environment() -> None:
    command, env = frost_benchmark_command(
        model="tgn",
        dataset="LASTFM",
        n_gpus=2,
        macro_batch_size=4000,
        epochs=2,
        seed=0,
        master_port=29501,
        gpu_ids="2,3",
        warmup_batches=5,
        measure_batches=10,
        omp_threads=8,
        mkl_threads=4,
        overrides=("if_carry_over_route=false",),
    )
    assert "if_carry_over_route=false" in command
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["FROST_THROUGHPUT_MEASURE_BATCHES"] == "10"


def test_tgl_adapter_rejects_ambiguous_global_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="per-GPU"):
        TGLAdapter().prepare(
            ExperimentSpec(
                model="tgn",
                dataset="LASTFM",
                seed=0,
                n_gpus=2,
                batch_size=4000,
            ),
            config_path=tmp_path / "tgl.yml",
            max_train_steps=10,
            warmup_batches=1,
            measure_batches=2,
            omp_threads=8,
            mkl_threads=8,
        )


def test_tgl_adapter_passes_the_frost_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "frost-data"
    data_root.mkdir()
    prepared = TGLAdapter(data_root=data_root).prepare(
        ExperimentSpec(
            model="tgn",
            dataset="LASTFM",
            seed=0,
            n_gpus=2,
            batch_size=2000,
            batch_size_scope=BatchSizeScope.PER_GPU,
        ),
        config_path=tmp_path / "tgl.yml",
        max_train_steps=10,
        warmup_batches=1,
        measure_start_epoch=1,
        measure_batches=2,
        omp_threads=8,
        mkl_threads=8,
    )

    assert prepared.env["TGL_DATA_ROOT"] == str(data_root)
    assert prepared.command[
        prepared.command.index("--measure-start-epoch") + 1
    ] == "1"
    run_id = prepared.command[prepared.command.index("--run_id") + 1]
    assert run_id.startswith("frost_bench_tgl_")
    assert {path.name for path in prepared.cleanup_paths} == {
        f"{run_id}_{name}"
        for name in (
            "node_feats",
            "edge_feats",
            "node_memory",
            "node_memory_ts",
            "mails",
            "mails_ts",
            "next_mail_pos",
            "update_mail_pos",
        )
    }


def test_tgl_adapter_dispatches_single_gpu_resident_to_serial_trainer(
    tmp_path: Path,
) -> None:
    prepared = TGLAdapter(data_root=tmp_path).prepare(
        ExperimentSpec(
            model="tgn",
            dataset="LASTFM",
            seed=0,
            n_gpus=1,
            batch_size=4000,
            batch_size_scope=BatchSizeScope.PER_GPU,
            single_gpu_resident=True,
            extra_args=("--tqdm",),
        ),
        config_path=tmp_path / "tgl.yml",
        max_train_steps=10,
        warmup_batches=1,
        measure_batches=2,
        omp_threads=8,
        mkl_threads=8,
        gpu_ids="3",
    )
    assert prepared.command[prepared.command.index("--gpu") + 1] == "3"
    assert "train.py" in prepared.command[5]
    assert "--gpu-resident-buffers" in prepared.command
    assert "--tqdm" not in prepared.command
    assert "CUDA_VISIBLE_DEVICES" not in prepared.env
    assert prepared.cleanup_paths == ()


def test_tgl_adapter_writes_structured_config(tmp_path: Path) -> None:
    source = tmp_path / "TGN.yml"
    source.write_text(
        yaml.safe_dump(
            {
                "train": [{"epoch": 1, "batch_size": 64}],
                "sampling": [{"layer": 1}],
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "tgl.yml"
    TGLAdapter().write_config(
        source=source,
        destination=destination,
        epochs=7,
        per_gpu_batch_size=123,
    )
    payload = yaml.safe_load(destination.read_text())
    assert payload["train"][0]["epoch"] == 7
    assert payload["train"][0]["batch_size"] == 123


def test_tgl_adapter_can_run_full_evaluation_without_appendix_mode(
    tmp_path: Path,
) -> None:
    prepared = TGLAdapter(data_root=tmp_path).prepare(
        ExperimentSpec(
            model="tgn",
            dataset="WIKIPEDIA",
            seed=0,
            n_gpus=2,
            batch_size=300,
            batch_size_scope=BatchSizeScope.PER_GPU,
            epochs=2,
        ),
        config_path=tmp_path / "tgl.yml",
        max_train_steps=0,
        warmup_batches=0,
        measure_batches=0,
        omp_threads=8,
        mkl_threads=8,
        appendix_train_e2e=False,
    )

    assert "--appendix-train-e2e" not in prepared.command
    assert "--max-train-steps" not in prepared.command


def test_dyglib_adapter_uses_canonical_frost_entrypoint(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "train_link_prediction.py"
    entrypoint.touch()
    data_root = tmp_path / "frost-data"
    data_root.mkdir()
    prepared = DyGLibAdapter(entrypoint=entrypoint, data_root=data_root).prepare(
        ExperimentSpec(
            model="dygformer",
            dataset="LASTFM",
            seed=0,
            n_gpus=1,
            batch_size=2000,
            epochs=1000,
        ),
        max_train_batches=20,
        warmup_batches=2,
        measure_batches=5,
        omp_threads=8,
        mkl_threads=8,
    )
    assert str(entrypoint) in prepared.command
    assert "--data-mode" not in prepared.command
    assert "--negative_sample_strategy" not in prepared.command
    assert prepared.command[prepared.command.index("--data-root") + 1] == str(data_root)
    assert "--use-node-features-on-disk" in prepared.command
    assert "--zero-node-features" not in prepared.command
    assert "--features-in-host-memory" in prepared.command


def test_pipetgl_adapter_uses_isolated_parity_entrypoint(tmp_path: Path) -> None:
    entrypoint = tmp_path / "train_frost_parity.py"
    entrypoint.touch()
    data_root = tmp_path / "frost-data"
    data_root.mkdir()
    prepared = PipeTGLAdapter(entrypoint=entrypoint, data_root=data_root).prepare(
        ExperimentSpec(
            model="tgn",
            dataset="WIKIPEDIA",
            seed=7,
            n_gpus=2,
            batch_size=4000,
            epochs=3,
        ),
        warmup_batches=2,
        measure_batches=5,
        omp_threads=8,
        mkl_threads=8,
        gpu_ids="0,1",
    )
    assert prepared.command[0:4] == ("uv", "run", "--project", str(prepared.cwd))
    assert prepared.command[prepared.command.index("--nproc_per_node") + 1] == "2"
    assert prepared.command[prepared.command.index("--data") + 1] == "WIKI"
    assert prepared.command[prepared.command.index("--data-root") + 1] == str(data_root)
    assert prepared.command[prepared.command.index("--batch-size") + 1] == "2000"
    assert "--no-scale-lr-with-world-size" not in prepared.command
    assert prepared.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert prepared.env["UV_SYSTEM_CERTS"] == "true"


def test_pipetgl_adapter_requires_pipeline_parallelism(tmp_path: Path) -> None:
    entrypoint = tmp_path / "train_frost_parity.py"
    entrypoint.touch()
    with pytest.raises(ValueError, match="at least two"):
        PipeTGLAdapter(entrypoint=entrypoint).prepare(
            ExperimentSpec(
                model="tgn",
                dataset="LASTFM",
                seed=0,
                n_gpus=1,
                batch_size=4000,
            ),
            warmup_batches=2,
            measure_batches=5,
            omp_threads=8,
            mkl_threads=8,
        )


def test_pipetgl_adapter_registers_official_tgbl_wiki(tmp_path: Path) -> None:
    entrypoint = tmp_path / "train_frost_parity.py"
    entrypoint.touch()
    data_root = tmp_path / "frost-data"
    tgb_root = tmp_path / "tgb-data"
    data_root.mkdir()
    tgb_root.mkdir()
    prepared = PipeTGLAdapter(
        entrypoint=entrypoint,
        data_root=data_root,
        tgb_root=tgb_root,
    ).prepare(
        ExperimentSpec(
            model="tgn",
            dataset="TGBL_WIKI",
            seed=0,
            n_gpus=2,
            batch_size=600,
        ),
        warmup_batches=2,
        measure_batches=5,
        omp_threads=8,
        mkl_threads=8,
    )

    assert prepared.command[prepared.command.index("--data") + 1] == "TGBL_WIKI"
    assert prepared.command[prepared.command.index("--tgb-data-root") + 1] == str(
        tgb_root
    )
    assert prepared.env["TGB_DATA_ROOT"] == str(tgb_root)
    assert prepared.env["PYTHONPATH"].split(os.pathsep)[0].endswith("/src")


@pytest.mark.parametrize("dataset", ["ENRON", "UCI", "USLEGIS", "CANPARL"])
def test_pipetgl_adapter_registers_additional_frost_datasets(
    tmp_path: Path, dataset: str
) -> None:
    entrypoint = tmp_path / "train_frost_parity.py"
    entrypoint.touch()
    prepared = PipeTGLAdapter(entrypoint=entrypoint, data_root=tmp_path).prepare(
        ExperimentSpec(
            model="tgn",
            dataset=dataset,
            seed=0,
            n_gpus=2,
            batch_size=1200,
        ),
        warmup_batches=0,
        measure_batches=0,
        omp_threads=8,
        mkl_threads=8,
    )

    assert prepared.command[prepared.command.index("--data") + 1] == dataset
    assert prepared.command[prepared.command.index("--batch-size") + 1] == "600"


def test_pipetgl_adapter_forwards_measure_start_epoch(tmp_path: Path) -> None:
    entrypoint = tmp_path / "train_frost_parity.py"
    entrypoint.touch()
    prepared = PipeTGLAdapter(entrypoint=entrypoint, data_root=tmp_path).prepare(
        ExperimentSpec(
            model="tgn",
            dataset="WIKIPEDIA",
            seed=0,
            n_gpus=2,
            batch_size=600,
        ),
        warmup_batches=0,
        measure_batches=0,
        measure_start_epoch=1,
        omp_threads=8,
        mkl_threads=8,
    )

    assert prepared.command[prepared.command.index("--measure-start-epoch") + 1] == "1"


@pytest.mark.parametrize("dataset", ["MAG", "GDELT"])
def test_pipetgl_adapter_supports_large_train_only_runs(
    tmp_path: Path, dataset: str
) -> None:
    entrypoint = tmp_path / "train_frost_parity.py"
    entrypoint.touch()
    prepared = PipeTGLAdapter(entrypoint=entrypoint, data_root=tmp_path).prepare(
        ExperimentSpec(
            model="tgn",
            dataset=dataset,
            seed=0,
            n_gpus=2,
            batch_size=32000,
        ),
        warmup_batches=100,
        measure_batches=7000,
        omp_threads=32,
        mkl_threads=32,
        max_train_batches=7102,
        train_only=True,
    )

    assert prepared.command[prepared.command.index("--data") + 1] == dataset
    assert prepared.command[prepared.command.index("--batch-size") + 1] == "16000"
    assert prepared.command[prepared.command.index("--max-train-batches") + 1] == "7102"
    assert "--train-only" in prepared.command
    assert "--ignore-node-features-on-disk" not in prepared.command


def test_pipetgl_adapter_keeps_node_feature_ablation_explicit(tmp_path: Path) -> None:
    entrypoint = tmp_path / "train_frost_parity.py"
    entrypoint.touch()
    prepared = PipeTGLAdapter(entrypoint=entrypoint, data_root=tmp_path).prepare(
        ExperimentSpec(
            model="tgn",
            dataset="MAG",
            seed=0,
            n_gpus=2,
            batch_size=32000,
        ),
        warmup_batches=100,
        measure_batches=100,
        omp_threads=32,
        mkl_threads=32,
        ignore_node_features_on_disk=True,
    )

    assert "--ignore-node-features-on-disk" in prepared.command
