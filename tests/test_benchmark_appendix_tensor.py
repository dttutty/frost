import importlib.util
import sys
from pathlib import Path

import yaml


def _load_benchmark_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_appendix_tensor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_appendix_tensor", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_frost_train_command_uses_gpu_plus_host_world_size():
    bench = _load_benchmark_module()

    command = bench.build_frost_train_command(
        preset="tgn",
        dataset="LASTFM",
        num_gpus=2,
        epochs=4,
        compile_enabled=True,
        macro_batch_size=600,
    )

    assert "--nproc-per-node=3" in command
    assert "compile=true" in command
    assert "epoch=4" in command
    assert "macro_batch_size=600" in command


def test_parse_summary_lines_extracts_json_payload():
    bench = _load_benchmark_module()
    text = """
noise
APPENDIX_TRAIN_E2E_SUMMARY {"avg_wall_ms": 12.5, "compile": false, "rank": 0, "system": "frost"}
more noise
APPENDIX_TRAIN_E2E_SUMMARY {"avg_wall_ms": 11.5, "compile": false, "rank": 1, "system": "frost"}
"""

    rows = bench.parse_summary_lines(text)

    assert len(rows) == 2
    assert rows[0]["rank"] == 0
    assert rows[1]["avg_wall_ms"] == 11.5


def test_parse_summary_lines_falls_back_to_current_throughput_output():
    bench = _load_benchmark_module()
    text = (
        "TRAIN_THROUGHPUT_PER_GPU rank=0 "
        "events_per_sec_per_gpu=100000.0 "
        "events_per_sec_per_gpu_gpu_clock=100100.0 "
        "events=400000 elapsed_s=4.000 gpu_elapsed_s=3.996 "
        "warmup_batches_per_epoch=0 warmup_batches_total=20 "
        "measured_batches=100 measure_target=100\n"
    )

    rows = bench.parse_summary_lines(text)

    assert rows == [
        {
            "system": "frost",
            "rank": 0,
            "avg_wall_ms": 40.0,
            "events_per_sec_per_gpu": 100000.0,
            "measured_batches": 100,
        }
    ]


def test_run_command_returns_captured_log_and_allowed_failure(tmp_path):
    bench = _load_benchmark_module()

    ok_log = tmp_path / "ok.log"
    returncode, text = bench.run_command(
        command=[sys.executable, "-c", "print('captured output')"],
        env={},
        log_path=ok_log,
        allow_failure=False,
        stage_name="success case",
    )
    assert returncode == 0
    assert text.strip() == "captured output"

    failed_log = tmp_path / "failed.log"
    returncode, text = bench.run_command(
        command=[sys.executable, "-c", "print('failed output'); raise SystemExit(7)"],
        env={},
        log_path=failed_log,
        allow_failure=True,
        stage_name="allowed failure case",
    )
    assert returncode == 7
    assert text.strip() == "failed output"


def test_build_tgl_train_command_uses_tgl_project_env(tmp_path):
    bench = _load_benchmark_module()

    command = bench.build_tgl_train_command(
        dataset="LASTFM",
        config_path=tmp_path / "tgl.yml",
        num_gpus=2,
        compile_enabled=True,
        measure_start_epoch=1,
        warmup_batches=20,
    )

    assert command[:5] == [
        "uv",
        "run",
        "--project",
        "third_party/tgl",
        "python",
    ]
    assert "third_party/tgl/train_dist.py" in command
    assert "--compile" in command


def test_write_tgl_config_copy_overrides_epoch(tmp_path, monkeypatch):
    bench = _load_benchmark_module()
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
    monkeypatch.setitem(bench.TGL_CONFIGS, "tgn", source)

    config_path = bench.write_tgl_config_copy(
        preset="tgn",
        epochs=7,
        output_path=tmp_path / "tgl_tgn_ep7.yml",
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["train"][0]["epoch"] == 7
    assert config["sampling"][0]["layer"] == 1


def test_parse_preset_names_all_expands_to_every_frost_preset():
    bench = _load_benchmark_module()

    presets = bench.parse_preset_names("all")

    assert presets == list(bench.FROST_PRESET_NAMES)
    assert "tgn" in presets
    assert "cawn" in presets


def test_resolve_effective_num_gpus_clamps_to_visible(monkeypatch):
    bench = _load_benchmark_module()

    monkeypatch.setattr(bench, "get_visible_cuda_device_count", lambda: 1)

    effective, visible = bench.resolve_effective_num_gpus(2)

    assert effective == 1
    assert visible == 1


def test_run_compile_uplift_skips_tgl_when_not_requested(monkeypatch, tmp_path):
    bench = _load_benchmark_module()

    frost_eager = bench.TrainRunResult(
        system="frost",
        preset="tgn",
        compile_enabled=False,
        status="ok",
        log_path=tmp_path / "frost_eager.log",
        summaries=[{"avg_wall_ms": 10.0}],
        avg_wall_ms=10.0,
        std_wall_ms=0.5,
    )
    frost_compiled = bench.TrainRunResult(
        system="frost",
        preset="tgn",
        compile_enabled=True,
        status="ok",
        log_path=tmp_path / "frost_compile.log",
        summaries=[{"avg_wall_ms": 8.0}],
        avg_wall_ms=8.0,
        std_wall_ms=0.4,
    )

    monkeypatch.setattr(bench, "run_frost_train_case", lambda **kwargs: frost_compiled if kwargs["compile_enabled"] else frost_eager)

    def _unexpected_tgl(**kwargs):
        raise AssertionError("TGL path should not run when include_tgl is False")

    monkeypatch.setattr(bench, "run_tgl_train_case", _unexpected_tgl)

    rows, raw_results = bench.run_compile_uplift(
        presets=["tgn"],
        dataset="LASTFM",
        num_gpus=1,
        epochs=4,
        measure_start_epoch=1,
        warmup_batches=20,
        output_dir=tmp_path,
        include_tgl=False,
        macro_batch_size=600,
    )

    assert len(rows) == 1
    assert rows[0].runtime == "FROST HierGraph"
    assert rows[0].uplift == 1.25
    assert len(raw_results) == 2


def test_run_compile_uplift_passes_macro_batch_size_to_each_frost_run(monkeypatch, tmp_path):
    bench = _load_benchmark_module()

    calls = []

    def _fake_frost_run(**kwargs):
        calls.append(kwargs)
        return bench.TrainRunResult(
            system="frost",
            preset=kwargs["preset"],
            compile_enabled=kwargs["compile_enabled"],
            status="ok",
            log_path=tmp_path / f'{kwargs["preset"]}_{kwargs["compile_enabled"]}.log',
            summaries=[{"avg_wall_ms": 10.0 if not kwargs["compile_enabled"] else 8.0}],
            avg_wall_ms=10.0 if not kwargs["compile_enabled"] else 8.0,
            std_wall_ms=0.5,
        )

    monkeypatch.setattr(bench, "run_frost_train_case", _fake_frost_run)
    monkeypatch.setattr(
        bench,
        "run_tgl_train_case",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected TGL run")),
    )

    rows, raw_results = bench.run_compile_uplift(
        presets=["tgn", "jodie"],
        dataset="LASTFM",
        num_gpus=1,
        epochs=4,
        measure_start_epoch=1,
        warmup_batches=20,
        output_dir=tmp_path,
        include_tgl=False,
        macro_batch_size=600,
    )

    assert [row.preset for row in rows] == ["tgn", "jodie"]
    assert len(raw_results) == 4
    assert len(calls) == 4
    assert all(call["macro_batch_size"] == 600 for call in calls)
