import subprocess
import sys
from pathlib import Path

import pytest

from frost_bench.suite import load_suites, parse_overrides
from frost_bench.cli.suite import DEFAULT_SPEC
from frost_bench.throughput import (
    _systems_for_job,
    _thread_setting,
    build_throughput_jobs,
)


def test_load_suite_and_build_command(tmp_path: Path) -> None:
    spec = tmp_path / "suites.yaml"
    spec.write_text(
        """
suites:
  smoke:
    profile: memory_models
    description: smoke suite
    arguments:
      models: [tgn, jodie]
      measure_batches: 100
      enabled: true
      disabled: false
""",
        encoding="utf-8",
    )

    suite = load_suites(spec)["smoke"]
    command = suite.command(overrides={"measure_batches": 3})

    assert command[:6] == (
        "uv",
        "run",
        "python",
        "-m",
        "frost_bench.cli.throughput",
        "smoke",
    )
    assert "models=[\"tgn\", \"jodie\"]" in command
    assert "measure_batches=3" in command
    assert "enabled=true" in command
    assert "disabled=false" in command


def test_parse_overrides_accepts_yaml_scalars_and_lists() -> None:
    assert parse_overrides(["measure-batches=3", "models=[tgn, jodie]"]) == {
        "measure_batches": 3,
        "models": ("tgn", "jodie"),
    }


def test_thread_setting_accepts_auto_or_positive_integer() -> None:
    assert _thread_setting({}, "omp_threads") is None
    assert _thread_setting({"omp_threads": "auto"}, "omp_threads") is None
    assert _thread_setting({"omp_threads": 16}, "omp_threads") == 16


def test_throughput_profile_expands_normalized_jobs() -> None:
    jobs = build_throughput_jobs(
        "tgn_worker_queue",
        {
            "macro_batch_size": 4000,
            "mem_dims": (128, 256),
            "worker_queue_sizes": (1, 2),
        },
    )

    assert len(jobs) == 4
    assert jobs[0].frost_overrides == ("model.embedding_dim=128",)
    assert jobs[-1].frost_env == {"FROST_MEMORY_WORKER_QUEUE_SIZE": "2"}


def test_memory_profile_uses_model_and_dataset_batch_defaults() -> None:
    jobs = build_throughput_jobs(
        "memory_models",
        {
            "models": ("tgn", "jodie"),
            "datasets": ("WIKIPEDIA", "LASTFM"),
        },
    )

    assert [(job.model, job.dataset, job.macro_batch_size) for job in jobs] == [
        ("tgn", "WIKIPEDIA", 600),
        ("tgn", "LASTFM", 4000),
        ("jodie", "WIKIPEDIA", 120),
        ("jodie", "LASTFM", 800),
    ]


def test_tgat_tgl_job_uses_matched_frost_preset_and_real_features() -> None:
    job = build_throughput_jobs(
        "memory_models",
        {
            "models": ("tgat",),
            "datasets": ("WIKIPEDIA",),
            "batch_sizes": (4000,),
        },
    )[0]

    assert job.compare_tgl is True
    assert job.tgl_overrides == {
        "sampling.0.neighbor": [20, 20],
        "sampling.0.strategy": "recent",
        "gnn.0.dim_time": 100,
    }
    assert job.tgl_args == ()


def test_all_maintained_throughput_profiles_expand() -> None:
    suites = load_suites(DEFAULT_SPEC)

    for suite in suites.values():
        assert suite.profile is not None
        assert build_throughput_jobs(suite.profile, suite.arguments)

    assert suites["large_tgn_batch_sweep"].arguments["omp_threads"] == "auto"


def test_large_profiles_select_the_expected_baseline() -> None:
    tgn_job = build_throughput_jobs(
        "large_tgn",
        {"datasets": ("MAG",), "batch_sizes": (32000,), "full_epoch": True},
    )[0]
    dygformer_job = build_throughput_jobs(
        "large_dygformer",
        {"datasets": ("GDELT",), "batch_sizes": (4000,), "full_epoch": False},
    )[0]

    assert tgn_job.compare_tgl is True
    assert tgn_job.frost_overrides == ()
    assert tgn_job.tgl_args == ()
    assert dygformer_job.compare_dyglib is True
    assert "ignore_nfeat_on_disk=true" not in dygformer_job.frost_overrides


def test_large_tgn_can_disable_all_disk_features_for_both_systems() -> None:
    job = build_throughput_jobs(
        "large_tgn",
        {
            "datasets": ("GDELT",),
            "batch_sizes": (32000,),
            "disable_features": True,
            "full_epoch": False,
        },
    )[0]

    assert job.dimensions["features"] == "none"
    assert job.frost_overrides == (
        "ignore_nfeat_on_disk=true",
        "ignore_efeat_on_disk=true",
    )
    assert job.tgl_args == (
        "--ignore-node-features-on-disk",
        "--ignore-edge-features-on-disk",
    )


def test_large_tgn_can_enable_stage_profiling_for_both_systems() -> None:
    job = build_throughput_jobs(
        "large_tgn",
        {
            "datasets": ("GDELT",),
            "batch_sizes": (64000,),
            "profile_stages": True,
            "full_epoch": False,
        },
    )[0]

    assert job.frost_env == {
        "FROST_HOST_TIMING_BENCH": "1",
        "FROST_PIPELINE_TIMING_BENCH": "1",
    }
    assert job.tgl_args == ("--stage-timing",)


def test_large_tgn_can_profile_with_all_disk_features_disabled() -> None:
    job = build_throughput_jobs(
        "large_tgn",
        {
            "datasets": ("GDELT",),
            "batch_sizes": (64000,),
            "disable_features": True,
            "profile_stages": True,
            "full_epoch": False,
        },
    )[0]

    assert job.frost_overrides == (
        "ignore_nfeat_on_disk=true",
        "ignore_efeat_on_disk=true",
    )
    assert job.tgl_args == (
        "--ignore-node-features-on-disk",
        "--ignore-edge-features-on-disk",
        "--stage-timing",
    )


def test_large_tgn_expands_compile_ablation() -> None:
    jobs = build_throughput_jobs(
        "large_tgn",
        {
            "datasets": ("MAG",),
            "batch_sizes": (32000, 64000),
            "compile": (False, True),
        },
    )

    assert len(jobs) == 4
    assert [job.compile for job in jobs] == [False, True, False, True]
    assert [job.dimensions["compile"] for job in jobs] == [
        False,
        True,
        False,
        True,
    ]
    assert len({job.slug for job in jobs}) == 4


def test_large_tgn_rejects_non_boolean_compile_values() -> None:
    with pytest.raises(TypeError, match="compile must be"):
        build_throughput_jobs(
            "large_tgn",
            {
                "datasets": ("MAG",),
                "batch_sizes": (32000,),
                "compile": (False, "true"),
            },
        )


def test_large_tgn_expands_compile_mode_ablation() -> None:
    jobs = build_throughput_jobs(
        "large_tgn",
        {
            "datasets": ("MAG",),
            "batch_sizes": (128000,),
            "compile": True,
            "compile_mode": (
                "default",
                "reduce-overhead",
                "max-autotune-no-cudagraphs",
            ),
        },
    )

    assert [job.dimensions["compile_mode"] for job in jobs] == [
        "default",
        "reduce-overhead",
        "max-autotune-no-cudagraphs",
    ]
    assert [job.frost_overrides[-1] for job in jobs] == [
        "compile_mode=default",
        "compile_mode=reduce-overhead",
        "compile_mode=max-autotune-no-cudagraphs",
    ]


def test_large_tgn_rejects_unknown_compile_mode() -> None:
    with pytest.raises(ValueError, match="compile_mode"):
        build_throughput_jobs(
            "large_tgn",
            {
                "datasets": ("MAG",),
                "batch_sizes": (128000,),
                "compile": True,
                "compile_mode": "fastest",
            },
        )


def test_large_tgn_expands_attention_layout_ablation() -> None:
    jobs = build_throughput_jobs(
        "large_tgn",
        {
            "datasets": ("MAG",),
            "batch_sizes": (32000,),
            "attention_layout": ("dense", "packed"),
        },
    )

    assert [job.dimensions["attention_layout"] for job in jobs] == [
        "dense",
        "packed",
    ]
    assert [job.frost_overrides[-1] for job in jobs] == [
        "model.encoder.attention_layout=dense",
        "model.encoder.attention_layout=packed",
    ]
    assert len({job.slug for job in jobs}) == 2


def test_large_tgn_accepts_auto_attention_layout() -> None:
    jobs = build_throughput_jobs(
        "large_tgn",
        {
            "datasets": ("MAG",),
            "batch_sizes": (32000,),
            "attention_layout": ("auto",),
        },
    )

    assert jobs[0].dimensions["attention_layout"] == "auto"
    assert jobs[0].frost_overrides[-1] == "model.encoder.attention_layout=auto"


def test_large_tgn_rejects_unknown_attention_layout() -> None:
    with pytest.raises(ValueError, match="attention_layout"):
        build_throughput_jobs(
            "large_tgn",
            {
                "datasets": ("MAG",),
                "batch_sizes": (32000,),
                "attention_layout": ("dense", "sparse"),
            },
        )


def test_large_tgn_expands_dynamic_state_ablation() -> None:
    jobs = build_throughput_jobs(
        "large_tgn",
        {
            "datasets": ("MAG",),
            "batch_sizes": (64000,),
            "dynamic_state_ablation": (
                "baseline",
                "baseline_pinned_gather",
            ),
        },
    )

    assert [job.dimensions["dynamic_state_ablation"] for job in jobs] == [
        "baseline",
        "baseline_pinned_gather",
    ]
    assert [job.frost_overrides[-1] for job in jobs] == [
        "dynamic_state_ablation=baseline",
        "dynamic_state_ablation=baseline_pinned_gather",
    ]
    assert len({job.slug for job in jobs}) == 2


def test_throughput_job_can_select_one_comparison_system() -> None:
    job = build_throughput_jobs(
        "large_tgn",
        {"datasets": ("GDELT",), "batch_sizes": (32000,), "full_epoch": False},
    )[0]

    assert _systems_for_job(job, ()) == ("frost", "tgl")
    assert _systems_for_job(job, ("frost",)) == ("frost",)
    assert _systems_for_job(job, ("tgl",)) == ("tgl",)


def test_module_entrypoint_lists_suites() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "frost_bench.cli.suite", "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "memory_models_2gpu" in result.stdout
