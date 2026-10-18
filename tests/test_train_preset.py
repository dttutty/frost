import subprocess
import sys

import pytest

from frost_bench.cli import train as train_preset


def test_module_entrypoint_runs_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "frost_bench.cli.train", "tgn", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "torch.distributed.run" in result.stdout


def test_cawn_preset_uses_dyglib_style_transductive_parity_overrides():
    overrides = train_preset.PRESETS["cawn"].hydra_overrides

    assert "model=cawn" in overrides
    assert "neg_sampler=rnd_dst_then_random" in overrides
    assert "neg_sampler.train.candidate_mode=all" in overrides
    assert "neg_sampler.seed=0" in overrides
    assert "ignore_nfeat_on_disk=true" not in overrides


def test_tgn_preset_matches_memory_sweep_defaults():
    overrides = train_preset.PRESETS["tgn"].hydra_overrides

    assert "model=tgn" in overrides
    assert "stable_mode=true" in overrides
    assert "lr=0.0001" in overrides
    assert "d_time=128" in overrides
    assert "dropout=0.2" in overrides
    assert "ngh_sampler.tgl_sampler_parity=true" in overrides
    assert "neg_sampler=rnd_dst_then_random" in overrides
    assert "neg_sampler.train.candidate_mode=all" in overrides
    assert not any(override.startswith("neg_sampler.seed=") for override in overrides)


def test_jodie_preset_matches_memory_sweep_defaults():
    overrides = train_preset.PRESETS["jodie"].hydra_overrides

    assert "model=jodie" in overrides
    assert "stable_mode=true" in overrides
    assert "lr=0.0001" in overrides
    assert "d_time=128" in overrides
    assert "dropout=0.1" in overrides
    assert "neg_sampler=rnd_dst_then_random" in overrides
    assert "neg_sampler.train.candidate_mode=all" in overrides
    assert "neg_sampler.seed=null" in overrides
    assert "normalize_event_time=false" in overrides
    assert "persist_all_updates_to_cpu=true" in overrides


def test_apan_preset_matches_memory_sweep_defaults():
    overrides = train_preset.PRESETS["apan"].hydra_overrides

    assert "model=apan" in overrides
    assert "stable_mode=true" in overrides
    assert "lr=0.0001" in overrides
    assert "d_time=128" in overrides
    assert "dropout=0.1" in overrides
    assert "neg_sampler=rnd_dst_then_random" in overrides
    assert "neg_sampler.train.candidate_mode=all" in overrides
    assert "neg_sampler.seed=null" in overrides


def test_tgat_preset_uses_dyglib_style_non_memory_parity_overrides():
    overrides = train_preset.PRESETS["tgat"].hydra_overrides

    assert "model=tgat" in overrides
    assert "neg_sampler=rnd_dst_then_random" in overrides
    assert "neg_sampler.train.candidate_mode=all" in overrides
    assert "neg_sampler.seed=0" in overrides
    assert not any(override.startswith("macro_batch_size=") for override in overrides)
    assert "patience=20" in overrides
    assert "ignore_nfeat_on_disk=true" not in overrides


def test_all_stateless_dyglib_presets_use_on_disk_node_features():
    for preset_name in ("tgat", "graph_mixer", "tcl", "dygformer", "cawn"):
        overrides = train_preset.PRESETS[preset_name].hydra_overrides
        assert "ignore_nfeat_on_disk=true" not in overrides
        assert "neg_sampler.seed=0" in overrides


def test_build_command_injects_dataset_default_macro_batch_size():
    command = train_preset.build_command(
        preset=train_preset.PRESETS["tgn"],
        nproc_per_node=3,
        master_addr="127.0.0.1",
        master_port=29805,
        local_ranks_filter=None,
        extra_overrides=[],
    )

    assert "macro_batch_size=4000" in command


def test_build_command_scales_jodie_dataset_default():
    command = train_preset.build_command(
        preset=train_preset.PRESETS["jodie"],
        nproc_per_node=3,
        master_addr="127.0.0.1",
        master_port=29505,
        local_ranks_filter=None,
        extra_overrides=["dataset=REDDIT"],
    )

    assert "macro_batch_size=400" in command


def test_explicit_macro_batch_size_override_beats_dataset_default():
    command = train_preset.build_command(
        preset=train_preset.PRESETS["tgn"],
        nproc_per_node=3,
        master_addr="127.0.0.1",
        master_port=29805,
        local_ranks_filter=None,
        extra_overrides=["macro_batch_size=256"],
    )

    assert "macro_batch_size=256" in command
    assert "macro_batch_size=4000" not in command


@pytest.mark.parametrize("nproc_per_node", [1, 2, 4])
def test_tpnet_preset_uses_multi_gpu_external_trainer(nproc_per_node: int):
    command = train_preset.build_command(
        preset=train_preset.PRESETS["tpnet"],
        nproc_per_node=nproc_per_node,
        master_addr="127.0.0.1",
        master_port=29525,
        local_ranks_filter=None,
        extra_overrides=["--dataset_name", "wikipedia", "--batch_size", "200"],
    )
    assert command[:5] == [
        f"--nproc_per_node={nproc_per_node}",
        "--master_addr=127.0.0.1",
        "--master_port=29525",
        "-m",
        "frost_bench.tpnet_train",
    ]
    assert command[5:7] == ["--model_name", "TPNet"]
    assert command[-4:] == ["--dataset_name", "wikipedia", "--batch_size", "200"]
