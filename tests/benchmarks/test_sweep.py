from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from frost_bench.cli.sweep import build_jobs
from frost_bench.paths import REPO_ROOT
from frost_bench.sweep import SweepSpec, load_sweeps, summarize_seed_logs


def test_load_declared_sweeps() -> None:
    sweeps = load_sweeps(REPO_ROOT / "benchmarks" / "specs" / "sweeps.yaml")

    assert sweeps["memory_seeds"].models == ("tgn", "jodie", "apan")
    assert sweeps["dyglib_grid"].datasets == (
        "LASTFM",
        "MOOC",
        "REDDIT",
        "WIKIPEDIA",
    )


def test_build_jobs_reuses_preset_and_dyglib_parity_settings(tmp_path: Path) -> None:
    spec = SweepSpec(
        name="smoke",
        description="",
        models=("dygformer",),
        datasets=("REDDIT",),
        seeds=(3,),
        epoch=2,
        stable_mode=False,
    )

    job = build_jobs(
        spec,
        output_dir=tmp_path,
        nproc_per_node=3,
        master_port_base=29900,
        macro_batch_size=None,
        extra_overrides=[],
    )[0]

    assert "model.neighborhood.fanout=[63]" in job.command
    assert "model.encoder.max_input_sequence_length=64" in job.command
    assert "model.encoder.patch_size=2" in job.command
    assert "dropout=0.2" in job.command
    assert "seed=3" in job.command
    assert "stable_mode=false" in job.command


def test_summarize_seed_logs(tmp_path: Path) -> None:
    log_path = tmp_path / "seed_0.log"
    log_path.write_text(
        "Best Val AP: 0.72 @ Epoch 4 | Val AUC: 0.80 | Test AP: 0.73 |\n",
        encoding="utf-8",
    )

    metrics = summarize_seed_logs([(0, log_path)], output_dir=tmp_path)

    assert metrics[0].test_ap == 0.73
    assert (tmp_path / "results.tsv").is_file()
    assert "mean_test_ap=" in (tmp_path / "summary.txt").read_text()


def test_module_entrypoint_lists_and_dry_runs_sweeps() -> None:
    listed = subprocess.run(
        [sys.executable, "-m", "frost_bench.cli.sweep", "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "memory_seeds" in listed.stdout
    assert "dyglib_grid" in listed.stdout

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "frost_bench.cli.sweep",
            "memory_seeds",
            "--models",
            "tgn",
            "--datasets",
            "LASTFM",
            "--seeds",
            "0",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "torch.distributed.run" in dry_run.stdout
    assert "macro_batch_size=4000" in dry_run.stdout
