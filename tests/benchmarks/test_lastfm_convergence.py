from __future__ import annotations

from frost_bench.cli.lastfm_convergence import _parser, _prepared_run


def _run_args(*extra: str):
    return _parser().parse_args(
        [
            "run",
            "--system",
            "frost",
            "--gpu-ids",
            "0,1",
            "--data-root",
            "/tmp/data",
            "--output",
            "/tmp/output",
            *extra,
        ]
    )


def test_convergence_defaults_preserve_lastfm_protocol() -> None:
    prepared = _prepared_run(_run_args(), seed=3)

    assert "dataset=LASTFM" in prepared.command
    assert "macro_batch_size=4000" in prepared.command
    assert "stable_mode=true" in prepared.command
    assert "seed=3" in prepared.command


def test_convergence_normalizes_wiki_and_uses_dataset_batch_size() -> None:
    prepared = _prepared_run(_run_args("--dataset", "WIKI"), seed=0)

    assert "dataset=WIKIPEDIA" in prepared.command
    assert "macro_batch_size=600" in prepared.command


def test_convergence_explicit_batch_size_overrides_dataset_default() -> None:
    prepared = _prepared_run(
        _run_args("--dataset", "MOOC", "--batch-size", "4096"), seed=0
    )

    assert "dataset=MOOC" in prepared.command
    assert "macro_batch_size=4096" in prepared.command


def test_convergence_maps_wikipedia_for_pipetgl() -> None:
    args = _parser().parse_args(
        [
            "run",
            "--system",
            "pipetgl",
            "--dataset",
            "WIKIPEDIA",
            "--gpu-ids",
            "2,3",
            "--data-root",
            "/tmp/data",
            "--output",
            "/tmp/output",
        ]
    )

    prepared = _prepared_run(args, seed=4)

    assert prepared.env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert prepared.command[prepared.command.index("--data") + 1] == "WIKI"
    assert prepared.command[prepared.command.index("--batch-size") + 1] == "300"
    assert prepared.command[prepared.command.index("--seed") + 1] == "4"
    assert (
        prepared.command[prepared.command.index("--evaluation-negative-file") + 1]
        == "neg_pre_sampling_1_all_nodes_seed0.csv"
    )
