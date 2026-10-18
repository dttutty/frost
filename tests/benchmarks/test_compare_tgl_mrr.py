from pathlib import Path

import pytest

import frost_bench.cli.compare_tgl_mrr as compare_tgl_mrr
from frost_bench.cli.compare_tgl_mrr import build_parser, prepare_runs


def test_prepare_runs_uses_matched_twenty_negative_mrr_protocol(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        [
            "--dataset",
            "WIKIPEDIA",
            "--epochs",
            "50",
            "--eval-neg-samples",
            "20",
        ]
    )

    frost, tgl = prepare_runs(args, tmp_path)

    assert "neg_sampler=rnd_dst_then_random" in frost.command
    assert "neg_sampler.eval.n_neg=20" in frost.command
    assert "eval_metric=mrr" in frost.command
    assert "reset_eval_neg_sampler_each_epoch=true" in frost.command
    assert tgl.command[tgl.command.index("--eval-metric") + 1] == "mrr"
    assert tgl.command[tgl.command.index("--eval-neg-samples") + 1] == "20"
    assert "--appendix-train-e2e" not in tgl.command


def test_prepare_runs_uses_official_tgbl_wiki_negatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    negative_dir = tmp_path / "raw" / "tgbl_wiki"
    monkeypatch.setattr(
        compare_tgl_mrr,
        "resolve_tgb_raw_dataset_dir",
        lambda _dataset: negative_dir,
    )
    args = build_parser().parse_args(
        [
            "--dataset",
            "TGBL_WIKI",
            "--epochs",
            "5",
            "--batch-size",
            "4000",
        ]
    )

    frost, tgl = prepare_runs(args, tmp_path)

    assert "neg_sampler=rnd_dst_then_tgb_wiki" in frost.command
    assert "eval_macro_batch_size=128" in frost.command
    assert "neg_sampler=rnd_dst_then_random" not in frost.command
    assert tgl.command[tgl.command.index("--eval-neg-samples") + 1] == "999"
    assert tgl.command[tgl.command.index("--eval-negative-dir") + 1] == str(
        negative_dir
    )
    assert tgl.command[tgl.command.index("--eval-batch-size") + 1] == "128"
    assert tgl.env["TGL_DATA_ROOT"] == str(tmp_path / "tgl_data")
