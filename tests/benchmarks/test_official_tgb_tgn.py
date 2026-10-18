from __future__ import annotations

import torch

from frost_bench.cli.official_tgb_tgn import (
    SUPPORTED_DATASETS,
    OPTIMIZED_SPEED_ENV,
    CorrectedLinkPredictor,
    _EvaluationResult,
    _make_negative_groups,
    _parser,
    _tgb_fast_path_enabled,
)


def test_official_tgb_tgn_defaults_match_requested_protocol() -> None:
    args = _parser().parse_args(["--dataset", "tgbl-wiki"])

    assert args.epochs == 30
    assert args.batch_size == 200
    assert args.seed == 1
    assert args.memory_dim == 100
    assert args.time_dim == 100
    assert args.embedding_dim == 100
    assert args.patience == 5
    assert args.tolerance == 1e-6
    assert args.loss_mode == "corrected"
    assert args.max_train_batches == 0
    assert args.max_val_batches == 0
    assert args.max_test_batches == 0
    assert args.train_throughput_warmup_batches == 0
    assert args.test_throughput_warmup_batches == 0


def test_corrected_predictor_returns_logits() -> None:
    predictor = CorrectedLinkPredictor(2)
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.zero_()
        predictor.lin_final.bias.fill_(2.0)

    score = predictor(torch.zeros(1, 2), torch.zeros(1, 2))

    assert score.item() == 2.0


def test_official_tgb_tgn_covers_download_catalog() -> None:
    assert set(SUPPORTED_DATASETS) == {
        "tgbl-wiki",
        "tgbl-coin",
        "tgbl-comment",
        "tgbl-flight",
        "tgbl-review",
    }


def test_official_tgb_tgn_prepare_only_flag() -> None:
    args = _parser().parse_args(["--dataset", "tgbl-flight", "--prepare-only"])

    assert args.prepare_only


def test_official_tgb_tgn_fast_path_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(OPTIMIZED_SPEED_ENV, raising=False)
    assert not _tgb_fast_path_enabled()

    monkeypatch.setenv(OPTIMIZED_SPEED_ENV, "true")
    assert _tgb_fast_path_enabled()

    monkeypatch.setenv(OPTIMIZED_SPEED_ENV, "0")
    assert not _tgb_fast_path_enabled()


def test_negative_groups_preserve_variable_width_candidates() -> None:
    groups = _make_negative_groups([[10, 11], [12], [13, 14]], torch.device("cpu"))

    assert [group.width for group in groups] == [1, 2]
    assert groups[0].indices.tolist() == [1]
    assert groups[0].destinations.tolist() == [[12]]
    assert groups[1].indices.tolist() == [0, 2]
    assert groups[1].destinations.tolist() == [[10, 11], [13, 14]]


def test_evaluation_result_reports_both_throughput_units() -> None:
    result = _EvaluationResult(
        mrr=0.5,
        queries=600,
        scored_pairs=60600,
        batches=3,
        measured_queries=400,
        measured_scored_pairs=40400,
        measured_batches=2,
        measured_elapsed_s=2.0,
    )

    assert result.queries_per_sec == 200.0
    assert result.scored_pairs_per_sec == 20200.0
