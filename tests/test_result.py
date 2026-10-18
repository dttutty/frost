"""Tests for the epoch result carrier."""

from frost.datatypes.result import Result


def test_result_carries_selection_and_report_metrics() -> None:
    result = Result(
        test_auc=0.85,
        test_ap=0.82,
        train_loss=0.35,
        val_auc=0.80,
        val_ap=0.78,
        val_mrr=0.75,
        test_mrr=0.72,
    )

    assert result.train_loss == 0.35
    assert result.val_ap == 0.78
    assert result.test_auc == 0.85
    assert result.val_mrr == 0.75
    assert result.test_mrr == 0.72


def test_result_defaults_mrr_metrics_for_ap_evaluation() -> None:
    result = Result(
        test_auc=0.9,
        test_ap=0.88,
        train_loss=0.1,
        val_auc=0.87,
        val_ap=0.85,
    )

    assert result.val_mrr == 0.0
    assert result.test_mrr == 0.0
