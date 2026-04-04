"""Tests for Result training metrics datatype."""

from frost.datatypes.result import Result


class TestResult:
    def test_basic_creation(self):
        result = Result(
            test_auc=0.85,
            test_ap=0.82,
            train_loss=0.35,
            val_loss=0.42,
            val_auc=0.80,
            val_ap=0.78,
        )
        assert result.test_auc == 0.85
        assert result.test_ap == 0.82
        assert result.train_loss == 0.35
        assert result.val_loss == 0.42
        assert result.val_auc == 0.80
        assert result.val_ap == 0.78

    def test_default_mrr_values(self):
        result = Result(
            test_auc=0.9,
            test_ap=0.88,
            train_loss=0.1,
            val_loss=0.15,
            val_auc=0.87,
            val_ap=0.85,
        )
        assert result.val_mrr == 0.0
        assert result.test_mrr == 0.0
        assert result.val_auc_batch_mean == 0.0
        assert result.val_ap_batch_mean == 0.0
        assert result.test_auc_batch_mean == 0.0
        assert result.test_ap_batch_mean == 0.0

    def test_with_mrr_values(self):
        result = Result(
            test_auc=0.9,
            test_ap=0.88,
            train_loss=0.1,
            val_loss=0.15,
            val_auc=0.87,
            val_ap=0.85,
            val_mrr=0.75,
            test_mrr=0.72,
        )
        assert result.val_mrr == 0.75
        assert result.test_mrr == 0.72

    def test_with_tgl_style_batch_mean_values(self):
        result = Result(
            test_auc=0.9,
            test_ap=0.88,
            train_loss=0.1,
            val_loss=0.15,
            val_auc=0.87,
            val_ap=0.85,
            val_auc_batch_mean=0.81,
            val_ap_batch_mean=0.82,
            test_auc_batch_mean=0.83,
            test_ap_batch_mean=0.84,
        )
        assert result.val_auc_batch_mean == 0.81
        assert result.val_ap_batch_mean == 0.82
        assert result.test_auc_batch_mean == 0.83
        assert result.test_ap_batch_mean == 0.84

    def test_mutable_instance(self):
        result = Result(
            test_auc=0.8,
            test_ap=0.7,
            train_loss=0.5,
            val_loss=0.6,
            val_auc=0.5,
            val_ap=0.5,
        )
        result.test_auc = 0.99
        assert result.test_auc == 0.99
