"""Training run result summary datatype."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class Result:
    """Aggregated metrics produced at the end of one training run.

    Attributes:
        test_auc: Area-under-ROC-curve on the test split.
        test_ap: Average precision on the test split.
        train_loss: Mean training loss over the epoch.
        val_auc: Area-under-ROC-curve on the validation split.
        val_ap: Average precision on the validation split.
        val_mrr: TGB-compatible validation mean reciprocal rank when enabled.
        test_mrr: TGB-compatible test mean reciprocal rank when enabled.
    """

    test_auc: float
    test_ap: float
    train_loss: float
    val_auc: float
    val_ap: float
    val_mrr: float = 0.0
    test_mrr: float = 0.0
