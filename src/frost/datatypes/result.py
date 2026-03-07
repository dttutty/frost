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
        val_loss: Mean validation loss over the epoch.
        val_auc: Area-under-ROC-curve on the validation split.
        val_ap: Average precision on the validation split.
    """

    test_auc: float
    test_ap: float
    train_loss: float
    val_loss: float
    val_auc: float
    val_ap: float
