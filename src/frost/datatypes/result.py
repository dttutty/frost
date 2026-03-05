from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class Result:
    test_auc: float
    test_ap: float
    train_loss: float
    val_loss: float
    val_auc: float
    val_ap: float
