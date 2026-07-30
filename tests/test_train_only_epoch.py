import torch

from frost.runtime.worker._engine import Worker


class _Tracker:
    def __init__(self, value):
        self.value = value
        self.reset_called = False

    def compute(self):
        return self.value

    def reset(self) -> None:
        self.reset_called = True


def test_train_only_epoch_does_not_compute_empty_evaluation_metrics() -> None:
    worker = Worker.__new__(Worker)
    worker.train_only = True
    worker.eval_metric = "ap"
    worker.train_loss_tracker = _Tracker(torch.tensor(1.25))
    worker.val_metrics = _Tracker(None)
    worker.test_metrics = _Tracker(None)
    worker.val_mrr_tracker = _Tracker(None)
    worker.test_mrr_tracker = _Tracker(None)

    result = worker.on_train_epoch_end()

    assert result.train_loss == 1.25
    assert result.val_ap == 0.0
    assert result.test_ap == 0.0
    assert worker.val_metrics.reset_called
    assert worker.test_metrics.reset_called
