from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

import frost.launcher as launcher
import frost.runtime.worker._trainer as trainer_module
from frost.runtime.worker._engine import Worker


class _Tracker:
    def __init__(self, value):
        self.value = value
        self.reset_called = False

    def compute(self):
        return self.value

    def reset(self) -> None:
        self.reset_called = True


class _MetricStub:
    instances = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def cuda(self):
        return self


class _MetricCollectionStub:
    def __init__(self, metrics) -> None:
        self.metrics = metrics

    def cuda(self):
        return self


@pytest.mark.parametrize(("n_gpu", "expected_sync"), [(1, False), (2, True)])
def test_metric_sync_is_disabled_for_a_single_gpu_worker(
    monkeypatch, n_gpu: int, expected_sync: bool
) -> None:
    metric_types = []
    for name in ("MeanMetric", "BinaryAUROC", "BinaryAveragePrecision"):
        metric_type = type(name, (_MetricStub,), {"instances": []})
        metric_types.append(metric_type)
        monkeypatch.setattr(trainer_module, name, metric_type)
    monkeypatch.setattr(trainer_module, "MetricCollection", _MetricCollectionStub)

    worker = Worker.__new__(Worker)
    worker.env = SimpleNamespace(n_gpu=n_gpu)
    worker.nccl = object()
    worker._init_metrics()

    instances = [
        instance for metric_type in metric_types for instance in metric_type.instances
    ]
    assert len(instances) == 7
    assert all(instance.kwargs["process_group"] is worker.nccl for instance in instances)
    assert all(
        instance.kwargs["sync_on_compute"] is expected_sync for instance in instances
    )


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


class _Sampler:
    def __init__(self) -> None:
        self.initialized_with = None

    def init_from_df(self, dataframe) -> None:
        self.initialized_with = dataframe


def test_train_only_launch_does_not_instantiate_eval_sampler(monkeypatch) -> None:
    dataframe = pd.DataFrame({"dst": [1, 2, 1]})
    train_sampler = _Sampler()
    instantiate_calls = []

    def fake_instantiate(config, **kwargs):
        instantiate_calls.append((config, kwargs))
        return train_sampler

    monkeypatch.setattr(launcher, "instantiate", fake_instantiate)
    sampler_cfg = OmegaConf.create(
        {
            "seed": 0,
            "train": {"_target_": "tests.TrainSampler"},
            "eval": {"_target_": "tests.MissingEvalSampler"},
        }
    )

    train, evaluation = launcher._instantiate_negative_samplers(
        sampler_cfg=sampler_cfg,
        dataframe=dataframe,
        n_nodes=3,
        dataset="large",
        train_n_neg=1,
        eval_n_neg=1,
        train_only=True,
    )

    assert len(instantiate_calls) == 1
    assert train is train_sampler
    assert evaluation is train_sampler
    assert train_sampler.initialized_with is dataframe
