import json

import pytest
import torch

from frost.runtime.worker import _split_benchmark_observer as observer_module


class _FakeEvent:
    timestamps = iter((1000.0, 3000.0))

    def __init__(self, *, enable_timing: bool) -> None:
        assert enable_timing
        self.timestamp = next(self.timestamps)

    def record(self, _stream) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return other.timestamp - self.timestamp


class _Env:
    local_rank = 0


def test_split_observer_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("FROST_BENCHMARK_SPLIT_THROUGHPUT", raising=False)
    observer = observer_module.WorkerSplitBenchmarkObserver(
        env=_Env(), device=torch.device("cuda:0")
    )
    observer.on_batch_start(split="val", epoch=1, events=200, scored_pairs=400)
    observer.on_batch_end()
    observer.on_epoch_end()
    assert capsys.readouterr().out == ""


def test_split_observer_emits_events_per_second(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FROST_BENCHMARK_SPLIT_THROUGHPUT", "1")
    _FakeEvent.timestamps = iter((1000.0, 3000.0))
    monkeypatch.setattr(observer_module.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(
        observer_module.torch.cuda,
        "current_stream",
        lambda _device: object(),
    )
    observer = observer_module.WorkerSplitBenchmarkObserver(
        env=_Env(), device=torch.device("cuda:0")
    )
    observer.on_batch_start(split="val", epoch=2, events=200, scored_pairs=20200)
    observer.on_batch_end()
    observer.on_batch_start(split="val", epoch=2, events=100, scored_pairs=10100)
    observer.on_batch_end()
    observer.on_epoch_end()

    line = capsys.readouterr().out.strip()
    prefix, encoded = line.split(" ", 1)
    assert prefix == "FROST_SPLIT_THROUGHPUT"
    payload = json.loads(encoded)
    assert payload == {
        "batches": 2,
        "elapsed_s": 2.0,
        "epoch": 2,
        "events": 300,
        "events_per_sec": 150.0,
        "rank": 0,
        "scored_pairs": 30300,
        "scored_pairs_per_sec": 15150.0,
        "split": "val",
        "system": "frost",
        "warmup_batches": 0,
    }


def test_split_observer_excludes_warmup_batches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FROST_BENCHMARK_SPLIT_THROUGHPUT", "1")
    monkeypatch.setenv("FROST_SPLIT_THROUGHPUT_WARMUP_BATCHES_TEST", "1")
    _FakeEvent.timestamps = iter((2000.0, 3000.0))
    monkeypatch.setattr(observer_module.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(
        observer_module.torch.cuda,
        "current_stream",
        lambda _device: object(),
    )
    observer = observer_module.WorkerSplitBenchmarkObserver(
        env=_Env(), device=torch.device("cuda:0")
    )
    observer.on_batch_start(split="test", epoch=1, events=200, scored_pairs=20200)
    observer.on_batch_end()
    observer.on_batch_start(split="test", epoch=1, events=100, scored_pairs=10100)
    observer.on_batch_end()
    observer.on_epoch_end()

    _, encoded = capsys.readouterr().out.strip().split(" ", 1)
    payload = json.loads(encoded)
    assert payload["warmup_batches"] == 1
    assert payload["batches"] == 1
    assert payload["events"] == 100
    assert payload["scored_pairs"] == 10100
    assert payload["elapsed_s"] == 1.0


def test_invalid_split_benchmark_env_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FROST_BENCHMARK_SPLIT_THROUGHPUT", "sometimes")
    with pytest.raises(ValueError, match="FROST_BENCHMARK_SPLIT_THROUGHPUT"):
        observer_module.WorkerSplitBenchmarkObserver(
            env=_Env(), device=torch.device("cuda:0")
        )
