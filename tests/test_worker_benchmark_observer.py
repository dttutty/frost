from types import SimpleNamespace

import torch

from frost.runtime.worker import _benchmark_observer as observer_module
from frost.runtime.worker._benchmark_observer import WorkerBenchmarkObserver


class _Event:
    def record(self, stream) -> None:
        del stream

    def elapsed_time(self, other) -> float:
        del other
        return 100.0


def test_observer_owns_training_telemetry_lifecycle(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH", "0")
    monkeypatch.setenv("FROST_THROUGHPUT_MEASURE_BATCHES", "1")
    perf_values = iter([1.0, 2.0])
    unix_values = iter([10.0, 11.0])
    monkeypatch.setattr(observer_module.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(observer_module.time, "time", lambda: next(unix_values))
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: _Event())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 2 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 1024**2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    observer = WorkerBenchmarkObserver(
        env=SimpleNamespace(local_rank=0),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    observer.on_train_batch_start(8)
    observer.on_train_batch_end()
    observer.on_train_batch_start(6)
    observer.on_train_batch_end()
    observer.on_epoch_end()
    observer.finalize()

    output = capsys.readouterr().out
    assert observer.measured_events == 8
    assert observer.measured_batches == 1
    assert observer.reserved_mib == [2.0]
    assert "TRAIN_THROUGHPUT_PER_GPU" in output
    assert "APPENDIX_TRAIN_WINDOW" in output
    assert "APPENDIX_TRAIN_MEMORY_SUMMARY" in output


def test_observer_skips_complete_epochs_before_measurement(monkeypatch) -> None:
    monkeypatch.setenv("FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH", "0")
    monkeypatch.setenv("FROST_THROUGHPUT_MEASURE_START_EPOCH", "1")
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: _Event())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)
    observer = WorkerBenchmarkObserver(
        env=SimpleNamespace(local_rank=0),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    observer.on_train_batch_start(8)
    observer.on_train_batch_end()
    observer.on_train_batch_start(8)
    observer.on_train_batch_end()
    observer.on_epoch_end()
    assert observer.measured_batches == 0

    observer.on_train_batch_start(8)
    observer.on_train_batch_end()
    observer.on_train_batch_start(8)
    observer.on_train_batch_end()

    assert observer.measured_batches == 1
    assert observer.measured_events == 8
