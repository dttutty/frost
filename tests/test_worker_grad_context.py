from __future__ import annotations

import threading
from queue import Queue
from types import SimpleNamespace

import torch

from frost.runtime.worker._engine import Worker


class _FakeCudaEvent:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def record(self, stream=None) -> None:
        del stream


class _FakeCudaStream:
    def wait_event(self, event) -> None:
        del event


def test_worker_run_keeps_grad_enabled_for_training(monkeypatch) -> None:
    """The worker dispatcher must not wrap training in ``torch.no_grad``."""
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeCudaStream())

    stop_ev = threading.Event()
    full_q: Queue = Queue()
    meta = SimpleNamespace(split="train", labels=torch.ones(1), n_src=1)
    full_q.put((meta, _FakeCudaEvent()))

    observed: list[bool] = []

    def train(_meta) -> None:
        observed.append(torch.is_grad_enabled())
        stop_ev.set()

    worker = SimpleNamespace(
        start_all_threads=lambda: None,
        mem_box_store=None,
        stop_ev=stop_ev,
        full_q=full_q,
        train=train,
        _throughput_epoch_train_batch_count=0,
        _throughput_total_train_batch_count=0,
        _throughput_measure_batches_target=1,
        _throughput_measured_batches=0,
        _throughput_warmup_batches_total=None,
        _throughput_warmup_batches_per_epoch=4,
        _throughput_prev_start_event=None,
        _throughput_prev_start_wall_s=None,
        _throughput_prev_start_wall_unix_s=None,
        _throughput_prev_batch_events=None,
        _throughput_gpu_pairs=[],
        _throughput_wall_intervals_s=[],
        _throughput_events=0,
        _throughput_window_start_unix_s=None,
        _throughput_window_end_unix_s=None,
        _throughput_interval_start_unix_s=[],
        _throughput_interval_end_unix_s=[],
        _cpu_util_enabled=False,
        _memory_reserved_mib=[],
        _memory_allocated_mib=[],
    )

    with torch.enable_grad():
        Worker.run(worker)

    assert observed == [True]
