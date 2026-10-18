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
    meta = SimpleNamespace(split="train", labels=torch.ones(1), n_src=1, n_neg=1)
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
        benchmark_observer=SimpleNamespace(
            on_train_batch_start=lambda _events: None,
            on_train_batch_end=lambda: None,
        ),
        split_benchmark_observer=SimpleNamespace(
            on_batch_start=lambda **_kwargs: None,
            on_batch_end=lambda: None,
        ),
        curr_epoch=0,
    )

    with torch.enable_grad():
        Worker.run(worker)

    assert observed == [True]
