from __future__ import annotations

import threading
from queue import Queue
from types import SimpleNamespace

import torch

from frost.runtime.control_plane import ControlMessageKind
from frost.runtime.worker import _receiver
from frost.runtime.worker._receiver import Receiver


class _ReusingControlPlane:
    def __init__(self) -> None:
        self.shared_payload = torch.tensor([1, 2, 3], dtype=torch.int64)
        self.released = False

    def receive(self):
        return ControlMessageKind.BATCH, self.shared_payload

    def release(self) -> None:
        self.released = True
        self.shared_payload.fill_(99)


def test_receiver_detaches_payload_before_releasing_shared_slot(monkeypatch):
    control_plane = _ReusingControlPlane()
    captured: dict[str, torch.Tensor] = {}

    def decode(payload: torch.Tensor, *, n_gpus: int):
        assert n_gpus == 1
        captured["payload"] = payload
        return None

    monkeypatch.setattr(_receiver, "decode_meta", decode)
    received_q = Queue()
    receiver = Receiver(
        env=SimpleNamespace(n_gpu=1, local_rank=0, host_rank=1),
        received_q=received_q,
        result_q=Queue(),
        stop_ev=threading.Event(),
        ring_buf=object(),
        control_plane=control_plane,
    )

    receiver.logic()

    assert control_plane.released
    assert control_plane.shared_payload.tolist() == [99, 99, 99]
    assert captured["payload"].tolist() == [1, 2, 3]
    assert captured["payload"].data_ptr() != control_plane.shared_payload.data_ptr()
    assert received_q.get_nowait() is None
