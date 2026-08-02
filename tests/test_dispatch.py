from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from frost.launch import dispatch


class _FailingWorker:
    def __init__(self) -> None:
        self.cleanup = Mock()

    def run(self) -> None:
        raise RuntimeError("worker boom")


class _FailingHost:
    def __init__(self) -> None:
        self.shutdown = Mock()

    def run(self) -> None:
        raise RuntimeError("host boom")


@pytest.mark.parametrize(
    ("entrypoint", "runtime", "cleanup_name", "message"),
    [
        (dispatch.dispatch_worker, _FailingWorker(), "cleanup", "Worker runtime"),
        (dispatch.dispatch_host, _FailingHost(), "shutdown", "Host runtime"),
    ],
)
def test_dispatch_logs_originating_error_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    entrypoint: object,
    runtime: object,
    cleanup_name: str,
    message: str,
) -> None:
    monkeypatch.setattr(dispatch, "cleanup_all_shm", Mock())

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        entrypoint(lambda: runtime)  # type: ignore[operator]

    getattr(runtime, cleanup_name).assert_called_once_with(graceful=False)
    assert message in caplog.text
