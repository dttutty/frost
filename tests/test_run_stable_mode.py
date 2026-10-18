import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import _apply_stable_mode_env


def test_apply_stable_mode_env_sets_expected_defaults(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("FROST_STABLE_MODE", raising=False)

    applied = _apply_stable_mode_env(OmegaConf.create({"stable_mode": True}))

    assert applied == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }
    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["FROST_STABLE_MODE"] == "1"


def test_apply_stable_mode_env_preserves_existing_env(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.delenv("FROST_STABLE_MODE", raising=False)

    applied = _apply_stable_mode_env(OmegaConf.create({"stable_mode": True}))

    assert applied == {}
    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "8"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert os.environ["FROST_STABLE_MODE"] == "1"


def test_apply_stable_mode_env_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("FROST_STABLE_MODE", raising=False)

    applied = _apply_stable_mode_env(OmegaConf.create({"stable_mode": False}))

    assert applied == {}
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert "FROST_STABLE_MODE" not in os.environ
