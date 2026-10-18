"""Tests for the fixed runtime dtype contract."""

import numpy as np
import pytest
import torch

from frost.utils.dtypes import RUNTIME_DTYPES


def test_runtime_dtypes_preserve_indices_and_timestamps() -> None:
    assert RUNTIME_DTYPES.index_np == np.dtype(np.int64)
    assert RUNTIME_DTYPES.index_torch == torch.int64
    assert RUNTIME_DTYPES.timestamp_np == np.dtype(np.int64)
    assert RUNTIME_DTYPES.timestamp_torch == torch.int64


def test_runtime_payloads_use_float32() -> None:
    assert RUNTIME_DTYPES.feature_torch == torch.float32
    assert RUNTIME_DTYPES.state_torch == torch.float32


def test_runtime_dtypes_are_immutable() -> None:
    with pytest.raises(AttributeError):
        RUNTIME_DTYPES.index_torch = torch.int32
