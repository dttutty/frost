"""Tests for PrecisionSpec and resolve_precision."""

import numpy as np
import torch
import pytest

from frost.utils.precision import PrecisionSpec, resolve_precision


class TestPrecisionSpec:
    def test_basic_creation(self):
        spec = PrecisionSpec(
            index_np=np.dtype(np.int64),
            index_torch=torch.int64,
            time_preprocessing_np=np.dtype(np.int64),
            time_preprocessing_torch=torch.int64,
            time_for_encoder_torch=torch.float32,
            feature_torch=torch.float32,
            memory_torch=torch.float32,
            mailbox_torch=torch.float32,
        )
        assert spec.index_np == np.dtype(np.int64)
        assert spec.index_torch == torch.int64
        assert spec.time_for_encoder_torch == torch.float32

    def test_frozen(self):
        spec = PrecisionSpec(
            index_np=np.dtype(np.int64),
            index_torch=torch.int64,
            time_preprocessing_np=np.dtype(np.int64),
            time_preprocessing_torch=torch.int64,
            time_for_encoder_torch=torch.float32,
            feature_torch=torch.float32,
            memory_torch=torch.float32,
            mailbox_torch=torch.float32,
        )
        with pytest.raises(AttributeError):
            spec.index_torch = torch.int32


class TestResolvePrecision:
    def test_valid_config(self):
        cfg = {
            "index": "int64",
            "time_for_preprocessing": "int64",
            "time_for_encoder": "float32",
            "efeat_nfeat_mem_mailbox": "float32",
        }
        spec = resolve_precision(cfg)
        assert spec.index_np == np.dtype(np.int64)
        assert spec.index_torch == torch.int64
        assert spec.time_preprocessing_np == np.dtype(np.int64)
        assert spec.time_preprocessing_torch == torch.int64
        assert spec.time_for_encoder_torch == torch.float32
        assert spec.feature_torch == torch.float32
        assert spec.memory_torch == torch.float32
        assert spec.mailbox_torch == torch.float32

    def test_nested_config(self):
        cfg = {
            "precision": {
                "index": "int64",
                "time_for_preprocessing": "int64",
                "time_for_encoder": "float32",
                "efeat_nfeat_mem_mailbox": "float32",
            }
        }
        spec = resolve_precision(cfg)
        assert spec.index_torch == torch.int64

    def test_missing_key_raises(self):
        cfg = {
            "index": "int64",
            "time_for_preprocessing": "int64",
            "time_for_encoder": "float32",
        }
        with pytest.raises(TypeError):
            resolve_precision(cfg)

    def test_non_string_value_raises(self):
        cfg = {
            "index": 123,
            "time_for_preprocessing": "int64",
            "time_for_encoder": "float32",
            "efeat_nfeat_mem_mailbox": "float32",
        }
        with pytest.raises(TypeError):
            resolve_precision(cfg)

    def test_invalid_index_raises(self):
        cfg = {
            "index": "int32",
            "time_for_preprocessing": "int64",
            "time_for_encoder": "float32",
            "efeat_nfeat_mem_mailbox": "float32",
        }
        with pytest.raises(ValueError):
            resolve_precision(cfg)

    def test_invalid_time_preprocessing_raises(self):
        cfg = {
            "index": "int64",
            "time_for_preprocessing": "float32",
            "time_for_encoder": "float32",
            "efeat_nfeat_mem_mailbox": "float32",
        }
        with pytest.raises(ValueError):
            resolve_precision(cfg)

    def test_invalid_time_encoder_raises(self):
        cfg = {
            "index": "int64",
            "time_for_preprocessing": "int64",
            "time_for_encoder": "int64",
            "efeat_nfeat_mem_mailbox": "float32",
        }
        with pytest.raises(ValueError):
            resolve_precision(cfg)

    def test_invalid_payload_raises(self):
        cfg = {
            "index": "int64",
            "time_for_preprocessing": "int64",
            "time_for_encoder": "float32",
            "efeat_nfeat_mem_mailbox": "float64",
        }
        with pytest.raises(ValueError):
            resolve_precision(cfg)
