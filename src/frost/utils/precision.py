from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class PrecisionSpec:
    index_np: np.dtype
    index_torch: torch.dtype
    time_preprocessing_np: np.dtype
    time_preprocessing_torch: torch.dtype
    time_for_encoder_torch: torch.dtype
    feature_torch: torch.dtype
    memory_torch: torch.dtype
    mailbox_torch: torch.dtype


_NP_DTYPES: dict[str, np.dtype[Any]] = {
    "int64": np.dtype(np.int64),
    "float32": np.dtype(np.float32),
}

_TORCH_DTYPES = {
    "int64": torch.int64,
    "float32": torch.float32,
}


def _read_str(cfg: Mapping[str, Any], key: str) -> str:
    value = cfg.get(key)
    if value is None:
        nested = cfg.get("precision")
        if isinstance(nested, Mapping):
            value = nested.get(key)
    if not isinstance(value, str):
        raise TypeError(f"precision.{key} must be a string, got {type(value)!r}")
    return value


def resolve_precision(cfg: Mapping[str, Any]) -> PrecisionSpec:
    """Resolve and validate the precision config used by the runtime.

    The current implementation intentionally supports a single production
    precision layout:
    - indices: int64
    - timestamps before encoder math: int64
    - encoder-side time math: float32
    - nfeat / efeat / memory / mailbox payloads: float32
    """
    index_name = _read_str(cfg, "index")
    time_pre_name = _read_str(cfg, "time_for_preprocessing")
    time_encoder_name = _read_str(cfg, "time_for_encoder")
    payload_name = _read_str(cfg, "efeat_nfeat_mem_mailbox")

    if index_name != "int64":
        raise ValueError("precision.index currently must be 'int64'.")
    if time_pre_name != "int64":
        raise ValueError("precision.time_for_preprocessing currently must be 'int64'.")
    if time_encoder_name != "float32":
        raise ValueError("precision.time_for_encoder currently must be 'float32'.")
    if payload_name != "float32":
        raise ValueError(
            "precision.efeat_nfeat_mem_mailbox currently must be 'float32'."
        )

    return PrecisionSpec(
        index_np=_NP_DTYPES[index_name],
        index_torch=_TORCH_DTYPES[index_name],
        time_preprocessing_np=_NP_DTYPES[time_pre_name],
        time_preprocessing_torch=_TORCH_DTYPES[time_pre_name],
        time_for_encoder_torch=_TORCH_DTYPES[time_encoder_name],
        feature_torch=_TORCH_DTYPES[payload_name],
        memory_torch=_TORCH_DTYPES[payload_name],
        mailbox_torch=_TORCH_DTYPES[payload_name],
    )
