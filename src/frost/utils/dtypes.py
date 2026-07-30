"""Fixed tensor dtypes shared by the data and runtime pipelines.

These are implementation invariants, not experiment parameters.  Keeping IDs
and timestamps integral until model-side relative-time math prevents large
absolute timestamps from losing ordering or low-order bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeDTypes:
    """Canonical storage and transport dtypes supported by FROST."""

    index_np: np.dtype
    index_torch: torch.dtype
    timestamp_np: np.dtype
    timestamp_torch: torch.dtype
    feature_torch: torch.dtype
    state_torch: torch.dtype


RUNTIME_DTYPES = RuntimeDTypes(
    index_np=np.dtype(np.int64),
    index_torch=torch.int64,
    timestamp_np=np.dtype(np.int64),
    timestamp_torch=torch.int64,
    feature_torch=torch.float32,
    state_torch=torch.float32,
)
