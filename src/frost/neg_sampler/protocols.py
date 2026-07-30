"""Runtime contract shared by every negative sampler."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class NegSamplerProto(Protocol):
    """Complete lifecycle used by the host, including optional-value hooks."""

    n_neg: int

    def init_from_df(self, df: Any) -> None: ...

    def reset(self) -> None: ...

    def set_batch_context(
        self,
        *,
        batch_src: np.ndarray,
        batch_dst: np.ndarray,
        batch_start_time: float,
        batch_end_time: float,
        batch_ts: np.ndarray | None = None,
    ) -> None: ...

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor: ...

    def get_last_negative_mask(self) -> torch.Tensor | None: ...
