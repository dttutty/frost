from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class NghSamplerProto(Protocol):
    @property
    def fanout(self) -> Sequence[int] | None: ...

    @property
    def sample_4_neg(self) -> bool: ...

    def sample(
        self,
        root_nid: torch.Tensor,
        root_ts: torch.Tensor,
        sample_nid: torch.Tensor,
        sample_ts: torch.Tensor,
        pos_eid: torch.Tensor,
        gpu_id: int,
    ) -> tuple[torch.Tensor, int]: ...
