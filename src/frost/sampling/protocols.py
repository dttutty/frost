from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from frost.models.contracts import RootExpansion


@runtime_checkable
class NeighborhoodSamplerProto(Protocol):
    @property
    def fanout(self) -> tuple[int, ...]: ...

    @property
    def root_expansion(self) -> RootExpansion: ...

    def sample(
        self,
        root_nid: torch.Tensor,
        root_ts: torch.Tensor,
        sample_nid: torch.Tensor,
        sample_ts: torch.Tensor,
        pos_eid: torch.Tensor,
        gpu_id: int,
    ) -> tuple[torch.Tensor, int]: ...
