from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from frost.utils.tensor_set import TensorSet


@runtime_checkable
class NghSamplerProto(Protocol):
    fanout: list[int]

    def sample(
        self,
        nid: torch.Tensor,
        ts: torch.Tensor,
        pos_eid: torch.Tensor,
        gpu_id: int,
    ) -> tuple[TensorSet, int]: ...
