from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from frost.datatypes.access import Access


@runtime_checkable
class AnalyzerProto(Protocol):
    n_neg: int

    def count_access(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        all_nid: torch.Tensor,
        fanout: list[int] | None,
    ) -> Access: ...
