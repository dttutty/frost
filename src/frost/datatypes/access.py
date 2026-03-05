from __future__ import annotations

from dataclasses import dataclass

from frost.utils.tensor_set import TensorSet


@dataclass(kw_only=True)
class Access:
    mem_need: TensorSet | None = None
    mem_update: TensorSet | None = None
    box_need: TensorSet | None = None
    box_update: TensorSet | None = None
    nfeat_need: TensorSet | None = None

    def __repr__(self) -> str:
        return (
            f"Access(mem_needed={self.mem_need.data.shape}, "
            f"mem_updated={self.mem_update.data.shape}, "
            f"box_needed={self.box_need.data.shape if self.box_need is not None else None}, "
            f"box_updated={self.box_update.data.shape if self.box_update is not None else None}, "
            f"nfeat_needed={self.nfeat_need.data.shape})"
        )
