from __future__ import annotations

from dataclasses import dataclass

from frost.utils.tensor_set import TensorSet


@dataclass(kw_only=True)
class Access:
    required_mem: TensorSet | None = None
    mem_update: TensorSet | None = None
    required_box: TensorSet | None = None
    box_update: TensorSet | None = None
    required_nfeat: TensorSet | None = None

    def __repr__(self) -> str:
        return (
            f"Access(mem_needed={self.required_mem.data.shape}, "
            f"mem_updated={self.mem_update.data.shape}, "
            f"box_needed={self.required_box.data.shape if self.required_box is not None else None}, "
            f"box_updated={self.box_update.data.shape if self.box_update is not None else None}, "
            f"nfeat_needed={self.required_nfeat.data.shape})"
        )
