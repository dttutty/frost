"""Datatype representing per-batch feature and memory access requirements."""

from __future__ import annotations

from dataclasses import dataclass

from frost.utils.tensor_set import TensorSet


@dataclass(kw_only=True)
class Access:
    """Describes the set of nodes whose features, memory, and mailbox must be
    fetched or written back for a single training batch.

    Attributes:
        required_mem: Node IDs whose memory vectors must be read before the
            forward pass.
        mem_update: Node IDs whose memory vectors will be updated after the
            forward pass.
        required_box: Node IDs whose mailbox entries must be read before the
            forward pass.  ``None`` when the model has no mailbox.
        box_update: Node IDs whose mailbox entries will be written after the
            forward pass.  ``None`` when the model has no mailbox.
        required_nfeat: Node IDs whose static node features must be loaded.
    """

    required_mem: TensorSet | None = None
    mem_update: TensorSet | None = None
    required_box: TensorSet | None = None
    box_update: TensorSet | None = None
    required_nfeat: TensorSet | None = None

    def __repr__(self) -> str:
        """Return a concise string showing tensor shapes for each field."""
        return (
            f"Access(mem_needed={self.required_mem.data.shape}, "
            f"mem_updated={self.mem_update.data.shape}, "
            f"box_needed={self.required_box.data.shape if self.required_box is not None else None}, "
            f"box_updated={self.box_update.data.shape if self.box_update is not None else None}, "
            f"nfeat_needed={self.required_nfeat.data.shape})"
        )
