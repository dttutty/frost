from __future__ import annotations

from typing import Protocol, runtime_checkable

from frost.datatypes.graph import FlatGraph


@runtime_checkable
class MemoryUpdaterProto(Protocol):
    d_mail: int

    def __call__(
        self,
        flat_graph: FlatGraph,
        n_target_nodes: int,
        n_root_nodes: int | None = None,
    ) -> FlatGraph: ...
