from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from frost.datatypes.graph import FlatGraph, HierGraph
from frost.datatypes.mem_mail_box.mail import MailEntry


@runtime_checkable
class MemoryUpdaterProto(Protocol):
    d_mail: int

    def __call__(
        self,
        flat_graph: FlatGraph,
        n_target_nodes: int,
        n_root_nodes: int | None = None,
    ) -> FlatGraph: ...


@runtime_checkable
class MailboxRouterProto(Protocol):
    def deliver(self, graph: HierGraph, mail_data: torch.Tensor) -> MailEntry: ...
