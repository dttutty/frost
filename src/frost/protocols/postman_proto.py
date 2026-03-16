from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from frost.datatypes.graph import Graph
from frost.datatypes.mem_mail_box.mail import MailEntry


@runtime_checkable
class PostmanProto(Protocol):
    def deliver(self, graph: Graph, mail_data: torch.Tensor) -> MailEntry: ...
