from __future__ import annotations

from queue import Queue
from typing import Any, Literal, TypeAlias, TypeVar

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.meta import Meta
from frost.datatypes.tasks import BoxTask, MemTask, P2PTask
from frost.runtime.protocols import CudaEvent


MetaWithEvent: TypeAlias = tuple[Meta, CudaEvent]
DeltaPayload: TypeAlias = tuple[P2PTask | None, P2PTask | None, list[Any]]
ReceivedQ: TypeAlias = Queue[Meta | None | Literal["end_of_epoch"]]
DeltaQueue: TypeAlias = Queue[DeltaPayload | Literal["nothing"] | None]
H2DOutQ: TypeAlias = Queue[MetaWithEvent | None | Literal["end_of_epoch"]]
AssembledQ: TypeAlias = Queue[MetaWithEvent | None | Literal["end_of_epoch"]]

MemTaskQ: TypeAlias = Queue[tuple[MemTask, CudaEvent] | None]
BoxTaskQ: TypeAlias = Queue[tuple[BoxTask, CudaEvent] | None]
MemP2PQ: TypeAlias = Queue[tuple[P2PTask, CudaEvent] | Literal["nothing_to_p2p"] | None]
BoxP2PQ: TypeAlias = Queue[tuple[P2PTask, CudaEvent] | Literal["nothing_to_p2p"] | None]
MemWrQ: TypeAlias = Queue[tuple[torch.Tensor | None, int, MemEntry, CudaEvent] | None]
BoxWriteEntry: TypeAlias = (
    BoxEntry | tuple[MailEntry, torch.Tensor] | tuple[MailEntry, torch.Tensor, int]
)
BoxWrQ: TypeAlias = Queue[
    tuple[torch.Tensor | None, int, BoxWriteEntry, CudaEvent] | None
]

AllType = TypeVar("AllType", MemEntry, MailEntry, BoxEntry)
