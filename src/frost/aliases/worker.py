from __future__ import annotations

from queue import Queue
from typing import Any, Literal, TypeAlias

from frost.datatypes.meta import Meta
from frost.datatypes.tasks import P2PTask
from frost.protocols.cuda_event_proto import CudaEvent

MetaWithEvent: TypeAlias = tuple[Meta, CudaEvent]
DeltaPayload: TypeAlias = tuple[P2PTask | None, P2PTask | None, list[Any]]

ReceivedQ: TypeAlias = Queue[Meta | None | Literal["end_of_epoch"]]
DeltaQ: TypeAlias = Queue[DeltaPayload | Literal["nothing"] | None]
H2DOutQ: TypeAlias = Queue[MetaWithEvent | None | Literal["end_of_epoch"]]
AssembledQ: TypeAlias = Queue[MetaWithEvent | None | Literal["end_of_epoch"]]
