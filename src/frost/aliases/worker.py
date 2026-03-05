from __future__ import annotations

from queue import Queue
from typing import Any, Literal, TypeAlias  # 必须导入这个

from frost.datatypes.meta import Meta
from frost.datatypes.tasks import P2PTask
from frost.protocols.cuda_event_proto import CudaEvent

MetaWEvent: TypeAlias = tuple[Meta, CudaEvent]

ReceivedQ: TypeAlias = Queue[Meta | None | Literal["end_of_epoch"]]
DeltaQ: TypeAlias = Queue[tuple[P2PTask, P2PTask, list[Any]] | Literal["nothing"]]
H2DOutQ: TypeAlias = Queue[MetaWEvent | None | Literal["end_of_epoch"]]
FullQ: TypeAlias = Queue[MetaWEvent | None | Literal["end_of_epoch"]]
