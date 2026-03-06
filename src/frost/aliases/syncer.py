from __future__ import annotations

from queue import Queue
from typing import Literal, TypeAlias, TypeVar  # 必须导入这个

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mail import MailEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.tasks import BoxTask, MemTask, P2PTask
from frost.protocols.cuda_event_proto import CudaEvent

MemTaskQ: TypeAlias = Queue[tuple[MemTask, CudaEvent] | None]
BoxTaskQ: TypeAlias = Queue[tuple[BoxTask, CudaEvent] | None]
MemP2PQ: TypeAlias = Queue[tuple[P2PTask, CudaEvent] | Literal["nothing_to_p2p"] | None]
BoxP2PQ: TypeAlias = Queue[tuple[P2PTask, CudaEvent] | Literal["nothing_to_p2p"]]
MemWrQ: TypeAlias = Queue[tuple[torch.Tensor | None, int, MemEntry, CudaEvent] | None]
BoxWrQ: TypeAlias = Queue[tuple[torch.Tensor | None, int, BoxEntry, CudaEvent] | None]

AllType = TypeVar("AllType", MemEntry, MailEntry, BoxEntry)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemEntry", "MailEntry")
