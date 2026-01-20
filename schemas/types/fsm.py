from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class WorkerState(Enum):
    IDLE = auto()
    WAIT_META = auto()
    PREFETCH = auto()
    COMPUTE = auto()
    APPLY_UPDATE = auto()
    SAVE = auto()
    STOPPED = auto()


class WorkerEvent(Enum):
    META_READY = auto()
    SAVE_MODEL = auto()
    STOP = auto()
    ERROR = auto()


@dataclass(slots=True)
class StepResult:
    next_state: WorkerState
