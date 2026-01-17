from enum import IntEnum, auto
class RunState(IntEnum):
    TRAIN = 0
    EXIT  = auto()
    SAVE  = auto()
    LOAD  = auto()
    REDUCE_LOSS = auto()
    SKIP  = auto()
