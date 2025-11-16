# common/state.py
from enum import Enum


class WorkerState(Enum):
    EXIT = -1
    TRAIN = 0
    VAL = 1
    SAVE_MODEL = 2
    LOAD_MODEL = 3
    PASS = 4
    REDUCE_LOSS = 5
    BARRIER = 6
    CLEANUP = 7
    TEST = 8
