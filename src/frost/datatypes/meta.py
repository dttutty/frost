from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import torch

from frost.datatypes.access import Access
from frost.datatypes.graph import FlatGraph, Graph

from .plan import PlanSet


@dataclass(kw_only=True)
class Meta:
    cur_epoch: int
    cur_macro_of_split: int
    n_macro_of_split: int
    cur_macro_of_epoch: int
    split: Literal["train", "val", "test"]
    n_src: int
    batch_idx: int
    flat_graph: FlatGraph | None
    buf_len: int
    graph: Graph | None = None
    plan_set: PlanSet | None = None
    access: Access | None = None
    labels: torch.Tensor | None = None
    fanout: list[int] | None = None

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d.pop("access", None)
        return d

    @cached_property
    def macro_position_of_split(self) -> str | None:
        if self.cur_macro_of_split + 1 == self.n_macro_of_split:
            return "last"
        elif self.cur_macro_of_split == 0:
            return "first"
        else:
            return None

    @cached_property
    def is_last_macro_of_epoch(self) -> bool:
        return self.macro_position_of_split == "last" and self.split == "test"
