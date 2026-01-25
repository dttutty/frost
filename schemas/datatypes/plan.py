from __future__ import annotations

from dataclasses import dataclass

import torch
from numpy.typing import NDArray


def set_to_tensor(s: set[int]) -> torch.Tensor:
    return torch.tensor(list(s)).long()


@dataclass(slots=True, kw_only=True)
class BatchNodeStateIO:
    mem_needed: set[int]
    mem_updated: set[int]
    mailbox_needed: set[int]
    mailbox_updated: set[int]
    nfeat_needed: set[int]


@dataclass(slots=True, kw_only=True)
class PlanSet:
    mem: DyPlan | None
    box: DyPlan | None
    nfeat: StPlan | None

    def tensorfy(self) -> TensorPlanSet:
        return TensorPlanSet(
            mem=self.mem.tensorfy() if self.mem is not None else None,
            box=self.box.tensorfy() if self.box is not None else None,
            nfeat=self.nfeat.tensorfy() if self.nfeat is not None else None,
        )


@dataclass(slots=True, kw_only=True)
class TensorPlanSet:
    """
    PlanSet = DyPlan + StPlan
    """

    mem: TensorDyPlan | None = None
    box: TensorDyPlan | None = None
    nfeat: TensorStPlan | None = None

    def to(self, device: torch.device) -> TensorPlanSet:
        return self.__class__(
            mem=self.mem.to(device) if self.mem is not None else None,
            box=self.box.to(device) if self.box is not None else None,
            nfeat=self.nfeat.to(device) if self.nfeat is not None else None,
        )

    def cuda(self) -> TensorPlanSet:
        return self.to(torch.device("cuda"))

    def sort(self) -> TensorPlanSet:
        return self.__class__(
            mem=self.mem.sort() if self.mem is not None else None,
            box=self.box.sort() if self.box is not None else None,
            nfeat=self.nfeat.sort() if self.nfeat is not None else None,
        )


@dataclass(slots=True, kw_only=True)
class DyPlan:
    p2p: list[set[int]]
    fetch: set[int]
    rec_cts: NDArray
    send_cts: NDArray
    writeback: set[int] | None = None

    def tensorfy(self) -> TensorDyPlan:
        return TensorDyPlan(
            p2p=[torch.tensor(list(s)).long() for s in self.p2p],
            fetch=torch.tensor(list(self.fetch)).long(),
            rec_cts=self.rec_cts,
            send_cts=self.send_cts,
            writeback=(
                torch.tensor(list(self.writeback)).long()
                if self.writeback is not None
                else None
            ),
        )


@dataclass(slots=True, kw_only=True)
class StPlan:
    to_cache: set[int]
    fetch: set[int]

    def tensorfy(self) -> TensorStPlan:
        return TensorStPlan(
            to_cache=torch.tensor(list(self.to_cache)).long(),
            fetch=torch.tensor(list(self.fetch)).long(),
        )


@dataclass(slots=True, kw_only=True)
class TensorDyPlan:
    p2p: list[torch.Tensor]
    fetch: torch.Tensor
    rec_cts: NDArray
    send_cts: NDArray
    writeback: torch.Tensor | None = None

    def to(self, device: torch.device) -> TensorDyPlan:
        return self.__class__(
            p2p=[s.to(device) for s in self.p2p],
            fetch=self.fetch.to(device),
            rec_cts=self.rec_cts,
            send_cts=self.send_cts,
            writeback=(
                self.writeback.to(device) if self.writeback is not None else None
            ),
        )

    def cuda(self) -> TensorDyPlan:
        return self.to(torch.device("cuda"))

    def sort(self) -> TensorDyPlan:
        new_fetch, _ = torch.sort(self.fetch)
        new_p2p = [torch.sort(s).values for s in self.p2p]
        if self.writeback is not None:
            new_writeback, _ = torch.sort(self.writeback)
        return self.__class__(
            p2p=new_p2p,
            fetch=new_fetch,
            rec_cts=self.rec_cts,  # 不需要排序，直接引用
            send_cts=self.send_cts,  # 不需要排序，直接引用
            writeback=new_writeback,
        )

    @property
    def device(self):
        return self.fetch.device


@dataclass(slots=True, kw_only=True)
class TensorStPlan:
    to_cache: torch.Tensor
    fetch: torch.Tensor

    def to(self, device: torch.device) -> TensorStPlan:
        return self.__class__(
            to_cache=self.to_cache.to(device),
            fetch=self.fetch.to(device),
        )

    def cuda(self) -> TensorStPlan:
        return self.to(torch.device("cuda"))

    def sort(self) -> TensorStPlan:
        new_to_cache, _ = torch.sort(self.to_cache)
        new_fetch, _ = torch.sort(self.fetch)
        return self.__class__(
            to_cache=new_to_cache,
            fetch=new_fetch,
        )

    @property
    def device(self):
        return self.to_cache.device
