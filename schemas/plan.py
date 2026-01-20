from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import torch
from numpy.typing import NDArray


@dataclass(slots=True, kw_only=True)
class BatchNodeStateIO:
    mem_needed: Set[int] | None | torch.Tensor
    mem_updated: Set[int] | None | torch.Tensor
    mailbox_needed: Set[int] | None | torch.Tensor
    mailbox_updated: Set[int] | None | torch.Tensor
    nfeat_needed: Set[int] | None | torch.Tensor


@dataclass(slots=True, kw_only=True)
class BatchNodeStateIOAPAN:
    # this is for mailboxsize=1
    mem_needed: Set[int] | None | torch.Tensor
    mem_updated: Set[int] | None | torch.Tensor
    mailbox_needed: Set[int] | None | torch.Tensor
    # 并不是所有节点都需要弄到GPU上,只有mailbox_needed - mailbox_updated的节点信息才需要全部搬到GPU上
    mailbox_updated: Set[int] | None | torch.Tensor
    nfeat_needed: Set[int] | None | torch.Tensor


@dataclass(slots=True, kw_only=True)
class PlanSet:
    mem: Plan | None
    box: Plan | None
    nfeat: Plan | None

    def tensorfy(self) -> "TensorPlanSet":
        return TensorPlanSet(
            mem=self.mem.tensorfy() if self.mem is not None else None,
            box=self.box.tensorfy() if self.box is not None else None,
            nfeat=self.nfeat.tensorfy() if self.nfeat is not None else None,
        )


@dataclass(slots=True, kw_only=True)
class TensorPlanSet:
    mem: TensorPlan | None = None
    box: TensorPlan | None = None
    nfeat: TensorPlan | None = None

    def to(self, device: torch.device) -> "TensorPlanSet":
        return self.__class__(
            mem=self.mem.to(device) if self.mem is not None else None,
            box=self.box.to(device) if self.box is not None else None,
            nfeat=self.nfeat.to(device) if self.nfeat is not None else None,
        )

    def cuda(self) -> "TensorPlanSet":
        return self.to(torch.device("cuda"))

    def sort(self) -> "TensorPlanSet":
        return self.__class__(
            mem=self.mem.sort() if self.mem is not None else None,
            box=self.box.sort() if self.box is not None else None,
            nfeat=self.nfeat.sort() if self.nfeat is not None else None,
        )


@dataclass(slots=True, kw_only=True)
class Plan:
    p2p: List[Set[int]]
    prefetch: Set[int]
    rec_cts: NDArray
    send_cts: NDArray
    writeback: Optional[Set[int]] = None

    def tensorfy(self) -> "TensorPlan":
        return TensorPlan(
            p2p=[torch.tensor(list(s)).long() for s in self.p2p],
            prefetch=torch.tensor(list(self.prefetch)).long(),
            rec_cts=self.rec_cts,
            send_cts=self.send_cts,
            writeback=(
                torch.tensor(list(self.writeback)).long()
                if self.writeback is not None
                else None
            ),
        )


def set_to_tensor(s: Set[int]) -> torch.Tensor:
    return torch.tensor(list(s)).long()


@dataclass(slots=True, kw_only=True)
class TensorPlan:
    p2p: List[torch.Tensor]
    prefetch: torch.Tensor
    rec_cts: NDArray
    send_cts: NDArray
    writeback: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "TensorPlan":
        return self.__class__(
            p2p=[s.to(device) for s in self.p2p],
            prefetch=self.prefetch.to(device),
            rec_cts=self.rec_cts,
            send_cts=self.send_cts,
            writeback=(
                self.writeback.to(device) if self.writeback is not None else None
            ),
        )

    def cuda(self) -> "TensorPlan":
        return self.to(torch.device("cuda"))

    def sort(self) -> "TensorPlan":
        new_prefetch, _ = torch.sort(self.prefetch)
        new_p2p = [torch.sort(s).values for s in self.p2p]
        if self.writeback is not None:
            new_writeback, _ = torch.sort(self.writeback)
        return self.__class__(
            p2p=new_p2p,
            prefetch=new_prefetch,
            rec_cts=self.rec_cts,  # 不需要排序，直接引用
            send_cts=self.send_cts,  # 不需要排序，直接引用
            writeback=new_writeback,
        )

    @property
    def device(self):
        return self.prefetch.device
