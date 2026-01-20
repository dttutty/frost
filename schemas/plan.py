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
    mem: Plan
    box: Plan | None

    def tensorfy(self):
        mem = self.mem.tensorfy()

        return TensorPlanSet(
            mem=mem,
        )


@dataclass(slots=True, kw_only=True)
class CudaPlanSet:
    mem: CudaPlan


@dataclass(slots=True, kw_only=True)
class TensorPlanSet:
    mem: TensorPlan

    def cuda(self):
        if self.mem is not None:
            assert isinstance(self.mem, TensorPlan)
            mem = self.mem.cuda()
        return CudaPlanSet(
            mem=mem,
        )

    def sort(self):
        if self.mem is not None:
            assert isinstance(self.mem, TensorPlan)
            self.mem = self.mem.sort()
        return self


@dataclass(slots=True, kw_only=True)
class ApanPlanSet:
    mem: Plan
    box: Plan

    def tensorfy(self):
        mem = self.mem.tensorfy()
        box = self.box.tensorfy

        return ApanTensorPlanSet(mem=mem, box=box)


@dataclass(slots=True, kw_only=True)
class ApanTensorPlanSet:
    mem: TensorPlan
    box: TensorPlan

    def cuda(self):
        if self.mem is not None:
            assert isinstance(self.mem, TensorPlan)
            mem = self.mem.cuda()
        if self.box is not None:
            assert isinstance(self.box, TensorPlan)
            box = self.box.cuda()
        return ApanCudaPlanSet(
            mem=mem,
            box=box,
        )

    def sort(self):
        if self.mem is not None:
            assert isinstance(self.mem, TensorPlan)
            self.mem = self.mem.sort()
        if self.box is not None:
            assert isinstance(self.box, TensorPlan)
            self.box = self.box.sort()
        return self


@dataclass(slots=True, kw_only=True)
class ApanCudaPlanSet:
    mem: CudaPlan
    box: CudaPlan


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

    def cuda(self):
        return CudaPlan(
            p2p=[s.cuda() for s in self.p2p],
            prefetch=self.prefetch.cuda(),
            rec_cts=self.rec_cts,
            send_cts=self.send_cts,
            writeback=(self.writeback.cuda() if self.writeback is not None else None),
        )

    def sort(self):
        self.prefetch, _ = torch.sort(self.prefetch)
        self.p2p = [torch.sort(s).values for s in self.p2p]
        if self.writeback is not None:
            self.writeback, _ = torch.sort(self.writeback)
        return self


@dataclass(slots=True, kw_only=True)
class CudaPlan:
    p2p: List[torch.Tensor]
    prefetch: torch.Tensor
    rec_cts: NDArray
    send_cts: NDArray
    writeback: Optional[torch.Tensor] = None
