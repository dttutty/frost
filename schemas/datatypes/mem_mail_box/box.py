from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from tensor_set_fn import TensorSetFn


if TYPE_CHECKING:
    from .mail import MailTsNid


def is_sorted(t: torch.Tensor) -> bool:
    return cast(bool, torch.all(torch.diff(t) >= 0).item())


@dataclass(kw_only=True)
class BoxTsNid:
    nid: Tensor
    data: Tensor
    ts: torch.Tensor
    next_pos: Tensor | None = None
    # if next_pos is None, then this BoxTsNid is actually a MailTsNid
    ct: Tensor | None = None  # 如果mean的话(每次只能累计一个mail)
    is_sorted: bool = False

    def __repr__(self) -> str:
        return (
            f"BoxTsNid(nid={self.nid.shape}, data={self.data.shape}, "
            f"ts={self.ts.shape}, is_sorted={self.is_sorted})"
        )

    def cuda(self) -> BoxTsNid:
        return BoxTsNid(
            nid=self.nid.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            next_pos=self.next_pos.cuda() if self.next_pos is not None else None,
            ct=self.ct.cuda() if self.ct is not None else None,
        )

    def merge_1_mail(self, new_mails: MailTsNid) -> BoxTsNid:
        assert self.next_pos is not None, "next_pos must be provided for merge_1_mail"
        if __debug__:
            assert is_sorted(self.nid)
            assert new_mails.nid.numel() == torch.unique(new_mails.nid).numel()
        idx = torch.searchsorted(self.nid, new_mails.nid)
        pos = self.next_pos[idx]
        self.data[idx, pos] = new_mails.data
        self.ts[idx, pos] = new_mails.ts
        self.next_pos[idx] = (self.next_pos[idx] + 1) % self.data.size(1)

        return self

    def merge_n_mail(self, new_mails: MailTsNid) -> BoxTsNid:
        raise NotImplementedError

    def keep_last_after_gather(self) -> BoxTsNid:
        return self

    def keep_mean_after_gather(self) -> BoxTsNid:
        return self

    def to_mail(self) -> MailTsNid:
        from .mail import MailTsNid

        assert self.next_pos is None, "next_pos must be None for mail"
        return MailTsNid(
            nid=self.nid,
            data=self.data.squeeze(1),
            ts=self.ts.squeeze(1),
            ct=self.ct,
        )

    def merge_box(self, other: BoxTsNid) -> BoxTsNid:
        assert TensorSetFn.intersection(self.nid, other.nid).numel() == 0, (
            "NID overlap between BoxTsNid instances."
        )
        if self.data.dim() == 3 and other.data.dim() == 2:
            other_data = other.data.unsqueeze(1)
        if self.ts.dim() == 2 and other.ts.dim() == 1:
            other_ts = other.ts.unsqueeze(1)
        self = BoxTsNid(
            nid=torch.cat((self.nid, other.nid), dim=0),
            data=torch.cat((self.data, other_data), dim=0),
            ts=torch.cat((self.ts, other_ts), dim=0),
            next_pos=None,
        )
        return self

    def sort_by_nid(self) -> BoxTsNid:
        sorted_indices = torch.argsort(self.nid)
        self.nid = self.nid[sorted_indices]
        self.data = self.data[sorted_indices]
        self.ts = self.ts[sorted_indices]
        if self.next_pos is not None:
            self.next_pos = self.next_pos[sorted_indices]
        self.is_sorted = True
        return self

    def to(self, device: torch.device) -> BoxTsNid:
        return BoxTsNid(
            nid=self.nid.to(device),
            data=self.data.to(device),
            ts=self.ts.to(device),
            next_pos=self.next_pos.to(device) if self.next_pos is not None else None,
            ct=self.ct.to(device) if self.ct is not None else None,
        )
