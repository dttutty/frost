from __future__ import annotations

from dataclasses import dataclass

import torch

from frost.utils.tensor_set_fn import TensorSetFn


@dataclass(kw_only=True)
class NodeFeatNid:
    nid: torch.Tensor
    nfeat: torch.Tensor
    is_sorted: bool = False

    def sort_by_nid(self) -> NodeFeatNid:
        if self.is_sorted:
            return self

        if self.nid is not None:
            sorted_indices = torch.argsort(self.nid)
            self.nid = self.nid[sorted_indices]

            if self.nfeat is not None:
                self.nfeat = self.nfeat[sorted_indices]

        self.is_sorted = True
        return self

    def merge(self, other: NodeFeatNid) -> NodeFeatNid:
        assert TensorSetFn.intersection(self.nid, other.nid).numel() == 0, (
            "NID overlap between RootX instances."
        )
        if TensorSetFn.intersection(self.nid, other.nid).numel() != 0:
            raise ValueError("NID overlap between RootX instances.")
        self.nid = torch.cat((self.nid, other.nid), dim=0)
        self.nfeat = torch.cat((self.nfeat, other.nfeat), dim=0)
        return self

    def filter_by_nid(self, nid: torch.Tensor) -> NodeFeatNid:
        mask = torch.isin(self.nid, nid)
        nid = self.nid[mask]
        nfeat = self.nfeat[mask]
        return self.__class__(nid=nid, nfeat=nfeat)

    def cuda(self) -> NodeFeatNid:
        return NodeFeatNid(nid=self.nid.cuda(), nfeat=self.nfeat.cuda())
