from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import torch
from dgl.heterograph import DGLBlock
from numpy.typing import NDArray
from torch import Tensor

from tensor_set import TensorSet

from ..alias import IntArr
from .data_to_sync import BoxTsNid, MemTsNid
from .plan import BatchNodeStateIO, PlanSet, TensorPlanSet


@dataclass(slots=True, kw_only=True)
class SampleReturn:
    ngh2root_mfgs_raw: list[DGLBlockMeta] | None
    root2ngh_mfg_raw: DGLBlockMeta | None
    batch_node_state_io: BatchNodeStateIO


@dataclass(slots=True)
class ForwardCtx:
    blockss: list[list[DGLBlock]]
    mem_n_box: MemNBox | None
    macro_bid: int
    plan_set: TensorPlanSet
    mail_delivery_tensor: Tensor | None
    is_last: bool
    n_root_src_n_pos_dst: int
    root_info: X | None


@dataclass(frozen=True, slots=True)
class ExtFullGraph:
    indptr: IntArr
    indices: IntArr
    eid: IntArr
    ts: NDArray[np.float32]  # 你后面会 astype(float32)，那这里就直接约束成 float32

    @property
    def num_nodes(self) -> int:
        # indptr 长度 = num_nodes + 1
        return int(self.indptr.shape[0] - 1)

    @property
    def num_edges(self) -> int:
        return int(self.indices.shape[0])

    @staticmethod
    def load(path: str) -> ExtFullGraph:
        npz = np.load(path)
        return ExtFullGraph(
            indptr=npz["indptr"].astype(np.int64, copy=False),
            indices=npz["indices"].astype(np.int64, copy=False),
            eid=npz["eid"].astype(np.int64, copy=False),
            ts=npz["ts"].astype(np.float32, copy=False),
        )


@dataclass(slots=True, kw_only=True)
class ScatterResult:
    m_id: int
    total_sample_time: float | None = None
    ap: list[float] | None = None
    auc: list[float] | None = None


@dataclass(slots=True, kw_only=True)
class NodeFeat:
    id: Tensor
    data: Tensor

    def cuda(self) -> NodeFeat:
        return NodeFeat(id=self.id.cuda(), data=self.data.cuda())


@dataclass(slots=True, kw_only=True)
class EdgeFeat:
    id: Tensor
    data: Tensor

    def cuda(self) -> EdgeFeat:
        return EdgeFeat(id=self.id.cuda(), data=self.data.cuda())


@dataclass(slots=True, kw_only=True)
class ModelComponents:
    memory_updater: torch.nn.Module | None
    gnn: torch.nn.Module
    header: torch.nn.Module | None


@dataclass(slots=True, kw_only=True)
class NFeatNid:
    nid: torch.Tensor
    nfeat: torch.Tensor
    is_zipped: bool = True
    is_sorted: bool = False

    def sort_by_nid(self) -> NFeatNid:
        if self.is_sorted:
            return self

        if self.nid is not None:
            sorted_indices = torch.argsort(self.nid)
            self.nid = self.nid[sorted_indices]

            if self.nfeat is not None:
                self.nfeat = self.nfeat[sorted_indices]

        self.is_sorted = True
        return self

    def merge(self, other: NFeatNid) -> NFeatNid:
        assert TensorSet.intersection(self.nid, other.nid).numel() == 0, (
            "NID overlap between RootX instances."
        )
        self.nid = torch.cat((self.nid, other.nid), dim=0)
        self.nfeat = torch.cat((self.nfeat, other.nfeat), dim=0)
        return self

    def unzip_by_root_nid(self, root_nid: torch.Tensor) -> NFeatNid:
        self = self.sort_by_nid()
        pos = torch.searchsorted(self.nid, root_nid)
        self.nid = root_nid

        self.nfeat = self.nfeat[pos]
        self.is_zipped = False
        return self.__class__(nid=self.nid, nfeat=self.nfeat)

    def filter_by_nid(self, nid: torch.Tensor) -> NFeatNid:
        mask = torch.isin(self.nid, nid)
        nid = self.nid[mask]
        nfeat = self.nfeat[mask]
        return self.__class__(nid=nid, nfeat=nfeat)


def unzip_and_align(mem_n_box: MemNBox, delta_x: MemNBox, root_nid: torch.Tensor) -> X:
    if delta_x.mem is not None:
        delta_x = delta_x.sort_mem_and_mail()
        pos = torch.searchsorted(delta_x.mem.nid, root_nid)
        delta_x.mem.ts = delta_x.mem.ts[pos]
        delta_x.mem.data = delta_x.mem.data[pos]
    if delta_x.box is not None:
        delta_x.box = delta_x.box[pos]
    if delta_x.box_ts is not None:
        delta_x.box_ts = delta_x.box_ts[pos]

    delta_x.is_zipped = False
    return delta_x


@dataclass(slots=True, kw_only=True)
class MemNBox:
    mem: MemTsNid
    box: BoxTsNid

    @cached_property
    def is_mem_mail_sorted(self) -> bool:
        assert self.mem is not None
        assert self.box is not None
        return self.mem.is_sorted and self.box.is_sorted

    def sort_mem_and_mail(self) -> MemNBox:
        if self.is_mem_mail_sorted:
            return self

        if self.mem is not None:
            self.mem = self.mem.sort_by_nid()

            assert self.box is not None
            self.box = self.box.sort_by_nid()

        return self


@dataclass(slots=True, kw_only=True)
class MetaList:
    queue: list[Meta]

    def __len__(self) -> int:
        return len(self.queue)


@dataclass(slots=True, kw_only=True)
class Meta:
    root_nid: Tensor
    root_ts: Tensor  # 没有3倍处理,这个信息不需要传输到gpu

    pos_edge_id: Tensor  # 没有3倍处理,这个信息不需要传输到gpu
    pos_efeat: Tensor | None = None

    gnn_mfgs_raw: list[DGLBlockMeta] | None
    gnn_mfgs: list[list[DGLBlock]] | None = None

    mdb_raw_data: DGLBlockMeta | None

    epoch_split_iter_info: EpochSplitIterInfo
    nfeat: NFeatNid | None = None
    mem_n_box: MemNBox | None = None
    batch_node_state_io: BatchNodeStateIO | None
    n_src: int = 1

    plan_set: TensorPlanSet | None = None

    @cached_property
    def pos_edge(self) -> torch.Tensor:
        return self.root_nid[: self.n_src * 2].reshape(-1, 2)

    @cached_property
    def pos_edge_ts(self) -> torch.Tensor:
        return self.root_ts[: self.n_src * 2].reshape(-1, 2)

    @cached_property
    def uniq_root(self) -> torch.Tensor:
        return torch.unique(self.pos_edge)

    @cached_property
    def uniq_all(self) -> torch.Tensor | None:
        if self.gnn_mfgs_raw is not None:
            all_nodes = set()
            for block in self.gnn_mfgs_raw:
                all_nodes.update(block.nodes.tolist())
            uniq_all = torch.as_tensor(list(all_nodes), dtype=torch.long)
        elif self.mdb_raw_data is not None:
            uniq_all = torch.as_tensor(
                np.unique(self.mdb_raw_data.nodes), dtype=torch.long
            )
        else:
            uniq_all = self.uniq_root
        return uniq_all

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d.pop("pos_edge", None)
        d.pop("pos_edge_ts", None)
        d.pop("uniq_root", None)
        d.pop("uniq_all", None)
        d.pop("batch_node_state_io", None)
        return d


@dataclass(slots=True, kw_only=True)
class ScatterData:
    metas: list[Meta]
    planset: PlanSet | TensorPlanSet | None = None


@dataclass(slots=True, kw_only=True)
class EpochSplitIterInfo:
    cur_epoch: int
    n_epoch: int

    cur_macro_of_split: int
    n_macro_of_split: int

    cur_macro_of_epoch: int
    n_macro_of_epoch: None = None

    split: Literal["train", "val", "test"]

    @cached_property
    def macro_position_of_split(self) -> str | None:
        if self.cur_macro_of_split + 1 == self.n_macro_of_split:
            return "last"
        elif self.cur_macro_of_split == 0:
            return "first"
        else:
            return None

    @cached_property
    def macro_lication_of_epoch(self) -> str | None:
        if self.cur_macro_of_epoch + 1 == self.n_macro_of_epoch:
            return "last"
        elif self.cur_macro_of_epoch == 0:
            return "first"
        else:
            return None

    @cached_property
    def macro_lication_of_all_epochs(self) -> str | None:
        if (
            self.cur_epoch + 1 == self.n_epoch
            and self.cur_macro_of_epoch + 1 == self.n_macro_of_epoch
        ):
            return "last"
        elif self.cur_epoch == 0 and self.cur_macro_of_epoch == 0:
            return "first"
        else:
            return None


@dataclass(slots=True, kw_only=True)
class Shared:
    """一个专门存放所有共享内存张量的数据类"""

    mem: Tensor
    mem_ts: Tensor
    box: Tensor
    box_ts: Tensor
    box_next_pos: Tensor | None = None

    def reset(self) -> None:
        """将所有共享内存张量清零"""
        self.mem.zero_()
        self.mem_ts.zero_()
        self.box.zero_()
        self.box_ts.zero_()


@dataclass(frozen=True, kw_only=True)
class DGLBlockMeta:
    nodes: torch.Tensor
    ts: torch.Tensor
    eid: torch.Tensor
    dts: torch.Tensor
    col: torch.Tensor
    row: torch.Tensor
    dim_in: int
    dim_out: int
