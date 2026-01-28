from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import torch
from dgl.heterograph import DGLBlock
from numpy.typing import NDArray
from torch import Tensor

from gnn.components import EdgePredictor
from gnn.memory_updater import (
    MfgMemoryUpdater,
    NoMfgMemoryUpdater,
    TransformerMemoryUpdater,
)
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from tensor_set_fn import TensorSetFn

from ..alias import IntArr
from .plan import BatchNodeStateIO, PlanSet, TensorPlanSet


@dataclass(slots=True, kw_only=True)
class SampleReturn:
    ngh2root_mfgs_raw: list[DGLBlockMeta] | None
    root2ngh_mfg_raw: DGLBlockMeta | None
    batch_node_state_io: BatchNodeStateIO


@dataclass(slots=True)
class ForwardCtx:
    blockss: list[list[DGLBlock]] | NecessrayData
    macro_bid: int
    plan_set: TensorPlanSet
    mail_delivery_tensor: Tensor | None
    is_last: bool
    n_root_src_n_pos_dst: int


@dataclass(slots=True, kw_only=True)
class NecessrayData:
    nfeat: NFeatNid | None
    mem_n_box: MemNBox | None


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
    memory_updater: (
        MfgMemoryUpdater | NoMfgMemoryUpdater | TransformerMemoryUpdater | None
    )
    gnn: torch.nn.Module
    header: EdgePredictor | None


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
        assert TensorSetFn.intersection(self.nid, other.nid).numel() == 0, (
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

    def cuda(self) -> NFeatNid:
        return NFeatNid(nid=self.nid.cuda(), nfeat=self.nfeat.cuda())


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

    def retain_and_align_to_root(self, root_nid: torch.Tensor) -> MemNBox:
        self = self.sort_mem_and_mail()
        pos = torch.searchsorted(self.mem.nid, root_nid)
        new_mem_ts_nid = MemTsNid(
            nid=root_nid, ts=self.mem.ts[pos], data=self.mem.data[pos]
        )
        pos = torch.searchsorted(self.box.nid, root_nid)
        new_box_ts_nid = BoxTsNid(
            nid=root_nid, ts=self.box.ts[pos], data=self.box.data[pos]
        )

        uniq_ngh_nid = TensorSetFn.difference(self.box.nid, root_nid)
        uniq_ngh_nid_pos = torch.searchsorted(self.box.nid, uniq_ngh_nid)
        ngh_box_ts_nid = BoxTsNid(
            nid=uniq_ngh_nid,
            ts=self.box.ts[uniq_ngh_nid_pos],
            data=self.box.data[uniq_ngh_nid_pos],
        )

        combined_box = BoxTsNid(
            nid=torch.cat((new_box_ts_nid.nid, ngh_box_ts_nid.nid)),
            ts=torch.cat((new_box_ts_nid.ts, ngh_box_ts_nid.ts)),
            data=torch.cat((new_box_ts_nid.data, ngh_box_ts_nid.data)),
        )

        return self.__class__(mem=new_mem_ts_nid, box=combined_box)


@dataclass(slots=True, kw_only=True)
class MetaList:
    queue: list[Meta]

    def __len__(self) -> int:
        return len(self.queue)


@dataclass(slots=True, kw_only=True)
class Result:
    train_loss: float
    val_loss: float
    val_auc: float
    val_ap: float
    test_auc: float
    test_ap: float


@dataclass(slots=True, kw_only=True)
class Meta:
    root_nid: Tensor
    root_ts: Tensor  # 没有3倍处理,这个信息不需要传输到gpu

    pos_edge_id: Tensor  # 没有3倍处理,这个信息不需要传输到gpu
    pos_efeat: Tensor | None = None

    n2r_meta: list[DGLBlockMeta] | None
    n2r_mfgs: list[list[DGLBlock]] | None = None

    r2n_meta: DGLBlockMeta | None
    r2n_mdt: Tensor | None = None

    m_state: EpochSplitIterInfo
    nfeat: NFeatNid | None = None
    mem_n_box: MemNBox | None = None
    root_mem_n_box: SeperateData | None = None
    state: BatchNodeStateIO | None
    n_src: int = 1

    plan_set: TensorPlanSet | None = None
    labels: Tensor | None = None

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
        if self.n2r_meta is not None:
            all_nodes = set()
            for block in self.n2r_meta:
                all_nodes.update(block.nodes.tolist())
            uniq_all = torch.as_tensor(list(all_nodes), dtype=torch.long)
        elif self.r2n_meta is not None:
            uniq_all = torch.as_tensor(np.unique(self.r2n_meta.nodes), dtype=torch.long)
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
class SeperateData:
    root_mem: MemTsNid
    root_mail: MemNBox
    ngh_mail: BoxTsNid | None = None


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
    def is_last_macro_of_epoch(self) -> bool:
        return self.macro_position_of_split == "last" and self.split == "test"

    @cached_property
    def is_last_macro_of_all_epoch(self) -> bool:
        return (
            self.cur_epoch + 1 == self.n_epoch
            and self.macro_position_of_split == "last"
        )


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
