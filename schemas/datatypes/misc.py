from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Literal, Optional

import numpy as np
import torch
from dgl.heterograph import DGLBlock
from numpy.typing import NDArray
from torch import Tensor

from ..alias import IntArr
from .data_to_sync import BoxTsNid, MemTsNid
from .plan import BatchNodeStateIO, PlanSet, TensorPlanSet


@dataclass(slots=True, kw_only=True)
class SampleReturn:
    all_nodes: set | None = None
    root_nodes: set | None = None
    ngh2root_mfg: DGLBlockMeta | None
    root2ngh_mfg: DGLBlockMeta | None
    batch_node_state_io: BatchNodeStateIO | None


@dataclass(slots=True)
class ForwardCtx:
    blockss: List[List[DGLBlock]]
    x: X
    macro_bid: int
    plan_set: TensorPlanSet
    mail_delivery_tensor: Tensor | None
    is_last: bool
    n_root_src_n_pos_dst: int
    root_info: Roots | None


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
    def load(path: str) -> "ExtFullGraph":
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
    total_sample_time: Optional[float] = None
    ap: Optional[List[float]] = None
    auc: Optional[List[float]] = None


@dataclass(slots=True, kw_only=True)
class NodeFeat:
    id: Tensor
    data: Tensor

    def cuda(self) -> "NodeFeat":
        return NodeFeat(id=self.id.cuda(), data=self.data.cuda())


@dataclass(slots=True, kw_only=True)
class EdgeFeat:
    id: Tensor
    data: Tensor

    def cuda(self) -> "EdgeFeat":
        return EdgeFeat(id=self.id.cuda(), data=self.data.cuda())


@dataclass(slots=True)
class X:
    nid: Tensor | None = None
    pos_efeat: Optional[Tensor] = None
    mem: MemTsNid | None = None
    box: BoxTsNid | None = None
    is_sorted: bool = False

    def sort_by_nid(self) -> "X":
        if self.is_sorted:
            return self

        if self.mem is not None:
            sorted_indices = torch.argsort(self.mem.id)
            self.mem.id = self.mem.id[sorted_indices]
            self.mem.data = self.mem.data[sorted_indices]
            self.mem.ts = self.mem.ts[sorted_indices]

        if self.box is not None:
            sorted_indices = torch.argsort(self.box.id)
            self.box.id = self.box.id[sorted_indices]
            self.box.data = self.box.data[sorted_indices]
            self.box.ts = self.box.ts[sorted_indices]
            if self.box.next_pos is not None:
                self.box.next_pos = self.box.next_pos[sorted_indices]

        # check if nids are consistent
        nid_for_compare = None

        if self.mem is not None:
            nid_for_compare = self.mem.id
        # if self.box is not None:
        #     if nid_for_compare is not None:
        #         assert torch.equal(nid_for_compare, self.box.id), (
        #             "NID mismatch between nfeat/mem and box_or_mail"
        #         )
        #     else:
        #         nid_for_compare = self.box.id

        if nid_for_compare is not None:
            self.nid = nid_for_compare
        self.is_sorted = True
        return self


@dataclass(slots=True, kw_only=True)
class MetaList:
    queue: List[Meta]

    def __len__(self) -> int:
        return len(self.queue)


@dataclass(slots=True, kw_only=True)
class Meta:
    roots: Roots
    pos_edge_id: Tensor  # 没有3倍处理,这个信息不需要传输到gpu

    gnn_mfgs_raw: List[DGLBlockMeta] | None
    gnn_mfgs: List[List[DGLBlock]] | None = None

    mdb_raw_data: DGLBlockMeta | None

    epoch_split_iter_info: EpochSplitIterInfo
    x: Optional[X] = None
    batch_node_state_io: Optional[BatchNodeStateIO]
    n_src: int = 1

    plan_set: Optional[TensorPlanSet] = None

    @cached_property
    def pos_edge(self) -> torch.Tensor:
        return self.roots.nid[: self.n_src * 2].reshape(-1, 2)

    @cached_property
    def pos_edge_ts(self) -> torch.Tensor:
        return self.roots.ts[: self.n_src * 2].reshape(-1, 2)

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


@dataclass(slots=True, kw_only=True)
class Roots:
    nid: torch.Tensor
    ts: torch.Tensor
    x: RootX


@dataclass(slots=True, kw_only=True)
class RootX:
    mem: torch.Tensor | None = None
    mem_ts: torch.Tensor | None = None
    box: torch.Tensor | None = None
    box_ts: torch.Tensor | None = None
    nfeat: torch.Tensor | None = None


@dataclass(slots=True, kw_only=True)
class ScatterData:
    metas: List[Meta]
    planset: Optional[PlanSet | TensorPlanSet] = None


@dataclass(slots=True, kw_only=True)
class EpochSplitIterInfo:
    cur_epoch: int
    n_epoch: int

    cur_macro_of_split: int
    n_macro_of_split: int

    cur_macro_of_epoch: int
    n_macro_of_epoch: None = None

    split: Literal["train", "val", "test"]

    @property
    def position_in_split(self) -> str | None:
        if self.cur_macro_of_split + 1 == self.n_macro_of_split:
            return "last"
        elif self.cur_macro_of_split == 0:
            return "first"
        else:
            return None

    @property
    def position_in_epoch(self) -> str | None:
        if self.cur_epoch + 1 == self.n_epoch:
            return "last"
        elif self.cur_epoch == 0:
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
