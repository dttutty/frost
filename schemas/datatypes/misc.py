from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from dgl.heterograph import DGLBlock
from numpy.typing import NDArray
from torch import Tensor

from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from tensor_set_fn import TensorSetFn

from .plan import BatchNodeStateIO, PackedPlanSet, PlanSet


if TYPE_CHECKING:
    from gnn.components import EdgePredictor
    from gnn.with_memory.memory_updater import (
        MfgMemoryUpdater,
        NoMfgMemoryUpdater,
        TransformerMemoryUpdater,
    )

    from ..alias import IntArr


@dataclass(kw_only=True)
class SampleReturn:
    ngh2root_mfgs_raw: list[DGLBlockMeta] | None
    root2ngh_mfg_raw: DGLBlockMeta | None
    batch_node_state_io: BatchNodeStateIO


@dataclass(slots=True)
class ForwardCtx:
    blockss: list[list[DGLBlock]] | NecessrayData
    macro_bid: int
    plan_set: PlanSet
    mail_delivery_tensor: Tensor | None
    is_last: bool
    n_root_src_n_pos_dst: int


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class ScatterResult:
    m_id: int
    total_sample_time: float | None = None
    ap: list[float] | None = None
    auc: list[float] | None = None


@dataclass(kw_only=True)
class NodeFeat:
    id: Tensor
    data: Tensor

    def cuda(self) -> NodeFeat:
        return NodeFeat(id=self.id.cuda(), data=self.data.cuda())


@dataclass(kw_only=True)
class EdgeFeat:
    id: Tensor
    data: Tensor

    def cuda(self) -> EdgeFeat:
        return EdgeFeat(id=self.id.cuda(), data=self.data.cuda())


@dataclass(kw_only=True)
class ModelComponents:
    memory_updater: (
        MfgMemoryUpdater | NoMfgMemoryUpdater | TransformerMemoryUpdater | None
    )
    gnn: torch.nn.Module
    header: EdgePredictor | None


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class Result:
    test_auc: float
    test_ap: float
    train_loss: float
    val_loss: float
    val_auc: float
    val_ap: float


@dataclass(kw_only=True)
class Meta:
    root_nid: Tensor | None
    root_ts: Tensor | None  # 没有3倍处理,这个信息不需要传输到gpu

    pos_edge_id: Tensor  # 没有3倍处理,这个信息不需要传输到gpu
    pos_efeat: Tensor | None = None

    n2r_meta: list[DGLBlockMeta] | None
    n2r_meta_packed: PackedDGLBlockMeta | None = None
    n2r_mfgs: list[list[DGLBlock]] | None = None

    r2n_meta: DGLBlockMeta | None
    r2n_meta_packed: PackedDGLBlockMeta | None = None
    r2n_mdt: Tensor | None = None

    m_state: EpochSplitIterInfo
    nfeat: NFeatNid | None = None
    mem_n_box: MemNBox | None = None
    root_mem_n_box: SeperateData | None = None
    state: BatchNodeStateIO | None
    n_src: int | None

    shm_lens: dict[str, int] | None = None
    batch_idx: int = 0

    plan_set: PlanSet | None = None
    packed_plan_set: PackedPlanSet | None = None
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


# misc.py


@dataclass(kw_only=True)
class SeperateData:
    root_mem: MemTsNid
    root_mail: MemNBox
    ngh_mail: BoxTsNid | None = None


@dataclass(kw_only=True)
class ScatterData:
    metas: list[Meta]
    planset: PlanSet | PlanSet | None = None


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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

    @staticmethod
    def pack_list(blocks: list[DGLBlockMeta]) -> PackedDGLBlockMeta:
        if not blocks:
            empty = torch.tensor([], dtype=torch.long)
            # 处理空情况，sections 用空 numpy 数组
            return PackedDGLBlockMeta(
                nodes_cat=empty,
                ts_cat=empty,
                eid_cat=empty,
                dts_cat=empty,
                col_cat=empty,
                row_cat=empty,
                node_sections=np.array([], dtype=np.int64),
                edge_sections=np.array([], dtype=np.int64),
                dims=np.array([], dtype=np.int64),
            )

        # 1. 收集所有 Tensor
        # 既然是为了 Gloo 优化，确保这里 cat 的时候产生的是 CPU Tensor
        # 如果原始 blocks 里的 tensor 已经是 CPU pinned memory，cat 之后最好再 pin 一下

        nodes_cat = torch.cat([b.nodes for b in blocks])
        ts_cat = torch.cat([b.ts for b in blocks])

        eid_cat = torch.cat([b.eid for b in blocks])
        dts_cat = torch.cat([b.dts for b in blocks])
        col_cat = torch.cat([b.col for b in blocks])
        row_cat = torch.cat([b.row for b in blocks])

        # 2. 收集切分长度 (Sections)
        node_sections = np.array([b.nodes.shape[0] for b in blocks], dtype=np.int64)
        edge_sections = np.array([b.eid.shape[0] for b in blocks], dtype=np.int64)
        # 3. 收集标量属性
        dims = np.array([[b.dim_in, b.dim_out] for b in blocks], dtype=np.int64)
        for i, b in enumerate(blocks):
            assert b.dts.shape[0] == b.eid.shape[0], (
                f"Block {i} Data Mismatch: dts len {b.dts.shape[0]} != eid len {b.eid.shape[0]}. "
                "dts implies Edge Attribute, so lengths must match!"
            )

        # 4. 返回 Packed 对象 (关键：Gloo 发送这个对象会非常快)
        return PackedDGLBlockMeta(
            nodes_cat=nodes_cat,
            ts_cat=ts_cat,
            eid_cat=eid_cat,
            dts_cat=dts_cat,
            col_cat=col_cat,
            row_cat=row_cat,
            node_sections=node_sections,
            edge_sections=edge_sections,
            dims=dims,
        )


@dataclass(kw_only=True)
class PackedDGLBlockMeta:
    # --- 修改点 4: 允许为 None ---
    nodes_cat: torch.Tensor | None
    ts_cat: torch.Tensor | None

    eid_cat: torch.Tensor | None
    dts_cat: torch.Tensor | None
    col_cat: torch.Tensor | None
    row_cat: torch.Tensor | None

    # 元数据通常很小，可以直接走网络，不需要 None (也不需要存 SHM)
    node_sections: np.ndarray | None
    edge_sections: np.ndarray | None
    dims: np.ndarray | None

    @staticmethod
    def unpack_from_tensors(
        nodes_cat: torch.Tensor,
        ts_cat: torch.Tensor,
        eid_cat: torch.Tensor,
        dts_cat: torch.Tensor,
        col_cat: torch.Tensor,
        row_cat: torch.Tensor,
        # 新增输入
        t_node_sec: torch.Tensor,
        t_edge_sec: torch.Tensor,
        t_dims: torch.Tensor,
    ) -> list[DGLBlockMeta]:
        # 将 Tensor 转回 Numpy 以复用原有逻辑 (split 需要 list/int)
        node_sections = t_node_sec.cpu().numpy()
        edge_sections = t_edge_sec.cpu().numpy()
        dims = t_dims.view(-1, 2).cpu().numpy()

        # 复用你原有的逻辑，只是不需要 self.node_sections 了，直接用传进来的
        nodes_splits = node_sections.tolist()
        edge_splits = edge_sections.tolist()

        nodes_list = nodes_cat.split(nodes_splits)
        ts_list = ts_cat.split(nodes_splits)
        eid_list = eid_cat.split(edge_splits)
        dts_list = dts_cat.split(edge_splits)
        col_list = col_cat.split(edge_splits)
        row_list = row_cat.split(edge_splits)

        res = []
        for i in range(len(node_sections)):
            res.append(
                DGLBlockMeta(
                    nodes=nodes_list[i],
                    ts=ts_list[i],
                    eid=eid_list[i],
                    dts=dts_list[i],
                    col=col_list[i],
                    row=row_list[i],
                    dim_in=int(dims[i, 0]),
                    dim_out=int(dims[i, 1]),
                )
            )
        return res

    def pack_metadata_to_tensor(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将 numpy 元数据转换为 tensor 以便写入 SHM"""
        # 注意：使用 int64 以防溢出，且与 LongTensor 兼容
        t_node_sec = torch.from_numpy(self.node_sections).long()
        t_edge_sec = torch.from_numpy(self.edge_sections).long()
        t_dims = torch.from_numpy(self.dims).long().view(-1)
        return t_node_sec, t_edge_sec, t_dims
