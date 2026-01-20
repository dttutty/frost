from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Set, Union

import numpy as np
import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock
from numpy.typing import NDArray
from torch import Tensor

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


IntArr = NDArray[np.int64]
FloatArr = NDArray[np.floating]


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
class GNNCFG:
    arch: Literal["identity", "attn"] = "identity"
    layer: int = 1
    d_hid: int = 100
    d_time: int = 100
    d_in: Optional[int] = None  # will later be set by train_dist.py
    d_out: Optional[int] = None  # will later be set by train_dist.py
    att_h: int = 2
    time_transform: Optional[str] = None
    combine: Optional[Literal["rnn"]] = None


@dataclass(slots=True, kw_only=True)
class TrainCFG:
    model_name: Literal["APAN", "JODIE", "TGN"] = "TGN"
    epoch: int = 10
    batch_size: int = 600
    reorder: Optional[int] = 8
    lr: float = 0.0001
    dropout: float = 0.1
    merge_src: bool = False  # add by sqp17
    ndim: int = 0  # will later be set by train_dist.py
    edim: int = 0  # will later be set by train_dist.py


@dataclass(slots=True, kw_only=True)
class SampleCFG:
    strategy: Literal["recent", "uniform"] = "recent"
    layer: int = 1
    history: int = 1
    neg_samples: int = 1
    sample_support_node: bool = True
    sample_support_for_neg_dst: bool = True
    neighbor: Optional[List[int]] = None
    prop_time: bool = False
    duration: int = 0
    num_thread: int = 32  # valid if history > 1


@dataclass(slots=True, kw_only=True)
class MemoryCFG:
    d_time: int = 0  # time dim
    d_mem: int = 128  # memory dim
    deliver_to: Literal["self", "neighbors"] = "self"
    mail_combine: Literal["mean", "last"] = "last"
    cell: Literal["gru", "rnn", "transformer"] = "gru"
    boxsize: int = 1
    combine_node_feature: bool = True
    att_h: Optional[int] = None  # only for transformer cell

    # shared_cfg = {
    #     'node_memory':      ([num_nodes, memory_param.d_hid], torch.float32),
    #     'node_memory_ts':   ([num_nodes], torch.float32),
    #     'mails':            ([num_nodes, memory_param.boxsize, 2 * memory_param.d_hid + edge_feat_dim], torch.float32),
    #     'mail_ts':          ([num_nodes, memory_param.boxsize], torch.float32),
    #     'next_mail_pos':    ([num_nodes], torch.long),
    #     'mail_pos_update_pending_flag':  ([num_nodes], torch.int8),
    # }


@dataclass(slots=True, kw_only=True)
class ScatterResult:
    m_id: int
    total_sample_time: Optional[float] = None
    ap: Optional[List[float]] = None
    auc: Optional[List[float]] = None


@dataclass(slots=True, kw_only=True)
class Config:
    g: GNNCFG
    t: TrainCFG
    s: SampleCFG  # if APAN or JODIE, do reverse sampling
    m: Optional[MemoryCFG] = None

    def dict(self):
        return asdict(self)


@dataclass(slots=True, kw_only=True)
class MemTsNid:
    id: Tensor
    data: Tensor
    ts: Tensor
    ct: Tensor | None = None

    def clone(self) -> "MemTsNid":
        return MemTsNid(id=self.id.clone(), data=self.data.clone(), ts=self.ts.clone())

    def cuda(self) -> "MemTsNid":
        return MemTsNid(id=self.id.cuda(), data=self.data.cuda(), ts=self.ts.cuda())


def is_sorted(t: torch.Tensor):
    return torch.all(torch.diff(t) >= 0)


@dataclass(slots=True, kw_only=True)
class BoxTsNid:
    id: Tensor
    data: Tensor
    ts: torch.Tensor
    next_pos: Optional[Tensor] = None
    # if next_pos is None, then this BoxTsNid is actually a MailTsNid
    ct: Optional[Tensor] = None  # 如果mean的话(每次只能累计一个mail)

    def cuda(self) -> "BoxTsNid":
        return BoxTsNid(
            id=self.id.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            next_pos=self.next_pos.cuda() if self.next_pos is not None else None,
            ct=self.ct.cuda() if self.ct is not None else None,
        )

    def merge_1_mail(self, new_mails: MailTsNid) -> "BoxTsNid":
        assert self.next_pos is not None, "next_pos must be provided for merge_1_mail"
        if __debug__:
            assert is_sorted(self.id)
            assert new_mails.id.numel() == torch.unique(new_mails.id).numel()
        idx = torch.searchsorted(self.id, new_mails.id)
        pos = self.next_pos[idx]
        self.data[idx, pos] = new_mails.data
        self.ts[idx, pos] = new_mails.ts
        self.next_pos[idx] = (self.next_pos[idx] + 1) % self.data.size(1)

        return self

    def merge_n_mail(self, new_mails: MailTsNid) -> "BoxTsNid":
        raise NotImplementedError

    def keep_latest_after_gather(self):
        return self

    def keep_mean_after_gather(self):
        return self

    def to_mail(self):
        assert self.next_pos is None, "next_pos must be None for mail"
        return MailTsNid(
            id=self.id,
            data=self.data.squeeze(1),
            ts=self.ts.squeeze(1),
            ct=self.ct,
        )


@dataclass(slots=True, kw_only=True)
class MailTsNid:
    id: Tensor
    data: Tensor

    ts: Tensor
    ct: Tensor | None = None

    def cuda(self) -> "MailTsNid":
        return MailTsNid(
            id=self.id.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            ct=self.ct.cuda() if self.ct is not None else None,
        )

    def to_box(self):
        return BoxTsNid(
            id=self.id,
            data=self.data.unsqueeze(1),
            ts=self.ts.unsqueeze(1),
            ct=self.ct,
            next_pos=None,
        )

    def keep_latest_after_gather(self):
        return self

    def deliver(
        self, delivery: Union[torch.Tensor, DGLBlock], include_self: bool = False
    ):
        if isinstance(delivery, torch.Tensor):
            # 走 “mail_delivery_tensor: Tensor” 这条逻辑
            return self._deliver_tensor(delivery, include_self)
        elif isinstance(delivery, DGLBlock):
            # 走 “mail_delivery_block: Block” 这条逻辑
            return self._deliver_block(delivery, include_self)
        else:
            raise TypeError(f"Unsupported delivery type: {type(delivery)}")

    def _deliver_tensor(
        self, mail_delivery_tensor: Tensor, include_self: bool = False
    ) -> "MailTsNid":
        """
        执行多对多 (Many-to-Many) 的消息分发 Join。

        Args:
            mail_delivery_tensor: [2, N_edges].
                                  Row 0: Src NID (发送者), Row 1: Dst NID (接收者)
            include_self:         是否保留原有的邮件（即发送给"自己"）。
        """
        # 如果你确实不支持 ct，可以保留这个 assert
        # assert self.ct is None

        # ======================
        # 特殊情况：空输入
        # ======================
        if self.id.numel() == 0 or mail_delivery_tensor.numel() == 0:
            if include_self:
                return self

            empty_nid = torch.empty(0, dtype=self.id.dtype, device=self.id.device)
            empty_mail = torch.empty(
                0,
                self.data.size(1),
                dtype=self.data.dtype,
                device=self.data.device,
            )
            empty_ts = torch.empty(0, dtype=self.ts.dtype, device=self.ts.device)
            empty_ct = None
            if self.ct is not None:
                empty_ct = torch.empty(0, dtype=self.ct.dtype, device=self.ct.device)

            return MailTsNid(id=empty_nid, data=empty_mail, ts=empty_ts, ct=empty_ct)

        # ======================
        # 主逻辑：按 edge 循环
        # ======================
        src_nodes = mail_delivery_tensor[0]  # [N_edges]
        dst_nodes = mail_delivery_tensor[1]  # [N_edges]

        new_nids_list = []
        new_mail_list = []
        new_ts_list = []
        new_ct_list = [] if self.ct is not None else None

        # 一条条 edge 处理
        for src, dst in zip(src_nodes, dst_nodes):
            # mask: 所有发给 src 的旧邮件
            mask = self.id == src
            assert mask.any()

            # 取出这些邮件
            mails_for_src = self.data[mask]  # [K, mail_dim]
            ts_for_src = self.ts[mask]  # [K]
            if self.ct is not None:
                ct_for_src = self.ct[mask]  # [K]

            k = mails_for_src.size(0)

            # 生成新的接收者 NID：全是 dst
            # dst 是 0-dim tensor，expand(k) 会返回 [k] 的视图
            new_nids_list.append(dst.expand(k))
            new_mail_list.append(mails_for_src)
            new_ts_list.append(ts_for_src)
            if new_ct_list is not None:
                new_ct_list.append(ct_for_src)

        # ======================
        # 没有任何匹配到的邮件
        # ======================
        if len(new_nids_list) == 0:
            if include_self:
                return self

            empty_nid = torch.empty(0, dtype=self.id.dtype, device=self.id.device)
            empty_mail = torch.empty(
                0,
                self.data.size(1),
                dtype=self.data.dtype,
                device=self.data.device,
            )
            empty_ts = torch.empty(0, dtype=self.ts.dtype, device=self.ts.device)
            empty_ct = None
            if self.ct is not None:
                empty_ct = torch.empty(0, dtype=self.ct.dtype, device=self.ct.device)

            return MailTsNid(id=empty_nid, data=empty_mail, ts=empty_ts, ct=empty_ct)

        # ======================
        # 把所有 edge 收集的结果拼在一起
        # ======================
        new_nids = torch.cat(new_nids_list, dim=0)
        new_mail_data = torch.cat(new_mail_list, dim=0)
        new_ts = torch.cat(new_ts_list, dim=0)
        new_ct = None
        if new_ct_list is not None:
            new_ct = torch.cat(new_ct_list, dim=0)

        # ======================
        # 是否保留原有 self
        # ======================
        if not include_self:
            return MailTsNid(new_nids, new_mail_data, new_ts, new_ct)

        return MailTsNid(
            id=torch.cat([self.id, new_nids], dim=0),
            data=torch.cat([self.data, new_mail_data], dim=0),
            ts=torch.cat([self.ts, new_ts], dim=0),
            ct=None if self.ct is None else torch.cat([self.ct, new_ct], dim=0),
        )

    def _deliver_block(
        self, mail_delivery_block, include_self: bool = False
    ) -> "MailTsNid":
        """
        使用 DGL Block 做多对多消息投递。

        mail_delivery_block:
            边 (global_src -> global_dst)，其中 global_* 是和 self.nid
            同一个“全局节点 id 空间”。

        include_self:
            - False: 只返回新投递出来的邮件
            - True:  返回 [原有邮件 + 新邮件]
        """
        nid = self.id
        mail = self.data
        ts = self.ts
        ct = self.ct

        # ======================
        # 取出 Block 中的所有边的 global src/dst
        # ======================
        src_eid, dst_eid = mail_delivery_block.edges(order="eid")
        global_src = mail_delivery_block.srcnodes()[src_eid]["ID"]
        global_dst = mail_delivery_block.dstnodes()[dst_eid]["ID"]

        device = nid.device
        dtype = nid.dtype

        new_nids_list = []
        new_mail_list = []
        new_ts_list = []
        new_ct_list = [] if ct is not None else None

        # ======================
        # 按边循环：src -> dst
        # ======================
        for s, d in zip(global_src.tolist(), global_dst.tolist()):
            # 找出所有“发给 s 的邮件”
            mask = nid == s
            if not mask.any():
                continue

            mails_for_s = mail[mask]  # [K, mail_dim]
            ts_for_s = ts[mask]  # [K]

            k = mails_for_s.size(0)

            # 这些邮件全部改为发给 d
            new_nids_list.append(torch.full((k,), d, dtype=dtype, device=device))
            new_mail_list.append(mails_for_s)
            new_ts_list.append(ts_for_s)

            if new_ct_list is not None:
                new_ct_list.append(ct[mask])

        # ======================
        # 如果没有任何匹配到 src
        # ======================
        if len(new_nids_list) == 0:
            if include_self:
                return self
            else:
                empty_nid = nid.new_empty((0,))
                empty_mail = mail.new_empty((0, mail.size(1)))
                empty_ts = ts.new_empty((0,))
                empty_ct = None
                if ct is not None:
                    empty_ct = ct.new_empty((0,))
                return MailTsNid(
                    id=empty_nid,
                    data=empty_mail,
                    ts=empty_ts,
                    ct=empty_ct,
                )

        # ======================
        # 把所有边产生的新邮件拼在一起
        # ======================
        new_nids = torch.cat(new_nids_list, dim=0)
        new_mail_data = torch.cat(new_mail_list, dim=0)
        new_ts = torch.cat(new_ts_list, dim=0)
        new_ct = None
        if new_ct_list is not None:
            new_ct = torch.cat(new_ct_list, dim=0)

        # ======================
        # 是否保留原邮件
        # ======================
        if not include_self:
            return MailTsNid(id=new_nids, data=new_mail_data, ts=new_ts, ct=new_ct)

        return MailTsNid(
            id=torch.cat([nid, new_nids], dim=0),
            data=torch.cat([mail, new_mail_data], dim=0),
            ts=torch.cat([ts, new_ts], dim=0),
            ct=None if ct is None else torch.cat([ct, new_ct], dim=0),
        )


@dataclass(slots=True, kw_only=True)
class Delta:
    box: BoxTsNid
    mem: MemTsNid


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


@dataclass(slots=True)
class MemTask:
    root_new_mem: MemTsNid
    old_mem_ts_nid: MemTsNid
    plan: CudaPlan
    is_last: bool


@dataclass(slots=True)
class BoxTask:
    root_new_mem_orig: MemTsNid
    pos_efeat: Tensor
    macro_bid: int
    old_box: BoxTsNid
    plan: CudaPlan
    is_last: bool
    mail_delivery_tensor: Tensor


@dataclass(slots=True)
class P2PTask:
    nid: GatherElement
    data: GatherElement
    ts: GatherElement
    type: Literal["mem", "box", "mail"]


@dataclass(slots=True)
class GatherElement:
    recv: Tensor
    send: Tensor
    out_splits: NDArray
    in_splits: NDArray


@dataclass(slots=True, kw_only=True)
class MetaList:
    queue: List[Meta]

    def __len__(self) -> int:
        return len(self.queue)


@dataclass(slots=True, kw_only=True)
class Meta:
    gnn_mfgs_raw: List[DGLBlockMeta] | None
    all_nodes: Set[int] | None | torch.Tensor
    root_nodes: Set[int] | None | torch.Tensor
    pos_edge: Tensor  # 没有去重的2root nodes
    pos_edge_ts: Tensor  # 没有3倍处理
    pos_edge_id: Tensor  # 没有3倍处理,这个信息不需要传输到gpu
    epoch_split_iter_info: Optional[EpochSplitIterInfo] = None
    feat: Optional[X] = None
    mdt: Optional[Tensor] = None
    mdb_raw_data: DGLBlockMeta | None = None
    batch_node_state_io: Optional[BatchNodeStateIO] = None


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
class MetaReady2Send:
    gnn_mfgs_raw: List[DGLBlockMeta] | None
    all_nodes: torch.Tensor
    root_nodes: torch.Tensor
    pos_edge: Tensor  # 没有去重的2root nodes
    pos_edge_ts: Tensor  # 没有3倍处理
    pos_edge_id: Tensor  # 没有3倍处理,这个信息不需要传输到gpu
    # pred_cts: List[int]
    # total_mbatchs_for_this_rank: Optional[int] = None
    feat: Optional[X] = None
    mdt: Optional[Tensor] = None
    mdb_raw_data: DGLBlockMeta | None = None
    plan_set: Optional[PlanSet | TensorPlanSet] = None
    epoch_split_iter_info: EpochSplitIterInfo


@dataclass(slots=True, kw_only=True)
class SyncTask:
    macro_bid: int
    plan_set: TensorPlanSet
    root_new_mem: MemTsNid
    x: X
    mail_delivery_tensor: torch.Tensor | None
    is_last: bool
    ready_event: torch.cuda.Event


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


Role = Literal["worker", "host"]


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


@dataclass(frozen=True, kw_only=True)
class Env:
    local_rank: int
    world_size: int
    host_rank: int  # 约定为 world_size - 1
    role: Role  # "worker" / "host"
    n_gpu: int  # = world_size - 1（每 worker 1 张）

    @property
    def is_host(self) -> bool:
        return self.role == "host"

    @classmethod
    def initial(cls) -> "Env":
        assert dist.is_initialized()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if rank != world_size - 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            torch.cuda.set_device(0)
        host_rank = world_size - 1  # the last rank is host
        role: Role = "host" if rank == host_rank else "worker"

        n_gpu = world_size - 1

        return cls(
            local_rank=rank,
            world_size=world_size,
            host_rank=host_rank,
            role=role,
            n_gpu=n_gpu,
        )
