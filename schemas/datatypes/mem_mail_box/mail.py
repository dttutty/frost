from __future__ import annotations

from dataclasses import dataclass

import torch
from dgl.heterograph import DGLBlock
from torch import Tensor

from .box import BoxTsNid


@dataclass(kw_only=True)
class MailTsNid:
    nid: Tensor
    data: Tensor
    ts: Tensor
    is_sorted: bool = False
    is_unique: bool = False

    def __repr__(self) -> str:
        return (
            f"MailTsNid(nid={self.nid.shape}, data={self.data.shape}, "
            f"ts={self.ts.shape}, is_sorted={self.is_sorted}, is_unique={self.is_unique})"
        )

    def clone(self) -> MailTsNid:
        return MailTsNid(
            nid=self.nid.clone(),
            data=self.data.clone(),
            ts=self.ts.clone(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def _ensure_sorted_unique_nid(self) -> MailTsNid:
        from ._data_ops import _ensure_sorted_unique_nid

        return _ensure_sorted_unique_nid(self)

    def cuda(self) -> MailTsNid:
        return MailTsNid(
            nid=self.nid.cuda(),
            data=self.data.cuda(),
            ts=self.ts.cuda(),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def to_box(self) -> BoxTsNid:
        return BoxTsNid(
            nid=self.nid,
            data=self.data.unsqueeze(1),
            ts=self.ts.unsqueeze(1),
            is_sorted=self.is_sorted,
            is_unique=self.is_unique,
        )

    def merge_with_priority(self, other: MailTsNid) -> MailTsNid:
        from ._data_ops import merge_with_priority

        return merge_with_priority(self, other)

    def keep_last(self) -> MailTsNid:
        from ._data_ops import keep_last

        ret = keep_last(self)
        return ret

    def keep_mean(self) -> MailTsNid:
        from ._data_ops import keep_mean

        ret = keep_mean(self)
        return ret

    def deliver(
        self, delivery: torch.Tensor | DGLBlock, include_self: bool = False
    ) -> MailTsNid:
        if isinstance(delivery, torch.Tensor):
            # 走 “mail_delivery_tensor: Tensor” 这条逻辑
            return self._deliver_tensor(delivery, include_self)
        elif isinstance(delivery, DGLBlock):
            # 走 “mail_delivery_block: Block” 这条逻辑
            return self._deliver_block(delivery, include_self)
        else:
            raise TypeError(f"Unsupported delivery type: {type(delivery)}")

    def _deliver_tensor(
        self, mail_delivery_tensor: Tensor, include_self: bool = True
    ) -> MailTsNid:
        """
        执行多对多 (Many-to-Many) 的消息分发 Join。

        Args:
            mail_delivery_tensor: [2, N_edges].
                                  Row 0: Src NID (发送者), Row 1: Dst NID (接收者)
            include_self:         是否保留原有的邮件（即发送给"自己"）。
        """

        # ======================
        # 特殊情况：空输入
        # ======================
        if self.nid.numel() == 0 or mail_delivery_tensor.numel() == 0:
            if include_self:
                return self

            empty_nid = torch.empty(0, dtype=self.nid.dtype, device=self.nid.device)
            empty_mail = torch.empty(
                0,
                self.data.size(1),
                dtype=self.data.dtype,
                device=self.data.device,
            )
            empty_ts = torch.empty(0, dtype=self.ts.dtype, device=self.ts.device)

            return MailTsNid(nid=empty_nid, data=empty_mail, ts=empty_ts)

        # ======================
        # 主逻辑：按 edge 循环
        # ======================
        src_nodes = mail_delivery_tensor[0]  # [N_edges]
        dst_nodes = mail_delivery_tensor[1]  # [N_edges]

        new_nids_list = []
        new_mail_list = []
        new_ts_list = []

        # 一条条 edge 处理
        for src, dst in zip(src_nodes, dst_nodes):
            # mask: 所有发给 src 的旧邮件
            mask = self.nid == src
            assert mask.any()

            # 取出这些邮件
            mails_for_src = self.data[mask]  # [K, mail_dim]
            ts_for_src = self.ts[mask]  # [K]

            k = mails_for_src.size(0)

            # 生成新的接收者 NID：全是 dst
            # dst 是 0-dim tensor，expand(k) 会返回 [k] 的视图
            new_nids_list.append(dst.expand(k))
            new_mail_list.append(mails_for_src)
            new_ts_list.append(ts_for_src)

        # ======================
        # 没有任何匹配到的邮件
        # ======================
        if len(new_nids_list) == 0:
            if include_self:
                return self

            empty_nid = torch.empty(0, dtype=self.nid.dtype, device=self.nid.device)
            empty_mail = torch.empty(
                0,
                self.data.size(1),
                dtype=self.data.dtype,
                device=self.data.device,
            )
            empty_ts = torch.empty(0, dtype=self.ts.dtype, device=self.ts.device)

            return MailTsNid(
                nid=empty_nid,
                data=empty_mail,
                ts=empty_ts,
                is_sorted=False,
                is_unique=False,
            )

        # ======================
        # 把所有 edge 收集的结果拼在一起
        # ======================
        new_nids = torch.cat(new_nids_list, dim=0)
        new_mail_data = torch.cat(new_mail_list, dim=0)
        new_ts = torch.cat(new_ts_list, dim=0)

        # ======================
        # 是否保留原有 self
        # ======================
        if not include_self:
            return MailTsNid(
                nid=new_nids,
                data=new_mail_data,
                ts=new_ts,
                is_sorted=False,
                is_unique=False,
            )

        return MailTsNid(
            nid=torch.cat([self.nid, new_nids], dim=0),
            data=torch.cat([self.data, new_mail_data], dim=0),
            ts=torch.cat([self.ts, new_ts], dim=0),
            is_sorted=False,
            is_unique=False,
        )

    def _deliver_block(
        self, mail_delivery_block: DGLBlock, include_self: bool = True
    ) -> MailTsNid:
        """
        使用 DGL Block 做多对多消息投递。

        mail_delivery_block:
            边 (global_src -> global_dst)，其中 global_* 是和 self.nid
            同一个“全局节点 id 空间”。

        include_self:
            - False: 只返回新投递出来的邮件
            - True:  返回 [原有邮件 + 新邮件]
        """
        nid = self.nid
        mail = self.data
        ts = self.ts

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
                return MailTsNid(
                    nid=empty_nid,
                    data=empty_mail,
                    ts=empty_ts,
                    is_sorted=False,
                    is_unique=False,
                )

        # ======================
        # 把所有边产生的新邮件拼在一起
        # ======================
        new_nids = torch.cat(new_nids_list, dim=0)
        new_mail_data = torch.cat(new_mail_list, dim=0)
        new_ts = torch.cat(new_ts_list, dim=0)

        # ======================
        # 是否保留原邮件
        # ======================
        if not include_self:
            return MailTsNid(
                nid=new_nids,
                data=new_mail_data,
                ts=new_ts,
                is_sorted=False,
                is_unique=False,
            )

        return MailTsNid(
            nid=torch.cat([nid, new_nids], dim=0),
            data=torch.cat([mail, new_mail_data], dim=0),
            ts=torch.cat([ts, new_ts], dim=0),
            is_sorted=False,
            is_unique=False,
        )

    def sort_by_nid(self) -> MailTsNid:
        if self.is_sorted:
            return self
        sorted_indices = torch.argsort(self.nid)
        self.nid = self.nid[sorted_indices]
        self.data = self.data[sorted_indices]
        self.ts = self.ts[sorted_indices]
        self.is_sorted = True
        return self
