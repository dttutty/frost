from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from dgl.heterograph import DGLBlock
from torch import Tensor


def is_sorted(t: torch.Tensor):
    return torch.all(torch.diff(t) >= 0)


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
