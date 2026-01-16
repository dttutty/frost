from typing import List

import torch
import torch.nn.functional as F
from dgl.heterograph import DGLBlock

from gnn.components import TimeEncode
from schemas.datatypes import MemoryCFG


class BasicMemoryUpdater(torch.nn.Module):
    def __init__(
        self,
        d_mem,
        d_time,
        ndim,
        edim,
        mem_updater_class,
    ):
        super().__init__()
        self.d_mem = d_mem
        self.d_time = d_time
        self.ndim = ndim
        self.edim = edim
        assert self.d_time >= 0, "must have time encoding if memory is used"

        self.time_enc = TimeEncode(self.d_time)
        self.d_mail = 2 * self.d_mem + self.edim
        self.d_input = self.d_mail + self.d_time
        self.updater = mem_updater_class(self.d_input, self.d_mem)

    def forward(
        self,
        block: DGLBlock,
    ):
        ts: torch.Tensor = block.srcdata["ts"]
        mem: torch.Tensor = block.srcdata["mem"]
        mem_ts: torch.Tensor = block.srcdata["mem_ts"]
        mail: torch.Tensor = block.srcdata["mail"]
        nid: torch.Tensor = block.srcdata["ID"]
        time_emb = self.time_enc(ts - mem_ts)
        input = torch.cat([mail, time_emb], dim=1)
        upd_mem = self.updater(input, mem)

        return ts, upd_mem, nid


class TransformerMemoryUpdater(torch.nn.Module):
    def __init__(
        self,
        m_cfg: MemoryCFG,
        ndim: int,  # not used
        edim: int,
        dropout: float,
    ):
        super().__init__()
        d_hid = m_cfg.d_mem
        d_time = m_cfg.d_time
        boxsize = m_cfg.boxsize
        att_h = m_cfg.att_h

        d_mail = 2 * d_hid + edim
        self.d_time = d_time
        self.att_h = att_h
        if self.d_time > 0:
            self.time_enc = TimeEncode(d_time)
        self.W_q = torch.nn.Linear(d_hid, d_hid)
        self.W_k = torch.nn.Linear(d_mail + d_time, d_hid)
        self.W_v = torch.nn.Linear(d_mail + d_time, d_hid)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(d_hid)
        self.mlp = torch.nn.Linear(d_hid, d_hid)
        self.dropout = torch.nn.Dropout(dropout)
        self.boxsize = boxsize

    def forward(
        self,
        blocks: List[DGLBlock],
    ):
        assert len(blocks) == 1, (
            "Node memory updater only supports one history window and one layer"
        )
        blocks_0 = blocks[0]

        nid = blocks_0.srcdata["ID"]
        ts = blocks_0.srcdata["ts"]
        mem = blocks_0.srcdata["mem"]
        mem_ts = blocks_0.srcdata["mem_ts"]
        mail = blocks_0.srcdata["mail"]
        mail_ts = blocks_0.srcdata["mail_ts"]
        assert isinstance(nid, torch.Tensor)
        assert isinstance(ts, torch.Tensor)
        assert isinstance(mem, torch.Tensor)
        assert isinstance(mem_ts, torch.Tensor)
        assert isinstance(mail, torch.Tensor)
        assert isinstance(mail_ts, torch.Tensor)
        n_nodes = blocks_0.num_src_nodes()
        Q = self.W_q(mem).reshape((n_nodes, self.att_h, -1))
        mails = mail.reshape((n_nodes, self.boxsize, -1))
        # mem_input dim (2*mem_dim + edge_dim)
        if self.d_time > 0:
            time_feat = self.time_enc(ts - mail_ts).reshape(
                (n_nodes, self.boxsize, -1)
            )  # 这里是ts和mail_ts的差值
            mails = torch.cat([mails, time_feat], dim=2)
        K = self.W_k(mails).reshape((n_nodes, self.boxsize, self.att_h, -1))
        V = self.W_v(mails).reshape((n_nodes, self.boxsize, self.att_h, -1))
        att = self.att_act((Q[:, None, :, :] * K).sum(dim=3))
        att = torch.nn.functional.softmax(att, dim=1)
        att = self.dropout(att)

        upd_mem = (att[:, :, :, None] * V).sum(dim=1)
        upd_mem = upd_mem.reshape((upd_mem.shape[0], -1))
        upd_mem += mem
        upd_mem = self.layer_norm(upd_mem)
        upd_mem = self.mlp(upd_mem)
        upd_mem = self.dropout(upd_mem)
        upd_mem = torch.nn.functional.relu(upd_mem)

        return ts, upd_mem, nid


class SwiGLU(torch.nn.Module):
    """
    Implements the SwiGLU activation module.
    It splits the input into two halves, applies SiLU to the first half (gate),
    and multiplies it with the second half.
    """

    def forward(self, x):
        # x shape: [batch_size, ..., 2 * hidden_dim]
        gate, up_proj = x.chunk(2, dim=-1)
        return F.silu(gate) * up_proj


class GluTransformerMemoryUpdater(torch.nn.Module):
    def __init__(
        self,
        m_cfg: MemoryCFG,
        ndim: int,  # not used
        edim: int,
        dropout: float,
    ):
        super().__init__()
        d_hid = m_cfg.d_mem
        d_time = m_cfg.d_time
        boxsize = m_cfg.boxsize
        dim_in = 2 * d_hid + edim
        self.d_time = d_time
        self.att_h = m_cfg.att_h
        if d_time > 0:
            self.time_enc = TimeEncode(d_time)
        self.w_q = torch.nn.Linear(d_hid, d_hid)
        self.w_k = torch.nn.Linear(dim_in + d_time, d_hid)
        self.w_v = torch.nn.Linear(dim_in + d_time, d_hid)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(d_hid)
        # self.mlp = torch.nn.Linear(d_hid, d_hid)

        hidden_dim = d_hid * 4  # 通常 FFN 的中间层维度会扩大4倍
        self.ffn_gate_proj = torch.nn.Linear(
            d_hid, hidden_dim * 2, bias=False
        )  # SwiGLU 通常将维度扩大两倍后切分
        self.swiglu = SwiGLU()
        self.ffn_down_proj = torch.nn.Linear(hidden_dim, d_hid, bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.boxsize = boxsize

    def forward(
        self,
        mailbox: torch.Tensor,
        mailbox_ts: torch.Tensor,
        mem: torch.Tensor,
        mem_ts: torch.Tensor,
        feat: torch.Tensor,
    ):  # 全部节点都要获得新的mem
        # for block in block:
        n_src = mem.shape[0]
        Q = self.w_q(mem).reshape((n_src, self.att_h, -1))
        mails = mailbox.reshape((n_src, self.boxsize, -1))

        # mem_input dim (2*mem_dim + edge_dim)
        if self.d_time > 0:
            time_feat = self.time_enc(mem_ts[:, None] - mailbox_ts).reshape(
                (n_src, self.boxsize, -1)
            )
            mails = torch.cat([mails, time_feat], dim=2)

        K = self.w_k(mails).reshape((n_src, self.boxsize, self.att_h, -1))
        V = self.w_v(mails).reshape((n_src, self.boxsize, self.att_h, -1))
        att = self.att_act((Q[:, None, :, :] * K).sum(dim=3))
        att = torch.nn.functional.softmax(att, dim=1)
        att = self.dropout(att)
        new_mem = (att[:, :, :, None] * V).sum(dim=1)

        new_mem = new_mem.reshape((new_mem.shape[0], -1))
        new_mem += mem
        new_mem = self.layer_norm(new_mem)

        residual = new_mem

        # 使用新的 FFN
        x = self.ffn_gate_proj(new_mem)
        x = self.swiglu(x)
        x = self.ffn_down_proj(x)
        x = self.dropout(x)

        new_mem = residual + x

        # rst = self.mlp(rst)
        # rst = self.dropout(rst)
        # rst = torch.nn.functional.relu(rst)
        return new_mem, mem_ts
