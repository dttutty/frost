from typing import List, Union

import torch
import torch.nn.functional as F

from layers import TimeEncode
from schemas.datatypes import MemoryCFG, SharedMailboxArrays


class MemoryMailbox:
    node_memory: torch.Tensor
    node_memory_ts: torch.Tensor
    mails: torch.Tensor
    mail_ts: torch.Tensor
    next_mail_pos: torch.Tensor
    mail_pos_update_pending_flag: torch.Tensor | None  # 允许为 None

    shared_cfg: dict  # 仍然保留它，用于非共享模式下的形状定义
    memory_param: MemoryCFG
    dim_edge_feat: int
    device: torch.device
    # 1. 【核心修复】在类级别声明所有属性
    #    我们明确告诉类型检查器，这些属性的类型是“不一致的”
    pinned_node_memory_buffs: Union[List[torch.Tensor], torch.Tensor]
    pinned_mailbox_buffs: Union[List[torch.Tensor], torch.Tensor]

    #    TS buffer (时间戳) 始终是 list，这很棒
    pinned_node_memory_ts_buffs: List[torch.Tensor]
    pinned_mailbox_ts_buffs: List[torch.Tensor]

    # ..._size 属性
    pinned_node_memory_buffs_size: int | None
    pinned_mailbox_buffs_size: int | None

    def __init__(
        self,
        mcfg: MemoryCFG,
        shared_cfg,
        dim_edge_feat,
        shared_arrays: SharedMailboxArrays,
    ):
        self.shared_cfg = shared_cfg
        self.memory_param = mcfg
        self.dim_edge_feat = dim_edge_feat

        if mcfg.type != "node":
            raise NotImplementedError

        # 2. 【核心修复】用直接赋值替换 setattr 循环
        if shared_arrays is not None:
            # --- 路径A: 使用共享内存---
            # 这既快又类型安全，自动补全也能完美工作
            self.node_memory = shared_arrays.node_memory
            self.node_memory_ts = shared_arrays.node_memory_ts
            self.mails = shared_arrays.mails
            self.mail_ts = shared_arrays.mail_ts
            self.next_mail_pos = shared_arrays.next_mail_pos
            self.mail_pos_update_pending_flag = (
                shared_arrays.mail_pos_update_pending_flag
            )
        else:
            # --- 路径B: 非共享内存 (fallback) ---
            # 我们仍然使用 shared_cfg 来获取形状，但仍然是直接赋值

            # (注意: 你原来的代码中 'shared_arrays' 为 None 时，
            # 'mail_pos_update_pending_flag' 会是 None,
            # 其它的是 torch.zeros。我们保留这个逻辑。)

            shape, dtype = shared_cfg["node_memory"]
            self.node_memory = torch.zeros(shape, dtype=dtype)

            shape, dtype = shared_cfg["node_memory_ts"]
            self.node_memory_ts = torch.zeros(shape, dtype=dtype)

            shape, dtype = shared_cfg["mails"]
            self.mails = torch.zeros(shape, dtype=dtype)

            shape, dtype = shared_cfg["mail_ts"]
            self.mail_ts = torch.zeros(shape, dtype=dtype)

            shape, dtype = shared_cfg["next_mail_pos"]
            self.next_mail_pos = torch.zeros(shape, dtype=dtype)

            self.mail_pos_update_pending_flag = None
        self.device = torch.device("cpu")

    def reset(self):
        self.node_memory.zero_()
        self.node_memory_ts.zero_()
        self.mails.zero_()
        self.mail_ts.zero_()
        self.next_mail_pos.zero_()

    def move_to_gpu(self):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mails = self.mails.cuda()
        self.mail_ts = self.mail_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        # self.mail_pos_update_pending_flag 保持在 CPU 或为 None

        self.device = torch.device("cuda:0")

    def alc_pin_mem_mailbox_buf(self, dedup, sample_param, batch_size):
        limit_of_n_nodes = int(batch_size * 3.3)
        if sample_param.neighbor is not None:
            for i in sample_param.neighbor:
                limit_of_n_nodes *= i + 1

        mem_dim = self.node_memory.shape[1]
        mailbox_size = self.mails.shape[1]
        mail_dim = self.mails.shape[2]

        # ts is only 1 dim tensor, no need to dedup
        self.pinned_node_memory_ts_buffs = []
        self.pinned_mailbox_ts_buffs = []

        if not dedup:
            # --- 分支 A: 为 history 里的每一层创建单独的 buffer ---
            self.pinned_node_memory_buffs = []
            self.pinned_mailbox_buffs = []

            # 3. 【修复 AttributeError】在这里也初始化 _size 属性
            self.pinned_node_memory_buffs_size = None
            self.pinned_mailbox_buffs_size = None

            for _ in range(sample_param.history):
                # ts buffer
                self.pinned_node_memory_ts_buffs.append(
                    torch.zeros([limit_of_n_nodes], pin_memory=True)
                )
                self.pinned_mailbox_ts_buffs.append(
                    torch.zeros([limit_of_n_nodes, mailbox_size], pin_memory=True)
                )
                # data buffer
                self.pinned_node_memory_buffs.append(
                    torch.zeros([limit_of_n_nodes, mem_dim], pin_memory=True)
                )
                self.pinned_mailbox_buffs.append(
                    torch.zeros(
                        [limit_of_n_nodes, mailbox_size, mail_dim], pin_memory=True
                    )
                )
        else:
            # --- 分支 B: (已修复) 为所有层创建 *一个* 连续的大 buffer ---

            # 4. 【修复 内部不一致】为 *所有* buffer 计算总大小
            self.pinned_node_memory_buffs_size = 0
            self.pinned_mailbox_buffs_size = 0

            for _ in range(sample_param.history):
                # --- TS buffers (List) [与分支 A 逻辑相同] ---
                self.pinned_node_memory_ts_buffs.append(
                    torch.zeros([limit_of_n_nodes], pin_memory=True)
                )
                self.pinned_mailbox_ts_buffs.append(
                    torch.zeros([limit_of_n_nodes, mailbox_size], pin_memory=True)
                )

                # --- Data _size (用于创建单个 Tensor) ---
                self.pinned_node_memory_buffs_size += limit_of_n_nodes
                self.pinned_mailbox_buffs_size += limit_of_n_nodes

            # 5. 在循环 *外* 创建那 2 个大的 *数据* Tensor
            self.pinned_node_memory_buffs = torch.zeros(
                [self.pinned_node_memory_buffs_size, mem_dim], pin_memory=True
            )
            self.pinned_mailbox_buffs = torch.zeros(
                [self.pinned_mailbox_buffs_size, mailbox_size, mail_dim],
                pin_memory=True,
            )


class BasicMemoryUpdater(torch.nn.Module):
    def __init__(self, memory_param: MemoryCFG, edge_feat_dim, node_feat_dim):
        super().__init__()

        self.dim_hid = memory_param.dim_out
        self.dim_time = memory_param.dim_time
        self.rnn_input_size = 2 * self.dim_hid + edge_feat_dim + self.dim_time
        self.combine_node_feature = memory_param.combine_node_feature

        self.dim_node_feat = node_feat_dim

        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if self.dim_time > 0:
            self.time_enc = TimeEncode(self.dim_time)
        if self.combine_node_feature:
            if self.dim_node_feat > 0 and self.dim_node_feat != self.dim_hid:
                self.node_feat_map = torch.nn.Linear(self.dim_node_feat, self.dim_hid)

    def forward(self, mfg):
        for block in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(block.srcdata["ts"] - block.srcdata["mem_ts"])
                block.srcdata["mem_input"] = torch.cat(
                    [block.srcdata["mem_input"], time_feat], dim=1
                )

            updated_memory = self.updater(
                block.srcdata["mem_input"], block.srcdata["mem"]
            )

            self.last_updated_ts = block.srcdata["ts"].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = (
                block.srcdata["ID"].detach().clone()
            )  # too much nodes for memory updater and mailbox updater

            if self.combine_node_feature:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        block.srcdata["h"] += updated_memory
                    else:
                        block.srcdata["h"] = updated_memory + self.node_feat_map(
                            block.srcdata["h"]
                        )
                else:
                    block.srcdata["h"] = updated_memory


class GRUMemoryUpdater(BasicMemoryUpdater):
    def __init__(self, memory_param: MemoryCFG, edge_feat_dim, node_feat_dim):
        super().__init__(memory_param, edge_feat_dim, node_feat_dim)
        self.updater = torch.nn.GRUCell(self.rnn_input_size, self.dim_hid)


class RNNMemoryUpdater(BasicMemoryUpdater):
    def __init__(self, memory_param: MemoryCFG, edge_feat_dim, node_feat_dim):
        super().__init__(memory_param, edge_feat_dim, node_feat_dim)
        self.updater = torch.nn.RNNCell(self.rnn_input_size, self.dim_hid)


class TransformerMemoryUpdater(torch.nn.Module):
    def __init__(self, memory_param, edge_feat_dim, train_param):
        super(TransformerMemoryUpdater, self).__init__()
        dim_in = 2 * memory_param.dim_out + edge_feat_dim
        dim_out = memory_param.dim_out
        self.memory_param = memory_param
        self.dim_time = memory_param.dim_time
        self.att_h = memory_param.attention_head
        if self.dim_time > 0:
            self.time_enc = TimeEncode(self.dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + self.dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + self.dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(train_param.dropout)
        self.att_dropout = torch.nn.Dropout(train_param.att_dropout)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for block in mfg:
            n_src = block.num_src_nodes()
            mailbox_size = self.memory_param.mailbox_size
            Q = self.w_q(block.srcdata["mem"]).reshape((n_src, self.att_h, -1))
            mails = block.srcdata["mem_input"].reshape((n_src, mailbox_size, -1))
            # mem_input dim (2*mem_dim + edge_dim)
            if self.dim_time > 0:
                time_feat = self.time_enc(
                    block.srcdata["ts"][:, None] - block.srcdata["mail_ts"]
                ).reshape((n_src, mailbox_size, -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape((n_src, mailbox_size, self.att_h, -1))
            V = self.w_v(mails).reshape((n_src, mailbox_size, self.att_h, -1))
            att = self.att_act((Q[:, None, :, :] * K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:, :, :, None] * V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += block.srcdata["mem"]
            rst = self.layer_norm(rst)
            rst = self.mlp(rst)
            rst = self.dropout(rst)
            rst = torch.nn.functional.relu(rst)
            block.srcdata["h"] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = block.srcdata["ID"].detach().clone()
            self.last_updated_ts = block.srcdata["ts"].detach().clone()


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
    def __init__(self, memory_param, edge_feat_dim, train_param):
        super(GluTransformerMemoryUpdater, self).__init__()
        dim_in = 2 * memory_param.dim_out + edge_feat_dim
        dim_out = memory_param.dim_out
        self.memory_param = memory_param
        self.dim_time = memory_param.dim_time
        self.att_h = memory_param.attention_head
        if self.dim_time > 0:
            self.time_enc = TimeEncode(self.dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + self.dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + self.dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        # self.mlp = torch.nn.Linear(dim_out, dim_out)

        hidden_dim = dim_out * 4  # 通常 FFN 的中间层维度会扩大4倍
        self.ffn_gate_proj = torch.nn.Linear(
            dim_out, hidden_dim * 2, bias=False
        )  # SwiGLU 通常将维度扩大两倍后切分
        self.swiglu = SwiGLU()
        self.ffn_down_proj = torch.nn.Linear(hidden_dim, dim_out, bias=False)

        self.dropout = torch.nn.Dropout(train_param.dropout)
        self.att_dropout = torch.nn.Dropout(train_param.att_dropout)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for block in mfg:
            n_src = block.num_src_nodes()
            mailbox_size = self.memory_param.mailbox_size
            Q = self.w_q(block.srcdata["mem"]).reshape((n_src, self.att_h, -1))
            mails = block.srcdata["mem_input"].reshape((n_src, mailbox_size, -1))
            # mem_input dim (2*mem_dim + edge_dim)
            if self.dim_time > 0:
                time_feat = self.time_enc(
                    block.srcdata["ts"][:, None] - block.srcdata["mail_ts"]
                ).reshape((n_src, mailbox_size, -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape((n_src, mailbox_size, self.att_h, -1))
            V = self.w_v(mails).reshape((n_src, mailbox_size, self.att_h, -1))
            att = self.att_act((Q[:, None, :, :] * K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:, :, :, None] * V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += block.srcdata["mem"]
            rst = self.layer_norm(rst)

            residual = rst

            # 使用新的 FFN
            x = self.ffn_gate_proj(rst)
            x = self.swiglu(x)
            x = self.ffn_down_proj(x)
            x = self.dropout(x)

            rst = residual + x

            # rst = self.mlp(rst)
            # rst = self.dropout(rst)
            # rst = torch.nn.functional.relu(rst)
            block.srcdata["h"] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = block.srcdata["ID"].detach().clone()
            self.last_updated_ts = block.srcdata["ts"].detach().clone()
