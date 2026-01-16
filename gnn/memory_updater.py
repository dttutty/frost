# worker/trainer.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock

from gnn.components import TimeEncode


class MemoryUpdater(torch.nn.Module):
    def __init__(self, time_encoder: nn.Module, rnn: nn.Module):
        super(MemoryUpdater, self).__init__()
        self.time_encoder = time_encoder
        self.rnn = rnn

    def forward(
        self,
        block: DGLBlock,
    ):
        mem: torch.Tensor = block.srcdata["mem"]
        ts: torch.Tensor = block.srcdata["ts"]
        mem_ts: torch.Tensor = block.srcdata["mem_ts"]
        box: torch.Tensor = block.srcdata["box"]

        mail = torch.squeeze(box, dim=1)  # Flatten because it has extra dimension

        time_feat = self.time_encoder(ts - mem_ts)
        rnn_input = torch.cat([mail, time_feat], dim=1)
        up_mem = self.rnn(rnn_input, mem)
        block.srcdata["updated_mem"] = up_mem

        return up_mem


class TransformerMemoryUpdater(nn.Module):
    """
    The core of APAN: Updates node memory by attending to the Mailbox.
    Corresponds to 'TransformerMemoryUpdater' in original memorys.py
    """

    def __init__(
        self,
        d_mail,
        d_mem,
        d_time,
        boxsize,
        heads=2,
        dropout=0.1,
        att_dropout=0.1,
    ):
        super(TransformerMemoryUpdater, self).__init__()
        self.dim_mem = d_mem
        self.dim_time = d_time
        self.boxsize = boxsize
        self.att_h = heads

        # Dimensions check
        assert d_mem % heads == 0, "dim_out must be divisible by heads"
        self.head_dim = d_mem // heads

        self.time_enc = TimeEncode(d_time)

        # Input dim for K, V is dim_out (message content) + dim_time (timestamp enc)
        # Note: In APAN, raw mailbox message is usually concat(mem_src, mem_dst, edge_feat)
        # We assume the input projection happens here or input is already dim_out compatible.
        # Based on original code:
        # W_q takes 'mem' (dim_out)
        # W_k, W_v take 'mails' (dim_in + dim_time).
        # In memorys.py, dim_in = 2*dim_out + dim_edge. Let's make this explicit in __init__.

        self.w_q = nn.Linear(d_mem, d_mem)

        self.w_k = nn.Linear(d_mail + d_time, self.dim_mem)
        self.w_v = nn.Linear(d_mail + d_time, self.dim_mem)

        # We define these dynamically in forward or assume input is projected?
        # Looking at memorys.py:
        # self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        # To make this class robust, we will use a lazy approach or standard dims.
        # Let's assume input mail dim is handled.

        self.layer_norm = nn.LayerNorm(d_mem)
        self.mlp = nn.Linear(d_mem, d_mem)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)

    def forward(self, block: DGLBlock):
        """
        block: DGL MFG (Message Flow Graph)
        Requires block.srcdata to have:
            - 'mem': Current node memory [N, dim_mem]
            - 'box': Flattened mailbox content [N, mailbox_size, mail_feature_dim]
            - 'ts': Current time [N]
            - 'box_ts': Mailbox timestamps [N, mailbox_size]
        """
        mem: torch.Tensor = block.srcdata["mem"]
        ts: torch.Tensor = block.srcdata["ts"]
        box: torch.Tensor = block.srcdata["box"]
        box_ts: torch.Tensor = block.srcdata["box_ts"]

        batch_size = block.num_src_nodes()

        # 1. Compute Query (from current Memory)
        # [N, H, D_head] D_head = dim_mem / heads
        Q = self.w_q(mem).reshape((batch_size, self.att_h, -1))

        # 2. Time Encoding for Mails
        # Time diff: t_curr - t_mail
        # ts[:, None] is [N, 1], box_ts is [N, L] -> [N, L]
        if self.dim_time > 0:
            time_feat = self.time_enc(ts[:, None] - box_ts)  # [N*L, dim_time]
            time_feat = time_feat.reshape((batch_size, self.boxsize, -1))
            # Concat time to mails: [N, L, D_mail + D_time]
            box_with_temb = torch.cat([box, time_feat], dim=2)

        # 3. Compute Key and Value (from Mails)
        # [N, L, H, D_head] D_head = dim_input / heads
        K = self.w_k(box_with_temb).reshape((batch_size, self.boxsize, self.att_h, -1))
        V = self.w_v(box_with_temb).reshape((batch_size, self.boxsize, self.att_h, -1))

        # 4. Attention
        # (Q * K).sum -> [N, L, H] (Attention scores for each mail item per head)
        # Note: Q is [N, H, D], K is [N, L, H, D]. Broadcast Q to [N, 1, H, D]
        att_score = (Q[:, None, :, :] * K).sum(dim=3)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        att_weights = F.softmax(att_score, dim=1)  # Softmax over L (mailbox items)
        att_weights = self.att_dropout(att_weights)

        # 5. Aggregate
        # Weighted sum of V: [N, L, H, 1] * [N, L, H, D] -> sum over L -> [N, H, D]
        rst = (att_weights[:, :, :, None] * V).sum(dim=1)
        rst = rst.reshape((batch_size, -1))  # [N, dim_out]

        # 6. Residual + Norm + MLP
        rst = self.layer_norm(rst + mem)
        rst = self.mlp(rst)
        rst = self.dropout(rst)
        rst = F.relu(rst)

        # Update block for downstream
        block.srcdata["updated_mem"] = rst

        return rst
