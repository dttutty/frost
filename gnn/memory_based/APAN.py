from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock

from gnn.components import EdgePredictor, TimeEncode
from schemas.datatypes.data_to_sync import MemTsNid
from schemas.datatypes.misc import DyX
from schemas.datatypes.plan import TensorPlanSet
from schemas.datatypes.task import SyncTask


class APANMemoryTransformer(nn.Module):
    """
    The core of APAN: Updates node memory by attending to the Mailbox.
    Corresponds to 'TransformerMemoryUpdater' in original memorys.py
    """

    def __init__(
        self,
        dim_edge,
        dim_mem,
        dim_time,
        mailbox_size,
        heads=2,
        dropout=0.1,
        att_dropout=0.1,
    ):
        super(APANMemoryTransformer, self).__init__()
        self.dim_mem = dim_mem
        self.dim_time = dim_time
        self.boxsize = mailbox_size
        self.att_h = heads

        # Dimensions check
        assert dim_mem % heads == 0, "dim_out must be divisible by heads"
        self.head_dim = dim_mem // heads

        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)

        # Input dim for K, V is dim_out (message content) + dim_time (timestamp enc)
        # Note: In APAN, raw mailbox message is usually concat(mem_src, mem_dst, edge_feat)
        # We assume the input projection happens here or input is already dim_out compatible.
        # Based on original code:
        # W_q takes 'mem' (dim_out)
        # W_k, W_v take 'mails' (dim_in + dim_time).
        # In memorys.py, dim_in = 2*dim_out + dim_edge. Let's make this explicit in __init__.

        self.w_q = nn.Linear(dim_mem, dim_mem)

        mail_dim = 2 * dim_mem + dim_edge
        self.w_k = nn.Linear(mail_dim + self.dim_time, self.dim_mem).to(
            self.w_q.weight.device
        )
        self.w_v = nn.Linear(mail_dim + self.dim_time, self.dim_mem).to(
            self.w_q.weight.device
        )

        # We define these dynamically in forward or assume input is projected?
        # Looking at memorys.py:
        # self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        # To make this class robust, we will use a lazy approach or standard dims.
        # Let's assume input mail dim is handled.

        self.att_act = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(dim_mem)
        self.mlp = nn.Linear(dim_mem, dim_mem)
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
        mem_ts: torch.Tensor = block.srcdata["mem_ts"]
        ts: torch.Tensor = block.srcdata["ts"]
        box: torch.Tensor = block.srcdata["box"]
        box_ts: torch.Tensor = block.srcdata["box_ts"]

        batch_size = block.num_src_nodes()

        # 1. Compute Query (from current Memory)
        # [N, H, D_head]
        Q = self.w_q(mem).reshape((batch_size, self.att_h, -1))

        # 2. Time Encoding for Mails
        # Time diff: t_curr - t_mail
        # ts[:, None] is [N, 1], box_ts is [N, L] -> [N, L]
        if self.dim_time > 0:
            time_feat = self.time_enc(ts - box_ts)  # [N*L, dim_time]
            time_feat = time_feat.reshape((batch_size, self.boxsize, -1))
            # Concat time to mails: [N, L, D_mail + D_time]
            box_with_temb = torch.cat([box, time_feat], dim=2)

        # 3. Compute Key and Value (from Mails)
        # [N, L, H, D_head]
        K = self.w_k(box_with_temb).reshape((batch_size, self.boxsize, self.att_h, -1))
        V = self.w_v(box_with_temb).reshape((batch_size, self.boxsize, self.att_h, -1))

        # 4. Attention
        # (Q * K).sum -> [N, L, H] (Attention scores for each mail item per head)
        # Note: Q is [N, H, D], K is [N, L, H, D]. Broadcast Q to [N, 1, H, D]
        att_score = (Q[:, None, :, :] * K).sum(dim=3)
        att_score = self.att_act(att_score)
        att_weights = F.softmax(att_score, dim=1)  # Softmax over L (mailbox items)
        att_weights = self.att_dropout(att_weights)

        # 5. Aggregate
        # Weighted sum of V: [N, L, H, 1] * [N, L, H, D] -> sum over L -> [N, H, D]
        rst = (att_weights[:, :, :, None] * V).sum(dim=1)
        rst = rst.reshape((batch_size, -1))  # [N, dim_out]

        # 6. Residual + Norm + MLP
        rst = rst + mem
        rst = self.layer_norm(rst)
        rst = self.mlp(rst)
        rst = self.dropout(rst)
        rst = F.relu(rst)

        # Update block for downstream
        block.srcdata["h"] = rst

        return rst


class APAN(nn.Module):
    """
    Simplified APAN Model.
    Architecture:
    1. Memory Update (Transformer over Mailbox)
    2. GNN (Identity + LayerNorm)
    3. Edge Prediction
    """

    def __init__(
        self,
        dim_node,  # Unused in standard APAN if pure identity, but kept for interface compatibility
        dim_edge,  # Used to calculate mailbox dimensions implicitely
        dim_memory=100,
        dim_time=100,
        mailbox_size=10,
        heads=2,
        dropout=0.1,
        att_dropout=0.1,
    ):
        super(APAN, self).__init__()
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_memory = dim_memory

        # --- 1. Memory Updater ---
        # Corresponds to memory_param['memory_update'] == 'transformer'
        self.memory_updater = APANMemoryTransformer(
            dim_edge=dim_edge,
            dim_mem=dim_memory,
            dim_time=dim_time,
            mailbox_size=mailbox_size,
            heads=heads,
            dropout=dropout,
            att_dropout=att_dropout,
        )

        # --- 2. GNN Layer (Identity) ---
        # APAN typically uses Identity GNN, meaning the embedding IS the memory.
        # We just add a LayerNorm as per original code 'IdentityNormLayer'.
        self.gnn_norm = nn.LayerNorm(dim_memory)

        # --- 3. Edge Predictor ---
        self.edge_predictor = EdgePredictor(dim_memory)

    def forward(
        self,
        macro_bid: int,
        blockss: List[List[DGLBlock]],
        n_root_src_n_pos_dst: int,
        mdt: torch.Tensor,
        x: DyX,
        plan_set: TensorPlanSet,
        neg_samples=1,
        is_last: bool = False,
    ):
        """
        mfgs: List of Message Flow Graphs (DGL Blocks).
              mfgs[0] is the Target Nodes block.
              It must contain 'mem', 'mem_input' (mailbox), 'ts', 'mail_ts'.
        """
        block = blockss[0][0]

        # 1. Update Memory
        # This writes 'h' into target_block.srcdata['h']
        updated_mem = self.memory_updater(block)

        ts = block.srcdata["ts"]
        nid = block.srcdata["ID"]

        root_new_mem = MemTsNid(
            ts=ts[:n_root_src_n_pos_dst],  # 新交互的ts
            data=updated_mem[:n_root_src_n_pos_dst],
            nid=nid[:n_root_src_n_pos_dst],  # 新交互的nid
        )
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())
        self.sync_q.put(
            SyncTask(
                macro_bid=macro_bid,
                plan_set=plan_set,
                root_new_mem=root_new_mem,
                x=x,
                mail_delivery_tensor=mdt,
                is_last=is_last,
                ready_event=ready_event,  # type: ignore
            )
        )

        # 2. GNN (Identity + Norm)
        # Original code: l0h0 is IdentityNormLayer
        # Since it's identity, we just normalize the output of the memory updater
        # Note: APAN assumes the memory updater output IS the node embedding.
        embedding = self.gnn_norm(updated_mem)

        # 3. Edge Prediction
        # The embedding tensor contains [Src, Pos_Dst, Neg_Dst] concatenated rows (usually)
        # The EdgePredictor handles splitting this internally.
        res_pos, res_neg = self.edge_predictor(embedding, neg_samples=neg_samples)

        return res_pos, res_neg
