"""Memory updater modules for temporal GNN models.

Provides two implementations:

* ``MemoryUpdater`` — RNN/GRU-based updater used by TGN and JODIE.
* ``TransformerMemoryUpdater`` — Multi-head attention updater used by APAN.

Both operate on a ``FlatGraph`` in-place and cache detached states for the
target root nodes so that the assembled model can push them to shared memory
without holding gradient references.
"""
# worker/trainer.py

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from frost.datatypes.graph import FlatGraph
from frost.models.modules.time_encoder import TimeEncoder

logger = logging.getLogger(__name__)


class MemoryUpdater(nn.Module):
    """GRU- or RNN-based node memory updater.

    For each node in the flat graph, reads its current mailbox message and
    the time elapsed since its last memory update, encodes the time delta,
    and passes the concatenated vector through a recurrent cell to produce a
    new memory state.

    Used by TGN and JODIE.

    Args:
        d_mail: Dimensionality of a single mailbox message.
        d_time: Dimensionality of the time encoding.
        d_mem: Dimensionality of the memory state (output dim of the RNN cell).
        cell: Recurrent cell type — ``"gru"`` (default) or ``"rnn"``.
    """

    # Map the exact parameters you need from YAML to the __init__ arguments
    def __init__(
        self,
        d_mail: int,
        d_time: int,  # from hydra config
        d_mem: int,  # from hydra config
        cell: Literal["rnn", "gru"] = "gru",  # from hydra config
    ) -> None:
        super().__init__()
        self.cell = cell
        # Use self. to save it as an instance attribute
        rnn: nn.GRUCell | nn.RNNCell
        if self.cell == "gru":
            rnn = torch.nn.GRUCell(d_mail + d_time, d_mem)
        elif self.cell == "rnn":
            rnn = torch.nn.RNNCell(d_mail + d_time, d_mem)
        else:
            raise ValueError(f"Unknown cell type: {self.cell}")

        self.time_encoder = TimeEncoder(d_time)
        self.rnn = rnn

    def forward(
        self,
        flat_graph: "FlatGraph",
        n_target_nodes: int,
        n_root_nodes: int | None = None,
    ) -> "FlatGraph":
        """Update node memories for all nodes in the flat graph.

        Encodes the time delta between the current event timestamp and the
        last memory update timestamp, concatenates it with the mailbox
        message, and feeds the result through the recurrent cell. Detached
        copies of the new memory, timestamp, and node IDs for the first
        ``n_target_nodes`` nodes (i.e., src + pos_dst) are cached on
        ``flat_graph`` for async global synchronization.

        Args:
            flat_graph: The flat graph carrying node IDs, timestamps,
                memories, and mailbox data for all nodes in the batch.
            n_target_nodes: Number of root nodes whose updated states should
                be cached for global memory sync (typically ``n_src * 2``).

        Returns:
            The same ``flat_graph`` with ``flat_graph.mem`` updated in-place
            and ``detached_mem`` / ``detached_ts`` / ``detached_nid`` set.
        """
        # Skip if no memory or box data exists
        assert flat_graph.mem is not None
        assert flat_graph.box is not None
        assert flat_graph.mem_ts is not None
        assert flat_graph.box_ts is not None

        # Flatten mailbox if it has an extra dimension
        if flat_graph.box.dim() > 1 and flat_graph.box.shape[1] == 1:
            mail = torch.squeeze(flat_graph.box, dim=1)
        else:
            mail = flat_graph.box

        # Single batched forward pass for all roots and neighbors
        time_feat = self.time_encoder(flat_graph.ts - flat_graph.mem_ts)
        rnn_input = torch.cat([mail, time_feat], dim=1)

        # In-place update of the flattened memory tensor
        flat_graph.mem = self.rnn(rnn_input, flat_graph.mem)
        assert flat_graph.mem is not None

        # Cache detached states for Positive Root nodes (assumes roots are at the left)
        flat_graph.detached_mem = flat_graph.mem[:n_target_nodes].detach().clone()
        flat_graph.detached_ts = flat_graph.ts[:n_target_nodes].detach().clone()
        flat_graph.detached_nid = flat_graph.nid[:n_target_nodes].detach().clone()

        return flat_graph


class TransformerMemoryUpdater(nn.Module):
    """
    Batched Transformer Memory Updater for APAN using FlatGraph inputs.
    Updates node memory by attending to the Mailbox messages.
    """

    def __init__(
        self,
        d_mail: int,
        d_mem: int,  # from hydra config
        d_time: int,  # from hydra config
        boxsize: int,  # from hydra config
        heads: int,  # from hydra config
        dropout: float,  # from hydra config\
    ) -> None:
        super().__init__()
        self.dim_mem = d_mem
        self.dim_time = d_time
        self.boxsize = boxsize
        self.att_h = heads

        assert d_mem % heads == 0, "dim_mem must be divisible by heads"
        self.head_dim = d_mem // heads

        self.time_enc = TimeEncoder(d_time) if d_time > 0 else None

        self.w_q = nn.Linear(d_mem, d_mem)
        self.w_k = nn.Linear(d_mail + d_time, d_mem)
        self.w_v = nn.Linear(d_mail + d_time, d_mem)

        self.layer_norm = nn.LayerNorm(d_mem)
        self.mlp = nn.Linear(d_mem, d_mem)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        flat_graph: "FlatGraph",
        n_target_nodes: int,
        n_root_nodes: int | None = None,
    ) -> "FlatGraph":
        """Update node memories using multi-head attention over mailbox messages.

        Computes queries from current node memory, keys and values from
        time-encoded mailbox messages, applies multi-head dot-product attention
        with LeakyReLU activation, and applies a residual + LayerNorm + MLP
        post-processing step.  Detached copies of the updated states for the
        first ``n_target_nodes`` nodes are cached on ``flat_graph`` for async
        global synchronization.

        Args:
            flat_graph: The flat graph carrying node IDs, timestamps,
                memories, and mailbox data for all nodes in the batch.
            n_target_nodes: Number of root nodes whose updated states should
                be cached (typically ``n_src * 2``).
            n_root_nodes: Optional prefix length to update. When provided,
                only ``flat_graph[:n_root_nodes]`` participates in the
                attention update and later rows are left untouched.

        Returns:
            The same ``flat_graph`` with ``flat_graph.mem`` updated in-place
            and ``detached_mem`` / ``detached_ts`` / ``detached_nid`` set.
        """
        # Skip if no memory or box data exists
        if flat_graph.mem is None or flat_graph.box is None:
            return flat_graph

        mem = flat_graph.mem  # [N_total, dim_mem]
        ts = flat_graph.ts  # [N_total]
        box = flat_graph.box  # [N_total, boxsize, dim_mail]
        box_ts = flat_graph.box_ts  # [N_total, boxsize]
        assert box_ts is not None

        if n_root_nodes is None:
            n_root_nodes = mem.size(0)
        if n_root_nodes < n_target_nodes:
            raise ValueError(
                f"n_root_nodes ({n_root_nodes}) must be >= n_target_nodes ({n_target_nodes})"
            )
        if n_root_nodes > mem.size(0):
            raise ValueError(
                f"n_root_nodes ({n_root_nodes}) exceeds flat_graph size ({mem.size(0)})"
            )

        mem = mem[:n_root_nodes]
        ts = ts[:n_root_nodes]
        box = box[:n_root_nodes]
        box_ts = box_ts[:n_root_nodes]

        batch_size = mem.size(0)

        # 1. Compute Query (from current Memory) -> [N, H, D_head]
        Q = self.w_q(mem).view(batch_size, self.att_h, self.head_dim)

        # 2. Time Encoding for Mails
        if self.time_enc is not None:
            # Time diff: t_curr - t_mail
            # ts.unsqueeze(1) is [N, 1], box_ts is [N, L]
            dt = ts.unsqueeze(1) - box_ts

            # Since time_enc might expect a 1D tensor depending on its implementation,
            # we flatten, encode, and reshape back.
            time_feat = self.time_enc(dt.view(-1))  # [N * L, dim_time]
            time_feat = time_feat.view(batch_size, self.boxsize, self.dim_time)

            # Concat time to mails: [N, L, d_mail + d_time]
            box_input = torch.cat([box, time_feat], dim=-1)
        else:
            box_input = box

        # 3. Compute Key and Value (from Mails) -> [N, L, H, D_head]
        K = self.w_k(box_input).view(
            batch_size, self.boxsize, self.att_h, self.head_dim
        )
        V = self.w_v(box_input).view(
            batch_size, self.boxsize, self.att_h, self.head_dim
        )

        # 4. Attention Score & Weights
        # Q: [N, 1, H, D_head] * K: [N, L, H, D_head] -> sum over D_head -> [N, L, H]
        att_score = (Q.unsqueeze(1) * K).sum(dim=-1)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)

        # Softmax over L (mailbox items) dimension
        att_weights = F.softmax(att_score, dim=1)
        att_weights = self.dropout(att_weights)

        # 5. Aggregate
        # Weighted sum of V: [N, L, H, 1] * [N, L, H, D_head] -> sum over L -> [N, H, D_head]
        rst = (att_weights.unsqueeze(-1) * V).sum(dim=1)
        rst = rst.view(batch_size, self.dim_mem)  # [N, dim_mem]

        # 6. Residual + Norm + MLP
        rst = self.layer_norm(rst + mem)
        rst = self.mlp(rst)
        rst = self.dropout(rst)
        rst = F.relu(rst)

        # Only root rows participate in APAN memory updates; neighbor rows stay untouched.
        updated_mem = flat_graph.mem.clone()
        updated_mem[:n_root_nodes] = rst
        flat_graph.mem = updated_mem

        # Cache detached states for the Root nodes
        flat_graph.detached_mem = flat_graph.mem[:n_target_nodes].detach().clone()
        flat_graph.detached_ts = flat_graph.ts[:n_target_nodes].detach().clone()
        flat_graph.detached_nid = flat_graph.nid[:n_target_nodes].detach().clone()

        return flat_graph
