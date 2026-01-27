import logging
import math

import torch
import torch.nn as nn

from schemas.datatypes.data_to_sync import MemTsNid
from schemas.datatypes.plan import BatchNodeStateIO
from tensor_set import TensorSet

from ..components import TimeEncode


logger = logging.getLogger(__name__)


# --- Helper Modules (Copied & Simplified from layers.py) ---


class JODIETimeEmbedding(nn.Module):
    """
    Specific to JODIE: Projects the static embedding based on time elapsed
    since the last update to predict the future state.
    """

    def __init__(self, dim_out: int) -> None:
        "dim_out = gnn dim_out = dim memory"
        super().__init__()
        self.time_emb = nn.Linear(1, dim_out)
        stdv = 1.0 / math.sqrt(self.time_emb.in_features)
        with torch.no_grad():
            self.time_emb.weight.normal_(0.0, stdv)
            self.time_emb.bias.normal_(0.0, stdv)

    def forward(
        self, h: torch.Tensor, mem_ts: torch.Tensor, ts: torch.Tensor
    ) -> torch.Tensor:
        # Time difference normalization as per JODIE paper/code
        time_diff = (ts - mem_ts) / (ts + 1)
        # Scaling the embedding: h * (1 + w * DyX_t)
        # h dim: [n_nodes, dim_mem]

        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst


def analyze_need(root_nodes: TensorSet) -> BatchNodeStateIO:
    return BatchNodeStateIO(
        mem_needed=root_nodes,
        mem_updated=root_nodes,
        mailbox_needed=None,
        mailbox_updated=None,
        nfeat_needed=root_nodes,
    )


class JODIE(nn.Module):
    def __init__(
        self, dim_node: int, dim_memory: int = 100, dim_time: int = 100
    ) -> None:
        """
        Args:
            dim_node: Dimension of raw node features (nfeat).
            dim_edge: Dimension of edge features (efeat).
            dim_memory: Dimension of memory / hidden state.
            dim_time: Dimension of time encoding.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dim_node = dim_node

        if dim_time > 0:
            self.memory_time_enc = TimeEncode(dim_time)

        self.node_feat_map = (
            nn.Linear(dim_node, dim_memory)
            if dim_node > 0 and dim_node != dim_memory
            else None
        )

        # --- 2. GNN / Embedding Layer ---
        # JODIE.yml: arch='identity', time_transform='JODIE'
        self.layer_norm = nn.LayerNorm(dim_memory)
        self.jodie_time_proj = JODIETimeEmbedding(dim_out=dim_memory)

    def forward(
        self,
        mem: MemTsNid,
        nfeat: torch.Tensor | None,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        mfgs: List of DGL Blocks.
              mfgs[0] is the target block containing memory and mailbox info.
        """
        # JODIE typically has 1 layer (Identity), so we focus on mfgs[0]
        updated_memory: torch.Tensor = mem.data
        ts = ts.repeat_interleave(2)  # because mem contains src + pos dst

        # 4. Combine with Node Features (Static)
        # JODIE.yml: combine_node_feature=True
        # Logic from RNNMemeoryUpdater:
        if self.dim_node > 0:
            # Assumes block.srcdata['nfeat'] holds the raw node features initially
            node_emb = self.node_feat_map(nfeat) if self.node_feat_map else nfeat

            h_combined = updated_memory + node_emb
        else:
            h_combined = updated_memory

        # In a real training loop, you'd also update the global memory storage here
        # or return updated_memory to be handled by the trainer.

        # --- Step 2: GNN / Embedding Transform ---
        # 2a. Normalization (IdentityNormLayer)
        h = self.layer_norm(h_combined)
        # h dim : [n_nodes, dim_memory]

        # 2b. JODIE Time Projection
        # Projects the embedding to the *current* interaction time `ts`
        # Using the *last* memory update time `mem_ts`
        # Note: JODIE paper projects the embedding *before* interaction based on elapsed time.
        h_projected = self.jodie_time_proj(h, mem.ts, ts)

        return h_projected
