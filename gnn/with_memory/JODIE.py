import math

import torch
import torch.nn as nn

from schemas.datatypes.plan import BatchNodeStateIO

from ..components import TimeEncode


# --- Helper Modules (Copied & Simplified from layers.py) ---


class JODIETimeEmbedding(nn.Module):
    """
    Specific to JODIE: Projects the static embedding based on time elapsed
    since the last update to predict the future state.
    """

    def __init__(self, dim_out):
        "dim_out = gnn dim_out = dim memory"
        super(JODIETimeEmbedding, self).__init__()
        self.time_emb = nn.Linear(1, dim_out)
        stdv = 1.0 / math.sqrt(self.time_emb.in_features)
        with torch.no_grad():
            self.time_emb.weight.normal_(0.0, stdv)
            self.time_emb.bias.normal_(0.0, stdv)

    def forward(self, h, mem_ts, ts):
        # Time difference normalization as per JODIE paper/code
        time_diff = (ts - mem_ts) / (ts + 1)
        # Scaling the embedding: h * (1 + w * delta_t)
        # h dim: [n_nodes, dim_mem]

        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst


class JODIE(nn.Module):
    def __init__(self, dim_node, dim_memory=100, dim_time=100):
        """
        Args:
            dim_node: Dimension of raw node features (nfeat).
            dim_edge: Dimension of edge features (efeat).
            dim_memory: Dimension of memory / hidden state.
            dim_time: Dimension of time encoding.
            dropout: Dropout rate.
        """
        super(JODIE, self).__init__()
        self.dim_node = dim_node

        if dim_time > 0:
            self.memory_time_enc = TimeEncode(dim_time)

        if dim_node > 0 and dim_node != dim_memory:
            self.node_feat_map = nn.Linear(dim_node, dim_memory)
        else:
            self.node_feat_map = None

        # --- 2. GNN / Embedding Layer ---
        # JODIE.yml: arch='identity', time_transform='JODIE'
        self.layer_norm = nn.LayerNorm(dim_memory)
        self.jodie_time_proj = JODIETimeEmbedding(dim_out=dim_memory)

        # --- 3. Edge Predictor ---

    @staticmethod
    def analyze_need(root_nodes):
        all_nodes = root_nodes
        return BatchNodeStateIO(
            mem_needed=all_nodes,
            mem_updated=all_nodes,
            mailbox_needed=None,
            mailbox_updated=None,
            nfeat_needed=all_nodes,
        )

    @staticmethod
    def sample(ngh_sampler, src, dst, neg_dst, ts):
        roots = torch.cat([src, dst, neg_dst])
        all_nodes = root_nodes = set(roots.numpy().tolist())
        batch_node_state_io = JODIE.analyze_need(root_nodes)
        return all_nodes, root_nodes, batch_node_state_io

    def forward(
        self,
        mfgs,
    ):
        """
        mfgs: List of DGL Blocks.
              mfgs[0] is the target block containing memory and mailbox info.
        """
        # JODIE typically has 1 layer (Identity), so we focus on mfgs[0]
        block = mfgs[0][0]
        updated_memory: torch.Tensor = block.srcdata["updated_mem"]

        # 4. Combine with Node Features (Static)
        # JODIE.yml: combine_node_feature=True
        # Logic from RNNMemeoryUpdater:
        if self.dim_node > 0:
            # Assumes block.srcdata['nfeat'] holds the raw node features initially
            nfeat = block.srcdata["nfeat"]
            if self.node_feat_map:
                node_emb = self.node_feat_map(nfeat)
            else:
                node_emb = nfeat
            h_combined = updated_memory + node_emb
        else:
            h_combined = updated_memory

        # Store results back to block for the next step
        block.srcdata["h"] = h_combined

        # In a real training loop, you'd also update the global memory storage here
        # or return updated_memory to be handled by the trainer.

        # --- Step 2: GNN / Embedding Transform ---
        # 2a. Normalization (IdentityNormLayer)
        h = self.layer_norm(block.srcdata["h"])
        # h dim : [n_nodes, dim_memory]

        # 2b. JODIE Time Projection
        # Projects the embedding to the *current* interaction time `ts`
        # Using the *last* memory update time `mem_ts`
        # Note: JODIE paper projects the embedding *before* interaction based on elapsed time.
        h_projected = self.jodie_time_proj(
            h,
            block.srcdata["mem_ts"],  # Time of the memory we just used/updated
            block.srcdata["ts"],  # Current interaction time
        )

        return h_projected
