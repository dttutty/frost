import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from gnn.sample_mixin import SampleMixin

from ..components import TimeEncode


class NeighborCooccurrenceEncoder(nn.Module):
    def __init__(self, dim_out):
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )

    def forward(self, src_nid, dst_nid):
        """
        计算 Src 和 Dst 历史序列中邻居的共现次数并编码。
        src_nid: [Batch, Seq_Len]
        dst_nid: [Batch, Seq_Len]
        """
        # 1. 计算共现矩阵 (Batch-wise BroadCasting)
        # [B, L, 1] == [B, 1, L] -> [B, L, L] (True where neighbors match)
        # 注意：这里需要 float() 转换来做矩阵乘法或求和
        overlap = (src_nid.unsqueeze(2) == dst_nid.unsqueeze(1)).float()

        # 忽略 Padding (nid=0)
        src_mask = (src_nid != 0).float().unsqueeze(-1)  # [B, L, 1]
        dst_mask = (dst_nid != 0).float().unsqueeze(-1)  # [B, L, 1]

        # Src 序列中每个邻居在 Dst 序列中出现的次数
        src_counts = overlap.sum(dim=2, keepdim=True) * src_mask
        # Dst 序列中每个邻居在 Src 序列中出现的次数
        dst_counts = overlap.sum(dim=1, keepdim=True).transpose(1, 2) * dst_mask

        # 2. MLP 编码
        return self.mlp(src_counts).squeeze(2), self.mlp(dst_counts).squeeze(2)


class DyGFormerAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super(DyGFormerAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, src):
        # Pre-Norm Architecture usually more stable
        src2 = self.norm1(src)
        src2, _ = self.multi_head_attention(src2, src2, src2)
        src = src + self.dropout(src2)

        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + self.dropout(src2)
        return src


class DyGFormer(nn.Module, SampleMixin):
    def __init__(
        self,
        dim_node,
        dim_edge,
        dim_time=100,
        seq_len=10,  # 历史序列长度 (即采样器的 num_neighbors)
        patch_size=1,
        channel_dim=50,
        num_layers=2,
        heads=2,
        dropout=0.1,
    ):
        super(DyGFormer, self).__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.channel_dim = channel_dim

        assert seq_len % patch_size == 0, (
            "Sequence length must be divisible by patch size"
        )
        self.num_patches = seq_len // patch_size
        self.total_dim = 4 * channel_dim  # 4 Channels: Node, Edge, Time, Co-occur

        # Encoders
        self.time_enc = TimeEncode(dim_time)
        self.co_occ_enc = NeighborCooccurrenceEncoder(channel_dim)

        # Projections (Flattened Patch -> Channel Dim)
        self.proj_node = nn.Linear(patch_size * dim_node, channel_dim)
        self.proj_edge = nn.Linear(patch_size * dim_edge, channel_dim)
        self.proj_time = nn.Linear(patch_size * dim_time, channel_dim)
        self.proj_co_occ = nn.Linear(patch_size * channel_dim, channel_dim)

        # Transformer
        self.layers = nn.ModuleList(
            [
                DyGFormerAttention(self.total_dim, heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Outputs
        self.out_proj = nn.Linear(self.total_dim, dim_node)
        self.score_fc = nn.Linear(dim_node, 1)

    @staticmethod
    def collate_block_to_sequence(block, feat_manager, seq_len):
        """
        核心适配器：将 Sampler 的 Block (稀疏 COO) 转换为 Transformer 的 Sequence (稠密 Tensor)。
        不修改 C++ 代码，利用 PyTorch 操作在 Python 端高效完成。

        Args:
            block: sampler_core 返回的 TemporalGraphBlock 对象
            feat_manager: 管理原始特征的对象 (提供 .nfeat 和 .efeat)
            seq_len: 目标序列长度
        Returns:
            src_seq, dst_seq (Dicts containing dense tensors)
        """
        # 1. 解包 Block 数据
        # row: Root Node Index in Batch (0 ~ 2*BatchSize-1)
        # col: Neighbor Node Index in block.nodes
        row = torch.from_numpy(block.row()).long()
        col = torch.from_numpy(block.col()).long()
        eid = torch.from_numpy(block.eid()).long()
        # dts: Root_TS - Neighbor_TS (采样器已经算好了!)
        dts = torch.from_numpy(block.dts()).float()

        # 获取邻居的 Global Node ID
        all_nodes = torch.from_numpy(block.nodes()).long()
        neighbor_nids = all_nodes[col]  # [Total_Edges]

        # 2. 排序与分组 (Sorting & Grouping)
        # Sampler 返回的边是乱序的(多线程)，我们需要按 Root Node 归组
        # 使用 argsort 进行排序
        sort_idx = torch.argsort(row)

        sorted_row = row[sort_idx]
        sorted_neighbor_nids = neighbor_nids[sort_idx]
        sorted_eid = eid[sort_idx]
        sorted_dts = dts[sort_idx]

        # 3. 切分 (Split)
        # 计算每个 Root Node 有多少个邻居
        num_roots = block.dim_out()  # 2 * BatchSize
        counts = torch.bincount(sorted_row, minlength=num_roots).cpu()

        # 将平铺的 Tensor 切分成 List of Tensors
        list_nids = torch.split(sorted_neighbor_nids, counts.tolist())
        list_eids = torch.split(sorted_eid, counts.tolist())
        list_dts = torch.split(sorted_dts, counts.tolist())

        # 4. 填充 (Pad) -> [2*B, Seq_Len]
        # 注意：DyGFormer 需要定长序列。pad_sequence 默认补0。
        # 我们需要截断超过 seq_len 的，或者补齐不足的。

        def pad_and_stack(tensor_list):
            # 这里的逻辑是 Suffix Padding (数据靠前，0补在后面) 还是 Prefix?
            # 采样器 recent=True 意味着返回的是最近的。
            # 通常 Transformer 习惯将 Padding 放在后面 (batch_first=True)。
            padded = pad_sequence(tensor_list, batch_first=True, padding_value=0)

            # 截断或补齐到固定 seq_len
            B, L = padded.shape
            if L > seq_len:
                padded = padded[:, :seq_len]
            elif L < seq_len:
                pad_size = seq_len - L
                padded = F.pad(padded, (0, pad_size))
            return padded

        dense_nids = pad_and_stack(list_nids).to(all_nodes.device)
        dense_eids = pad_and_stack(list_eids).to(all_nodes.device)
        dense_dts = pad_and_stack(list_dts).to(all_nodes.device)

        # 5. 获取原始特征
        # 假设 feat_manager 支持通过 ID 索引特征
        dense_nfeat = feat_manager.nfeat[dense_nids]
        dense_efeat = feat_manager.efeat[dense_eids]

        # 6. 拆分 Src 和 Dst
        # 假设 Block 中前一半是 Src Roots，后一半是 Dst Roots
        batch_size = num_roots // 2

        def make_dict(start, end):
            return {
                "nid": dense_nids[start:end],
                "feat_n": dense_nfeat[start:end],
                "feat_e": dense_efeat[start:end],
                "dts": dense_dts[start:end],  # DyX Time (已经算好的)
            }

        src_seq = make_dict(0, batch_size)
        dst_seq = make_dict(batch_size, 2 * batch_size)

        return src_seq, dst_seq

    def get_patches(self, x):
        """[B, L, D] -> [B, Num_Patches, Patch_Size*D]"""
        B, L, D = x.shape
        x = x.reshape(B, self.num_patches, self.patch_size, D)
        x = x.reshape(B, self.num_patches, -1)
        return x

    def forward(self, block, feat_manager):
        """
        Args:
            block: 包含 (Src + Dst) 历史采样的 TemporalGraphBlock
            feat_manager: 特征存储对象
        """
        # 1. 适配器：Block -> Dense Sequence
        src_seq, dst_seq = self.collate_block_to_sequence(
            block, feat_manager, self.seq_len
        )

        # 2. 共现计算
        # [B, L, Channel]
        src_co, dst_co = self.co_occ_enc(src_seq["nid"], dst_seq["nid"])

        # 3. 处理每个分支 (Patches & Projection)
        def process_branch(seq, co_occ):
            # Patching & Projection
            p_node = self.proj_node(self.get_patches(seq["feat_n"]))
            p_edge = self.proj_edge(self.get_patches(seq["feat_e"]))
            p_co = self.proj_co_occ(self.get_patches(co_occ))

            # Time Encoding
            # block.dts 已经是 (Root_TS - Neighbor_TS)，正是我们要的 DyX
            # 需要处理 Padding (nid=0 的位置 dts 应该无效，mask掉或者由 time_enc 处理)
            time_emb = self.time_enc(seq["dts"]).squeeze(2)  # [B, L, Dt]
            # Mask out padding time (optional but recommended)
            time_emb[seq["nid"] == 0] = 0
            p_time = self.proj_time(self.get_patches(time_emb))

            # Stack Channels: [B, Patches, 4, Channel]
            stacked = torch.stack([p_node, p_edge, p_time, p_co], dim=2)
            # Flatten: [B, Patches, 4*Channel]
            return stacked.reshape(seq["nid"].shape[0], -1, self.total_dim)

        src_emb_in = process_branch(src_seq, src_co)
        dst_emb_in = process_branch(dst_seq, dst_co)

        # 4. Transformer (Concat Src & Dst)
        # [B, 2*Num_Patches, Dim]
        combined = torch.cat([src_emb_in, dst_emb_in], dim=1)

        for layer in self.layers:
            combined = layer(combined)

        # 5. Output Aggregation
        # Split back
        src_out = combined[:, : self.num_patches, :]
        dst_out = combined[:, self.num_patches :, :]

        # Mean Pooling over patches
        src_emb = self.out_proj(src_out.mean(dim=1))
        dst_emb = self.out_proj(dst_out.mean(dim=1))

        # 6. Score
        score = self.score_fc(F.relu(src_emb + dst_emb))
        return score, src_emb, dst_emb
