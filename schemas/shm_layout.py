from enum import IntEnum

import torch


def calculate_gnn_graph_size(batch_size: int, fanouts: list[int]) -> tuple[int, int]:
    """
    计算 GNN 采样在最坏情况（无重叠）下的节点和边数量。
    返回: (max_input_nodes, total_edges)
    """
    current_layer_nodes = batch_size
    total_edges = 0

    # 从 Output 向 Input 倒推
    reversed_fanouts = list(reversed(fanouts))

    for fanout in reversed_fanouts:
        # 这一层的边数 = 当前目标节点数 * 采样数
        edges_count = current_layer_nodes * fanout
        total_edges += edges_count

        # 下一层的源节点数
        current_layer_nodes = edges_count

    final_input_nodes = current_layer_nodes
    return final_input_nodes, total_edges


def get_shm_layout_config(
    batch_size: int, fanouts: list[int], history: int = 1
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """
    生成 SHM 配置: Key -> (Shape, Dtype)
    """
    # 预估最大尺寸
    max_input_nodes, total_edges_per_snapshot = calculate_gnn_graph_size(
        batch_size, fanouts
    )
    SAFETY_MARGIN = 2.0
    MAX_EDGES = int(total_edges_per_snapshot * history * SAFETY_MARGIN)
    MAX_NODES = int((total_edges_per_snapshot + batch_size) * history * SAFETY_MARGIN)

    META_LEN = 2000  # 用于 sections, dims 等

    layout = {}

    def add(k: int, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        # 统一转为 str(int) 作为 key
        layout[str(int(k))] = (shape, dtype)

    ROOT_FACTOR = 6
    # --- 1. Basic ---
    add(ShmKeys.ROOT_NID, (batch_size * ROOT_FACTOR,), torch.long)
    add(
        ShmKeys.ROOT_TS, (batch_size * ROOT_FACTOR,), torch.float64
    )  # 时间戳通常是 float64
    add(ShmKeys.POS_EDGE_ID, (batch_size * ROOT_FACTOR,), torch.long)

    # --- 2. N2R ---
    add(ShmKeys.N2R_NODES, (MAX_NODES,), torch.long)
    add(ShmKeys.N2R_TS, (MAX_NODES,), torch.float64)
    add(ShmKeys.N2R_EID, (MAX_EDGES,), torch.long)
    add(ShmKeys.N2R_DTS, (MAX_EDGES,), torch.float64)  # 边时间/特征
    add(ShmKeys.N2R_COL, (MAX_EDGES,), torch.long)
    add(ShmKeys.N2R_ROW, (MAX_EDGES,), torch.long)
    # 元数据 (必须加)
    add(ShmKeys.N2R_NODE_SECTIONS, (META_LEN,), torch.long)
    add(ShmKeys.N2R_EDGE_SECTIONS, (META_LEN,), torch.long)
    add(ShmKeys.N2R_DIMS, (META_LEN,), torch.long)

    # --- 3. R2N (同上) ---
    add(ShmKeys.R2N_NODES, (MAX_NODES,), torch.long)
    add(ShmKeys.R2N_TS, (MAX_NODES,), torch.float64)
    add(ShmKeys.R2N_EID, (MAX_EDGES,), torch.long)
    add(ShmKeys.R2N_DTS, (MAX_EDGES,), torch.float64)
    add(ShmKeys.R2N_COL, (MAX_EDGES,), torch.long)
    add(ShmKeys.R2N_ROW, (MAX_EDGES,), torch.long)
    add(ShmKeys.R2N_NODE_SECTIONS, (META_LEN,), torch.long)
    add(ShmKeys.R2N_EDGE_SECTIONS, (META_LEN,), torch.long)
    add(ShmKeys.R2N_DIMS, (META_LEN,), torch.long)

    # --- 4. Plan: Mem ---
    add(ShmKeys.MEM_P2P, (MAX_NODES,), torch.long)
    add(ShmKeys.MEM_FETCH, (MAX_NODES,), torch.long)
    add(ShmKeys.MEM_WB, (MAX_NODES,), torch.long)
    # 元数据
    add(ShmKeys.MEM_P2P_SIZES, (META_LEN,), torch.long)
    add(ShmKeys.MEM_REC_CTS, (META_LEN,), torch.long)  # 原 numpy 数组转 long tensor
    add(ShmKeys.MEM_SEND_CTS, (META_LEN,), torch.long)

    # --- 5. Plan: Box ---
    add(ShmKeys.BOX_P2P, (MAX_NODES,), torch.long)
    add(ShmKeys.BOX_FETCH, (MAX_NODES,), torch.long)
    add(ShmKeys.BOX_WB, (MAX_NODES,), torch.long)
    add(ShmKeys.BOX_P2P_SIZES, (META_LEN,), torch.long)
    add(ShmKeys.BOX_REC_CTS, (META_LEN,), torch.long)
    add(ShmKeys.BOX_SEND_CTS, (META_LEN,), torch.long)

    # --- 6. Plan: Nfeat ---
    add(ShmKeys.NFEAT_CACHE, (MAX_NODES,), torch.long)
    add(ShmKeys.NFEAT_FETCH, (MAX_NODES,), torch.long)

    return layout


class ShmKeys(IntEnum):
    """
    共享内存键位映射表 (Protocol)
    定义每个数据字段在 Manifest Tensor 中的索引位置
    """

    # --- Header ---
    BATCH_IDX = 0  # 特殊位：存放 batch_idx 或 结束信号 (-1/-2)

    # --- Root / Pos (Meta 基础信息) ---
    ROOT_NID = 1
    ROOT_TS = 2
    POS_EDGE_ID = 3

    # --- 新增: EpochSplitIterInfo (m_state) ---
    MS_CUR_EPOCH = 4
    MS_N_EPOCH = 5
    MS_CUR_MACRO_SPLIT = 6
    MS_N_MACRO_SPLIT = 7
    MS_CUR_MACRO_EPOCH = 8
    MS_SPLIT_TYPE = 9  # 0: train, 1: val, 2: test

    # --- N2R: PackedDGLBlockMeta (Sampled Subgraphs) ---
    N2R_NODES = 10
    N2R_TS = 11
    N2R_EID = 12
    N2R_DTS = 13
    N2R_COL = 14
    N2R_ROW = 15
    N2R_NODE_SECTIONS = 16  # Meta: 节点切分长度
    N2R_EDGE_SECTIONS = 17  # Meta: 边切分长度
    N2R_DIMS = 18  # Meta: 输入输出维度

    # --- R2N: PackedDGLBlockMeta (Mailbox/Temporal Neighbor) ---
    R2N_NODES = 20
    R2N_TS = 21
    R2N_EID = 22
    R2N_DTS = 23
    R2N_COL = 24
    R2N_ROW = 25
    R2N_NODE_SECTIONS = 26
    R2N_EDGE_SECTIONS = 27
    R2N_DIMS = 28
    R2N_MDT = 29  # [新增] Mail Delivery Tensor (如果有)

    # --- Plan: Mem (DyPlan) ---
    MEM_P2P = 30
    MEM_FETCH = 31
    MEM_WB = 32
    MEM_P2P_SIZES = 33  # Meta: P2P 切分长度
    MEM_REC_CTS = 34  # Meta: 接收计数 (原 numpy)
    MEM_SEND_CTS = 35  # Meta: 发送计数 (原 numpy)

    # --- Plan: Box (DyPlan) ---
    BOX_P2P = 40
    BOX_FETCH = 41
    BOX_WB = 42
    BOX_P2P_SIZES = 43
    BOX_REC_CTS = 44
    BOX_SEND_CTS = 45

    # --- Plan: Nfeat (StPlan) ---
    NFEAT_CACHE = 50
    NFEAT_FETCH = 51

    # --- Size ---
    TOTAL_KEYS = 60  # Manifest Tensor 的总长度


# 这是一个用于通信的 Tensor 长度
MANIFEST_SIZE = ShmKeys.TOTAL_KEYS
