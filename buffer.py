from dataclasses import dataclass

import torch

from schemas.datatypes.misc import Shared
from worker.receiver import Receiver as Receiver


@dataclass(kw_only=True)
class DyBuf:
    data: torch.Tensor
    nid: torch.Tensor
    ts: torch.Tensor


@dataclass(kw_only=True)
class StBuf:
    data: torch.Tensor
    nid: torch.Tensor


@dataclass(kw_only=True)
class Buffers:
    pos_efeat_h2d: None | torch.Tensor
    nfeat_h2d: None | StBuf
    efeat_h2d: None | torch.Tensor
    mem_h2d: None | DyBuf
    box_h2d: None | DyBuf
    mem_d2h: None | DyBuf
    box_d2h: None | DyBuf


def calculate_gnn_graph_size(batch_size: int, fanouts: list[int]) -> tuple[int, int]:
    """
    计算 GNN 采样在最坏情况（无重叠）下的节点和边数量。
    假设 fanouts 顺序对应 DGL/PyG 的层顺序：
    fanouts[0] 是第一层 GNN (离 Input 最近 / 离 Output 最远)
    fanouts[-1] 是最后一层 GNN (离 Input 最远 / 离 Output 最近)

    注意：如果你的 config [10, 25] 含义是 [Hop1, Hop2]，请将输入 reverse 为 [25, 10]
    """

    # 这里的 nodes 指的是“作为下一层输入的目标节点”
    # 初始状态：输出层节点 (Seed Nodes)
    current_layer_nodes = batch_size

    total_edges = 0
    input_nodes = 0

    # print(f"--- 计算详情 (Batch Size = {batch_size}) ---")
    # print(f"Output Layer (Seed Nodes): {batch_size:,}")

    # 通常采样是从 Output 往 Input 倒推的
    # fanouts[-1] 是连接 Output 的那一层
    # fanouts[0] 是最底层

    # 为了方便计算，我们从 Output (最后一层) 向 Input (第一层) 遍历
    # 假设 fanouts = [10, 25]，在 DGL 中通常意味着:
    # Layer 1 (Input->Hidden): 采样 10
    # Layer 2 (Hidden->Output): 采样 25
    # 所以倒推顺序是：先采 25 (Layer 2)，再对这 25*B 个点采 10 (Layer 1)

    reversed_fanouts = list(reversed(fanouts))

    cumulative_nodes = batch_size  # 包含 Output nodes

    for i, fanout in enumerate(reversed_fanouts):
        layer_idx = len(fanouts) - 1 - i

        # 这一层的边数 = 当前目标节点数 * 采样数
        edges_count = current_layer_nodes * fanout
        total_edges += edges_count

        # 这一层的源节点数 (Src Nodes) = 这一层的边数 (在无重叠假设下)
        src_nodes = edges_count

        # print(f"Layer {layer_idx} (Fanout {fanout}):")
        # print(f"  - Edges: {current_layer_nodes:,} * {fanout} = {edges_count:,}")
        # print(f"  - Input Nodes needed: {src_nodes:,}")

        # 更新下一轮的目标节点
        current_layer_nodes = src_nodes
        cumulative_nodes += src_nodes

    # 最底层的 src_nodes 就是最终需要加载特征的节点数
    final_input_nodes = current_layer_nodes

    # print("-" * 30)
    # print(f"【总结 (Batch Size={batch_size}, Config={fanouts})】")
    # print(f"1. 最大输入节点数 (Input Nodes): {final_input_nodes:,}")
    # print("   (需要加载 Feature 的节点数量)")
    # print(f"2. 最大总边数 (Total Edges):     {total_edges:,}")
    # print("   (MFG 图中的总边数)")

    return final_input_nodes, total_edges


def init_all_pin_mem(
    shared: Shared | None,
    shared_nfeat: torch.Tensor | None,
    shared_efeat: torch.Tensor | None,
    batch_size: int,
    fanout: list[int] | None,
    n_neg: int,
) -> Buffers:
    total_pos_edge = batch_size * (1 + n_neg)
    if fanout is not None:
        final_input_nodes, total_gnn_edges = calculate_gnn_graph_size(
            batch_size, fanout
        )
    else:
        final_input_nodes, total_gnn_edges = batch_size * 3, 1000
    pos_efeat_h2d = None
    nfeat_h2d = None
    efeat_h2d = None
    mem_h2d = None
    box_h2d = None
    mem_d2h = None
    box_d2h = None
    if shared_nfeat is not None:
        nfeat_h2d = _init_st_buf(
            dim=shared_nfeat.shape[1],
            max_num=int(final_input_nodes * 0.2),
            dtype=shared_nfeat.dtype,
        )
    if shared_efeat is not None:
        pos_efeat_h2d = _init_buf(
            dim=shared_efeat.shape[1],
            max_num=total_pos_edge,
            dtype=shared_efeat.dtype,
        )
        efeat_h2d = _init_buf(
            dim=shared_efeat.shape[1],
            max_num=total_gnn_edges,
            dtype=shared_efeat.dtype,
        )
    if shared is not None:
        shared_mem_dim = shared.mem.shape[1]
        mem_h2d = _init_mem_buf(
            dim=shared_mem_dim,
            max_num=int(final_input_nodes * 0.2),
            dtype=shared.mem.dtype,
        )
        mem_d2h = _init_mem_buf(
            dim=shared_mem_dim,
            max_num=int(final_input_nodes * 0.2),
            dtype=shared.mem.dtype,
        )
        shared_box_shape_tuple = (shared.box.shape[1], shared.box.shape[2])

        box_h2d = _init_box_buf(
            dim=shared_box_shape_tuple,
            max_num=int(final_input_nodes * 0.2),
            dtype=shared.box.dtype,
        )
        box_d2h = _init_box_buf(
            dim=shared_box_shape_tuple,
            max_num=int(final_input_nodes * 0.2),
            dtype=shared.box.dtype,
        )
    buffers = Buffers(
        pos_efeat_h2d=pos_efeat_h2d,
        nfeat_h2d=nfeat_h2d,
        efeat_h2d=efeat_h2d,
        mem_h2d=mem_h2d,
        box_h2d=box_h2d,
        mem_d2h=mem_d2h,
        box_d2h=box_d2h,
    )
    return buffers


def _init_buf(
    dim: int | tuple[int, ...] | list[int],
    max_num: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    shape = (max_num, dim) if isinstance(dim, int) else (max_num, *dim)

    pin_mem = torch.empty(
        shape,
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )
    return pin_mem


def _init_st_buf(
    dim: int,
    max_num: int,
    dtype: torch.dtype,
) -> StBuf:
    shape_for_data = (max_num, dim)

    pin_for_data = torch.empty(
        shape_for_data,
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )
    shape_for_nid = max_num
    pin_for_nid = torch.empty(
        shape_for_nid,
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )

    return StBuf(data=pin_for_data, nid=pin_for_nid)


def _init_mem_buf(
    dim: int,
    max_num: int,
    dtype: torch.dtype,
) -> DyBuf:
    pin_for_data = torch.empty(
        (max_num, dim),
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )
    pin_for_nid = torch.empty(
        max_num,
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )

    pin_for_ts = torch.empty(
        max_num,
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )
    return DyBuf(data=pin_for_data, nid=pin_for_nid, ts=pin_for_ts)


def _init_box_buf(
    dim: tuple[int, int],
    max_num: int,
    dtype: torch.dtype,
) -> DyBuf:
    pin_for_data = torch.empty(
        (max_num, *dim),
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )

    pin_for_nid = torch.empty(
        max_num,
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )

    boxsize = dim[0]

    pin_for_ts = torch.empty(
        (max_num, boxsize),
        dtype=dtype,
        pin_memory=True,  # Pinned memory for faster  transfer
    )
    return DyBuf(data=pin_for_data, nid=pin_for_nid, ts=pin_for_ts)
