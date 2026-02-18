import logging

import torch
from dgl.heterograph import DGLBlock

from buffer import StBuf  # Import StBuf
from schemas.datatypes.misc import NFeatNid
from schemas.datatypes.plan import StPlan


logger = logging.getLogger(__name__)


class StAssembler:
    """
    静态数据组装器线程，将分块数据和增量数据合并，并填充到 DGL Block 中。
    """

    nfeat_delta: NFeatNid | None = None

    def __init__(
        self,
        shared_nfeat: torch.Tensor | None,
        shared_efeat: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        self.nfeat_delta = None
        self.device = device
        self.shared_nfeat = shared_nfeat
        self.shared_efeat = shared_efeat
        super().__init__()

    def fill_block_with_efeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        efeat: torch.Tensor,
        eids: list[list[torch.Tensor]],
        pin_efeat_buf: torch.Tensor | None = None,
    ) -> list[list[DGLBlock]]:
        # 1. 收集所有 Block 的 EID (CPU) 并记录大小
        all_eids = []
        block_sizes = []

        # 注意：这里我们假设 blocks 和 eids 的结构是一一对应的
        # 即使 blocks 已经在 GPU 上，只要 eids 还在 CPU 上，我们就能在这里 gather
        for group_idx, (blocks, eid_group) in enumerate(zip(gnn_mfgs, eids)):
            for block_idx, (block, eid) in enumerate(zip(blocks, eid_group)):
                all_eids.append(eid)
                block_sizes.append(eid.numel())

        if not all_eids:
            return gnn_mfgs

        # 2. 拼接成一个大 Tensor (CPU) 用于 Gather
        flat_eids = torch.cat(all_eids)
        total_edges = flat_eids.numel()

        # 3. 准备 GPU 上的数据 (通过 Pinned Memory 加速传输)
        gpu_all_efeat: torch.Tensor

        if pin_efeat_buf is not None and total_edges <= pin_efeat_buf.shape[0]:
            # --- Fast Path: 使用 Pinned Memory ---

            # 3a. Gather: CPU Shared Mem -> CPU Pinned Buffer
            buf_view = pin_efeat_buf[:total_edges]

            # (Debug 检查保留)
            if __debug__:
                assert flat_eids.numel() > 0
                assert flat_eids.max().item() < efeat.size(0)

            torch.index_select(efeat, 0, flat_eids, out=buf_view)

            # 3b. Copy: CPU Pinned -> GPU (Async)
            # 因为 blocks 在 GPU 上，所以我们也必须把数据搬到 GPU
            gpu_all_efeat = buf_view.to(self.device, non_blocking=False)

        else:
            # print total_edges <= pin_efeat_buf.shape[0]
            logger.warning(
                "StAsm: Pinned buffer insufficient or missing for efeat, using fallback path"
            )
            # --- Fallback: 普通路径 ---
            # 直接 Gather 并 .to(device)
            # 注意：如果 blocks 在 GPU，这里必须显式 to(device)
            cpu_efeat = efeat[flat_eids]
            gpu_all_efeat = cpu_efeat.to(self.device, non_blocking=False)

        # 4. 在 GPU 上切分并赋值
        # 此时 gpu_all_efeat 和 block 都在 GPU 上，赋值是合法的
        curr = 0
        idx = 0
        for blocks in gnn_mfgs:
            for block in blocks:
                size = block_sizes[idx]

                # Narrow 切片 (Zero-Copy)
                block_feat = gpu_all_efeat.narrow(0, curr, size)

                # 赋值给 edata
                # 因为 block 已经在 GPU 上，且 block_feat 也在 GPU 上，DGL 不会报错
                block.edata["efeat"] = block_feat

                curr += size
                idx += 1

        return gnn_mfgs

    def fill_block_with_nfeat(
        self, gnn_mfgs_0: list[DGLBlock], nfeat: NFeatNid
    ) -> list[DGLBlock]:
        assert nfeat.is_sorted  # make sure nfeat.nid is sorted

        for idx, block in enumerate(gnn_mfgs_0):
            global_nid: torch.Tensor = block.srcdata["ID"]

            permute_nidx = torch.searchsorted(nfeat.nid, global_nid)
            block.srcdata["nfeat"] = nfeat.nfeat[permute_nidx]  # 这里放memory

        return gnn_mfgs_0

    def prepare_nfeat_cache(
        self,
        nfeat_plan: StPlan,
        pin_nfeat_buf: StBuf | None = None,  # 接收 StBuf
    ) -> NFeatNid | None:
        pin_nfeat_buf = None
        assert self.shared_nfeat is not None
        to_cache, fetch = nfeat_plan.to_cache.data, nfeat_plan.fetch.data

        nfeat_obj = None

        if fetch.numel() == 0:
            nfeat_obj = self.nfeat_delta
            assert nfeat_obj is not None, (
                "NFeat delta should not be None when fetch is empty"
            )

            nfeat_obj = nfeat_obj.sort_by_nid()

            if to_cache.numel() > 0:
                self.nfeat_delta = nfeat_obj.filter_by_nid(to_cache)
            else:
                self.nfeat_delta = None
            return nfeat_obj
        # --- Pin Memory Logic ---
        if pin_nfeat_buf is not None and fetch.numel() <= pin_nfeat_buf.data.shape[0]:
            n = fetch.numel()
            buf_data = pin_nfeat_buf.data[:n]
            buf_nid = pin_nfeat_buf.nid[:n]

            # Gather Data
            if __debug__:
                # 1. 确保 fetch 索引张量不为空
                assert fetch.numel() > 0, f"{fetch}, Fetch indices tensor is empty"

                # 2. 核心边界检查：确保 fetch 里的所有节点 ID 都在共享内存范围内
                # self.shared_nfeat.size(0) 通常等于全局节点数 n_nodes
                max_node_id = fetch.max().item()
                min_node_id = fetch.min().item()
                assert max_node_id < self.shared_nfeat.size(0), (
                    f"Node ID {max_node_id} out of bounds for shared_nfeat size {self.shared_nfeat.size(0)}"
                )
                assert min_node_id >= 0, (
                    f"Negative Node ID {min_node_id} detected in fetch"
                )

                # 3. 检查输出缓冲区 buf_data 的第一维是否与 fetch 的数量匹配
                assert buf_data.size(0) == fetch.numel(), (
                    f"Buffer size mismatch: buf_data={buf_data.size(0)}, fetch={fetch.numel()}"
                )
            torch.index_select(self.shared_nfeat, 0, fetch, out=buf_data)

            # Gather NID
            # buf_nid 是 (Max, 1), fetch 是 (N,)
            buf_nid.copy_(fetch.to(dtype=buf_nid.dtype))

            # Async Transfer
            gpu_data = buf_data.to(self.device, non_blocking=False)
            gpu_nid = buf_nid.to(self.device, non_blocking=False).long()

            nfeat_obj = NFeatNid(nid=gpu_nid, nfeat=gpu_data)
        elif pin_nfeat_buf is not None:
            pass
        # ------------------------

        # Fallback
        if nfeat_obj is None:
            # show error info for debug
            logger.warning(
                "StAsm: Falling back to regular path for nfeat due to insufficient pinned buffer"
            )
            nfeat_obj = NFeatNid(nid=fetch, nfeat=self.shared_nfeat[fetch])
            nfeat_obj = nfeat_obj.cuda()
            # 注意: 普通路径下数据在 CPU，后续如果需要 GPU 数据，
            # fill_mfg_with_efeat_n_nfeat 或 H2D 逻辑可能需要处理 .to(cuda)
            # 为了一致性，建议此处不上 GPU，由 H2D 统一调度，或者在这里统一上 GPU。
            # 假设 H2D 现在的逻辑是可以处理 GPU Tensor 的。

        if self.nfeat_delta is not None:
            nfeat_obj = nfeat_obj.merge(self.nfeat_delta)

        nfeat_obj = nfeat_obj.sort_by_nid()

        if to_cache.numel() > 0:
            self.nfeat_delta = nfeat_obj.filter_by_nid(to_cache)
        else:
            self.nfeat_delta = None

        return nfeat_obj

    def fill_mfg_with_nfeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        nfeat: NFeatNid | None,
    ) -> list[list[DGLBlock]]:
        # 这一部分是负责nfeat重用的,不管是mfg版本还是no_mfg版本,只要有nfeat都要做

        if self.shared_nfeat is not None:
            assert nfeat is not None
            gnn_mfgs[0] = self.fill_block_with_nfeat(gnn_mfgs[0], nfeat)

        return gnn_mfgs

    def fill_mfg_with_efeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        eids: list[list[torch.Tensor]],
        pin_efeat_buf: torch.Tensor | None = None,  # 记得在接口这里也透传 buf
    ) -> list[list[DGLBlock]]:
        # 这一部分是负责nfeat重用的,不管是mfg版本还是no_mfg版本,只要有nfeat都要做

        if self.shared_efeat is not None:
            gnn_mfgs = self.fill_block_with_efeat(
                gnn_mfgs, self.shared_efeat, pin_efeat_buf=pin_efeat_buf, eids=eids
            )

        return gnn_mfgs
