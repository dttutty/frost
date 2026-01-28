import logging

import torch
from dgl.heterograph import DGLBlock

from buffer import StBuf  # Import StBuf
from schemas.datatypes.misc import NFeatNid
from schemas.datatypes.plan import TensorStPlan


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
    ) -> None:
        self.nfeat_delta = None
        self.shared_nfeat = shared_nfeat
        self.shared_efeat = shared_efeat
        super().__init__()

    def fill_block_with_efeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        efeat: torch.Tensor,
        pin_efeat_buf: torch.Tensor | None = None,  # 新增参数
    ) -> list[list[DGLBlock]]:
        # --- 优化路径：使用 Pinned Memory ---
        if pin_efeat_buf is not None:
            # 1. 收集所有 Block 的 EID (Flatten)
            # 我们需要知道每个 block 有多少边，以便后续切分
            all_eids = []
            block_sizes = []

            for blocks in gnn_mfgs:
                for block in blocks:
                    eid = block.edata["ID"]
                    all_eids.append(eid)
                    block_sizes.append(eid.numel())

            if not all_eids:
                return gnn_mfgs

            # 拼接成一个大 Tensor (CPU)
            flat_eids = torch.cat(all_eids)
            total_edges = flat_eids.numel()

            # 2. 检查 Buffer 容量是否足够
            if total_edges <= pin_efeat_buf.shape[0]:
                # 截取 Buffer 视图
                buf_view = pin_efeat_buf[:total_edges]

                # Gather: CPU -> Pinned CPU
                # 这是一个大块的内存拷贝，比循环里每次拷一点要快
                torch.index_select(efeat, 0, flat_eids, out=buf_view)

                # Async Copy: Pinned CPU -> GPU
                gpu_all_efeat = buf_view.to("cuda", non_blocking=True)

                # 3. 切分并赋值回 Block
                # 此时 block 在 CPU, 但 edata['efeat'] 指向 GPU
                curr = 0
                idx = 0
                for blocks in gnn_mfgs:
                    for block in blocks:
                        size = block_sizes[idx]
                        # Narrow 是零拷贝切片
                        block.edata["efeat"] = gpu_all_efeat.narrow(0, curr, size)
                        curr += size
                        idx += 1

                return gnn_mfgs

        # --- Fallback: 原始 CPU 路径 ---
        # 如果没有 Buffer 或者 Buffer 不够大
        for blocks in gnn_mfgs:
            for block in blocks:
                global_eid: torch.Tensor = block.edata["ID"]
                # 这里产生的是 CPU Tensor
                block.edata["efeat"] = efeat[global_eid]

        return gnn_mfgs

    def fill_block_with_nfeat(
        self, gnn_mfgs_0: list[DGLBlock], nfeat: NFeatNid
    ) -> list[DGLBlock]:
        assert nfeat.is_sorted  # make sure nfeat.nid is sorted

        for block in gnn_mfgs_0:
            global_nid: torch.Tensor = block.srcdata["w"]
            permute_nidx = torch.searchsorted(nfeat.nid, global_nid)
            block.srcdata["nfeat"] = nfeat.nfeat[permute_nidx]  # 这里放memory

        return gnn_mfgs_0

    def prepare_nfeat_cache(
        self,
        nfeat_plan: TensorStPlan,
        pin_nfeat_buf: StBuf | None = None,  # 接收 StBuf
    ) -> NFeatNid | None:
        if self.shared_nfeat is not None:
            to_cache, fetch = nfeat_plan.to_cache, nfeat_plan.fetch

            nfeat_obj = None

            # --- Pin Memory Logic ---
            if (
                pin_nfeat_buf is not None
                and fetch.numel() <= pin_nfeat_buf.data.shape[0]
            ):
                n = fetch.numel()
                buf_data = pin_nfeat_buf.data[:n]
                buf_nid = pin_nfeat_buf.nid[:n]

                # Gather Data
                torch.index_select(self.shared_nfeat, 0, fetch, out=buf_data)

                # Gather NID
                # buf_nid 是 (Max, 1), fetch 是 (N,)
                buf_nid.copy_(fetch.view(-1, 1).to(dtype=buf_nid.dtype))

                # Async Transfer
                gpu_data = buf_data.to("cuda", non_blocking=True)
                gpu_nid = buf_nid.to("cuda", non_blocking=True).long().squeeze(1)

                nfeat_obj = NFeatNid(nid=gpu_nid, nfeat=gpu_data)
            # ------------------------

            # Fallback
            if nfeat_obj is None:
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
        return None

    def fill_mfg_with_efeat_n_nfeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        nfeat: NFeatNid | None,
        pin_efeat_buf: torch.Tensor | None = None,  # 记得在接口这里也透传 buf
    ) -> list[list[DGLBlock]]:
        # 这一部分是负责nfeat重用的,不管是mfg版本还是no_mfg版本,只要有nfeat都要做

        if self.shared_nfeat is not None:
            assert nfeat is not None
            gnn_mfgs[0] = self.fill_block_with_nfeat(gnn_mfgs[0], nfeat)
        if self.shared_efeat is not None:
            gnn_mfgs = self.fill_block_with_efeat(
                gnn_mfgs, self.shared_efeat, pin_efeat_buf=pin_efeat_buf
            )

        return gnn_mfgs
