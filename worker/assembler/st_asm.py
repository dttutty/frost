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
        logger.debug(
            f"StAsm: Initializing StAssembler, shared_nfeat={'enabled' if shared_nfeat is not None else 'disabled'}, shared_efeat={'enabled' if shared_efeat is not None else 'disabled'}"
        )
        self.nfeat_delta = None
        self.device = device
        self.shared_nfeat = shared_nfeat
        self.shared_efeat = shared_efeat
        if shared_nfeat is not None:
            logger.debug(
                f"StAsm: shared_nfeat shape={shared_nfeat.shape}, dtype={shared_nfeat.dtype}"
            )
        if shared_efeat is not None:
            logger.debug(
                f"StAsm: shared_efeat shape={shared_efeat.shape}, dtype={shared_efeat.dtype}"
            )
        super().__init__()
        logger.debug("StAsm: StAssembler initialized")

    def fill_block_with_efeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        efeat: torch.Tensor,
        eids: list[list[torch.Tensor]],
        pin_efeat_buf: torch.Tensor | None = None,  # 新增参数
    ) -> list[list[DGLBlock]]:
        logger.debug(
            f"StAsm: fill_block_with_efeat called, num_mfg_groups={len(gnn_mfgs)}, pin_efeat_buf={'enabled' if pin_efeat_buf is not None else 'disabled'}"
        )

        # --- 优化路径：使用 Pinned Memory ---
        if pin_efeat_buf is not None:
            logger.debug("StAsm: Using pinned memory path for efeat")
            # 1. 收集所有 Block 的 EID (Flatten)
            # 我们需要知道每个 block 有多少边，以便后续切分
            all_eids = []
            block_sizes = []

            for group_idx, blocks in enumerate(eids if eids is not None else gnn_mfgs):
                for block_idx, eid in enumerate(blocks):
                    all_eids.append(eid)
                    block_sizes.append(eid.numel())
                    logger.debug(
                        f"StAsm: Group {group_idx}, Block {block_idx}, num_edges={eid.numel()}"
                    )

            if not all_eids:
                logger.debug("StAsm: No edges found, returning early")
                return gnn_mfgs

            # 拼接成一个大 Tensor (CPU)
            flat_eids = torch.cat(all_eids)
            total_edges = flat_eids.numel()
            logger.debug(
                f"StAsm: Total edges to gather={total_edges}, buffer capacity={pin_efeat_buf.shape[0]}"
            )

            # 2. 检查 Buffer 容量是否足够
            if total_edges <= pin_efeat_buf.shape[0]:
                logger.debug("StAsm: Buffer sufficient, using pinned path")
                # 截取 Buffer 视图
                buf_view = pin_efeat_buf[:total_edges]

                # Gather: CPU -> Pinned CPU
                # 这是一个大块的内存拷贝，比循环里每次拷一点要快

                if __debug__:
                    # 1. 检查索引张量是否为空
                    assert flat_eids.numel() > 0, (
                        "flat_eids is empty, check sampler output"
                    )

                    # 2. 核心边界检查：确保所有边 ID 都在 efeat 范围内
                    # efeat.size(0) 通常是数据集的总边数
                    max_eid = flat_eids.max().item()
                    min_eid = flat_eids.min().item()
                    assert max_eid < efeat.size(0), (
                        f"Edge ID {max_eid} out of bounds for efeat size {efeat.size(0)}"
                    )
                    assert min_eid >= 0, f"Negative Edge ID {min_eid} detected"

                    # 3. 缓冲区形状校验
                    # 确保 buf_view 的空间足够容纳这些边特征
                    assert buf_view.size(0) == flat_eids.numel(), (
                        f"Buffer mismatch: buf_view={buf_view.size(0)}, flat_eids={flat_eids.numel()}"
                    )
                torch.index_select(efeat, 0, flat_eids, out=buf_view)
                logger.debug("StAsm: Gathered efeat to pinned buffer")

                # Async Copy: Pinned CPU -> GPU
                gpu_all_efeat = buf_view.to(self.device, non_blocking=False)
                logger.debug("StAsm: Async copying efeat to GPU")

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
                logger.debug(f"StAsm: Assigned efeat to {idx} blocks via pinned path")

                return gnn_mfgs
            else:
                logger.debug(
                    f"StAsm: Buffer insufficient (need {total_edges}, have {pin_efeat_buf.shape[0]}), falling back to regular path"
                )

        # --- Fallback: 原始 CPU 路径 ---
        # 如果没有 Buffer 或者 Buffer 不够大
        logger.debug("StAsm: Using regular (blocking) path for efeat")
        block_count = 0
        for blocks in gnn_mfgs:
            for block in blocks:
                global_eid: torch.Tensor = block.edata["ID"]
                # 这里产生的是 CPU Tensor
                block.edata["efeat"] = efeat[global_eid]
                block_count += 1
        logger.debug(f"StAsm: Assigned efeat to {block_count} blocks via regular path")

        return gnn_mfgs

    def fill_block_with_nfeat(
        self, gnn_mfgs_0: list[DGLBlock], nfeat: NFeatNid
    ) -> list[DGLBlock]:
        logger.debug(
            f"StAsm: fill_block_with_nfeat called, num_blocks={len(gnn_mfgs_0)}, nfeat size={nfeat.nid.numel()}"
        )
        assert nfeat.is_sorted  # make sure nfeat.nid is sorted

        for idx, block in enumerate(gnn_mfgs_0):
            global_nid: torch.Tensor = block.srcdata["ID"]

            permute_nidx = torch.searchsorted(nfeat.nid, global_nid)
            block.srcdata["nfeat"] = nfeat.nfeat[permute_nidx]  # 这里放memory
            logger.debug(
                f"StAsm: Block {idx} filled with nfeat, num_nodes={global_nid.numel()}"
            )

        logger.debug("StAsm: All blocks filled with nfeat")
        return gnn_mfgs_0

    def prepare_nfeat_cache(
        self,
        nfeat_plan: StPlan,
        pin_nfeat_buf: StBuf | None = None,  # 接收 StBuf
    ) -> NFeatNid | None:
        logger.debug(
            f"StAsm: prepare_nfeat_cache called, pin_nfeat_buf={'enabled' if pin_nfeat_buf is not None else 'disabled'}"
        )
        assert self.shared_nfeat is not None
        to_cache, fetch = nfeat_plan.to_cache.data, nfeat_plan.fetch.data
        logger.debug(
            f"StAsm: to_cache size={to_cache.numel()}, fetch size={fetch.numel()}"
        )

        nfeat_obj = None

        # --- Pin Memory Logic ---
        if pin_nfeat_buf is not None and fetch.numel() <= pin_nfeat_buf.data.shape[0]:
            logger.debug(
                f"StAsm: Using pinned buffer for nfeat, fetch={fetch.numel()}, buffer capacity={pin_nfeat_buf.data.shape[0]}"
            )
            n = fetch.numel()
            buf_data = pin_nfeat_buf.data[:n]
            buf_nid = pin_nfeat_buf.nid[:n]

            # Gather Data
            if __debug__:
                # 1. 确保 fetch 索引张量不为空
                assert fetch.numel() > 0, "Fetch indices tensor is empty"

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
            logger.debug("StAsm: Gathered nfeat data to pinned buffer")

            # Gather NID
            # buf_nid 是 (Max, 1), fetch 是 (N,)
            buf_nid.copy_(fetch.to(dtype=buf_nid.dtype))

            # Async Transfer
            gpu_data = buf_data.to(self.device, non_blocking=False)
            gpu_nid = buf_nid.to(self.device, non_blocking=False).long()
            logger.debug("StAsm: Async copying nfeat to GPU")

            nfeat_obj = NFeatNid(nid=gpu_nid, nfeat=gpu_data)
        elif pin_nfeat_buf is not None:
            logger.debug(
                f"StAsm: Buffer insufficient for nfeat (need {fetch.numel()}, have {pin_nfeat_buf.data.shape[0]}), falling back"
            )
        # ------------------------

        # Fallback
        if nfeat_obj is None:
            logger.debug("StAsm: Using regular (blocking) path for nfeat")
            nfeat_obj = NFeatNid(nid=fetch, nfeat=self.shared_nfeat[fetch])
            nfeat_obj = nfeat_obj.cuda()
            logger.debug("StAsm: nfeat gathered and moved to GPU via regular path")
            # 注意: 普通路径下数据在 CPU，后续如果需要 GPU 数据，
            # fill_mfg_with_efeat_n_nfeat 或 H2D 逻辑可能需要处理 .to(cuda)
            # 为了一致性，建议此处不上 GPU，由 H2D 统一调度，或者在这里统一上 GPU。
            # 假设 H2D 现在的逻辑是可以处理 GPU Tensor 的。

        if self.nfeat_delta is not None:
            logger.debug(
                f"StAsm: Merging with nfeat_delta, delta size={self.nfeat_delta.nid.numel()}"
            )

            nfeat_obj = nfeat_obj.merge(self.nfeat_delta)
            logger.debug(f"StAsm: After merge, nfeat size={nfeat_obj.nid.numel()}")

        logger.debug("StAsm: Sorting nfeat by nid")
        nfeat_obj = nfeat_obj.sort_by_nid()

        if to_cache.numel() > 0:
            logger.debug(f"StAsm: Caching {to_cache.numel()} nfeat entries")
            self.nfeat_delta = nfeat_obj.filter_by_nid(to_cache)
            logger.debug(
                f"StAsm: nfeat_delta updated, size={self.nfeat_delta.nid.numel()}"
            )
        else:
            logger.debug("StAsm: No caching needed, clearing nfeat_delta")
            self.nfeat_delta = None

        logger.debug("StAsm: prepare_nfeat_cache complete")
        return nfeat_obj

    def fill_mfg_with_nfeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        nfeat: NFeatNid | None,
    ) -> list[list[DGLBlock]]:
        logger.debug(
            f"StAsm: fill_mfg_with_nfeat called, num_mfg_groups={len(gnn_mfgs)}, nfeat is {'provided' if nfeat is not None else 'None'}"
        )
        # 这一部分是负责nfeat重用的,不管是mfg版本还是no_mfg版本,只要有nfeat都要做

        if self.shared_nfeat is not None:
            logger.debug("StAsm: Filling blocks with nfeat")
            assert nfeat is not None
            gnn_mfgs[0] = self.fill_block_with_nfeat(gnn_mfgs[0], nfeat)
        else:
            logger.debug("StAsm: shared_nfeat is None, skipping nfeat fill")

        logger.debug("StAsm: fill_mfg_with_nfeat complete")
        return gnn_mfgs

    def fill_mfg_with_efeat(
        self,
        gnn_mfgs: list[list[DGLBlock]],
        eids: list[list[torch.Tensor]],
        pin_efeat_buf: torch.Tensor | None = None,  # 记得在接口这里也透传 buf
    ) -> list[list[DGLBlock]]:
        logger.debug(
            f"StAsm: fill_mfg_with_efeat called, num_mfg_groups={len(gnn_mfgs)}"
        )
        # 这一部分是负责nfeat重用的,不管是mfg版本还是no_mfg版本,只要有nfeat都要做

        if self.shared_efeat is not None:
            logger.debug("StAsm: Filling blocks with efeat")
            gnn_mfgs = self.fill_block_with_efeat(
                gnn_mfgs, self.shared_efeat, pin_efeat_buf=pin_efeat_buf, eids=eids
            )
        else:
            logger.debug("StAsm: shared_efeat is None, skipping efeat fill")

        logger.debug("StAsm: fill_mfg_with_efeat complete")
        return gnn_mfgs
