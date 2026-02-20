import logging
import threading
from collections import deque
from queue import Queue as Q
from typing import Literal

import numpy
import torch
import torch.distributed as dist

from schemas.alias import MetaBundle
from schemas.datatypes.env import Env
from schemas.datatypes.misc import (
    DGLBlockMeta,  # 确保导入 DGLBlockMeta
    Meta,
    Result,
)
from schemas.datatypes.plan import DyPlan, PlanSet, StPlan
from schemas.shm_layout import MANIFEST_SIZE, ShmKeys  # 导入第一步定义的
from shm_manager import DoubleBufferedSHM
from thread_base import ThreadBase


logger = logging.getLogger(__name__)


class Scatter(ThreadBase):
    def __init__(
        self,
        n_gpu: int,
        in_q: Q[MetaBundle | None | Literal["end_of_epoch"]],
        mem_plan_out_q: Q[list[DyPlan] | None] | None,
        mail_plan_out_q: Q[list[DyPlan] | None] | None,
        nfeat_plan_out_q: Q[list[StPlan] | None] | None,
        env: Env,
        result_q: Q[Result],
        group: dist.ProcessGroup,
        stop_ev: threading.Event,
        shm_managers: list[DoubleBufferedSHM],
    ) -> None:
        super().__init__(name="scatter", stop_ev=stop_ev)
        self.n_gpu = n_gpu
        self.in_q = in_q
        self.mem_plan_out_q = mem_plan_out_q
        self.mail_plan_out_q = mail_plan_out_q
        self.nfeat_plan_out_q = nfeat_plan_out_q
        self.stop_ev = stop_ev
        self.result_q = result_q
        self.shm_managers = shm_managers

        self.world_size: int = env.world_size
        self.local_rank: int = env.local_rank

        self.inflight = deque()
        self.inflight_K = 2
        # 2. 将所有全局变量变为实例属性
        self.host_rank: int = env.host_rank
        self.ngpus: int = env.n_gpu
        self.gloo: dist.ProcessGroup = group
        self.dummy_recv = torch.zeros(MANIFEST_SIZE, dtype=torch.long)
        self.control_tensor_template = torch.zeros(MANIFEST_SIZE, dtype=torch.long)
        self._gather_seq = 0
        self._ctrl_send_seq = 0

        assert self.local_rank == self.host_rank

    def _offload_to_shm_v2(
        self, mgr: DoubleBufferedSHM, batch_idx: int, meta: Meta
    ) -> torch.Tensor:
        """
        将 Meta 所有数据写入 SHM，并返回一个 Manifest Tensor (LongTensor)。
        Tensor[i] 存储 ShmKeys(i) 对应数据的长度 (字节或元素数)。
        """
        # 初始化清单 Tensor

        if meta.root_nid is None:
            logger.error(
                f"!!! [CRITICAL] Batch {batch_idx}: meta.root_nid is None BEFORE writing to SHM!"
            )
            raise ValueError("Sampler produced None root_nid!")

        # --- [调试] 打印关键数据状态 ---
        # if meta.root_nid is None:
        #     logger.error(f">>> [SCATTER CHECK] Batch {batch_idx}: root_nid is NONE!")
        # else:
        #     # 正常情况下应该看到这行日志
        #     logger.info(
        #         f">>> [SCATTER CHECK] Batch {batch_idx}: root_nid shape={meta.root_nid.shape}, pos_edge_id shape={meta.pos_edge_id.shape if meta.pos_edge_id is not None else 'None'}"
        #     )
        # # ----------------------------

        # if meta.pos_edge_id is None:
        #     logger.error(
        #         f"[Scatter ERROR] Batch {batch_idx}: meta.pos_edge_id is None!"
        #     )
        # elif meta.pos_edge_id.numel() == 0:
        #     logger.error(
        #         f"[Scatter ERROR] Batch {batch_idx}: meta.pos_edge_id is Empty Tensor!"
        #     )
        # else:
        #     # 正常情况
        #     pass
        manifest = torch.zeros(MANIFEST_SIZE, dtype=torch.long)

        # 写入 Batch ID
        manifest[ShmKeys.BATCH_IDX] = batch_idx

        def write(key_idx: int, tensor: torch.Tensor | None) -> None:
            if tensor is None:
                return
            # [关键修复] 即使是空 tensor (numel=0) 也要写入!
            # 否则 Worker 读取时会得到 None，导致结构缺失。
            # SHM write 内部通常处理了 length=0 的情况 (返回0)，但 key 必须存在

            try:
                # [关键修复] 强制内存连续，防止 Numpy 切片导致的 stride 问题
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()

                str_key = str(int(key_idx))
                l = mgr.write(batch_idx, str_key, tensor)
                manifest[key_idx] = l
            except Exception as e:
                logger.error(
                    f"[Scatter Write Error] Batch {batch_idx} Key {key_idx}: {e}"
                )
                raise e

        def pack_cts(np_array: None | numpy.ndarray) -> torch.Tensor | None:
            if np_array is None:
                return None
            # [核心修复] 强制转换为 int64 (long) 并且 contiguous
            # 这一步解决了绝大多数 SHM 读写不一致/为0的问题
            return torch.as_tensor(np_array, dtype=torch.long).contiguous()

        # --- 新增: 写入 m_state ---
        if meta.m_state is not None:
            ms = meta.m_state
            manifest[ShmKeys.MS_CUR_EPOCH] = ms.cur_epoch
            manifest[ShmKeys.MS_N_EPOCH] = ms.n_epoch
            manifest[ShmKeys.MS_CUR_MACRO_SPLIT] = ms.cur_macro_of_split
            manifest[ShmKeys.MS_N_MACRO_SPLIT] = ms.n_macro_of_split
            manifest[ShmKeys.MS_CUR_MACRO_EPOCH] = ms.cur_macro_of_epoch

            # 将字符串 split 转换为 int 编码
            split_map = {"train": 0, "val": 1, "test": 2}
            manifest[ShmKeys.MS_SPLIT_TYPE] = split_map.get(ms.split, 0)

        # 1. Basic
        write(ShmKeys.ROOT_NID, meta.root_nid)
        write(ShmKeys.ROOT_TS, meta.root_ts)
        write(ShmKeys.POS_EDGE_ID, meta.pos_edge_id)

        # 2. N2R Packed
        if meta.n2r_meta_packed:
            p = meta.n2r_meta_packed
            write(ShmKeys.N2R_NODES, p.nodes_cat)
            write(ShmKeys.N2R_TS, p.ts_cat)
            write(ShmKeys.N2R_EID, p.eid_cat)
            write(ShmKeys.N2R_DTS, p.dts_cat)
            write(ShmKeys.N2R_COL, p.col_cat)
            write(ShmKeys.N2R_ROW, p.row_cat)

            # 【优化点】元数据也进 SHM
            t_nsec, t_esec, t_dims = p.pack_metadata_to_tensor()
            write(ShmKeys.N2R_NODE_SECTIONS, t_nsec)
            write(ShmKeys.N2R_EDGE_SECTIONS, t_esec)
            write(ShmKeys.N2R_DIMS, t_dims)

        # --- 3. R2N Packed (Temporal Neighbors) ---
        if meta.r2n_meta_packed:
            p = meta.r2n_meta_packed
            write(ShmKeys.R2N_NODES, p.nodes_cat)
            write(ShmKeys.R2N_TS, p.ts_cat)
            write(ShmKeys.R2N_EID, p.eid_cat)
            write(ShmKeys.R2N_DTS, p.dts_cat)
            write(ShmKeys.R2N_COL, p.col_cat)
            write(ShmKeys.R2N_ROW, p.row_cat)

            t_nsec, t_esec, t_dims = p.pack_metadata_to_tensor()
            write(ShmKeys.R2N_NODE_SECTIONS, t_nsec)
            write(ShmKeys.R2N_EDGE_SECTIONS, t_esec)
            write(ShmKeys.R2N_DIMS, t_dims)

            if meta.r2n_mdt is not None:
                write(ShmKeys.R2N_MDT, meta.r2n_mdt)

        # --- 4. PlanSet ---
        if meta.packed_plan_set:
            ps = meta.packed_plan_set

            # A. Memory Plan
            if ps.mem:
                write(ShmKeys.MEM_P2P, ps.mem.p2p_data)
                # p2p_sizes 也要转
                write(
                    ShmKeys.MEM_P2P_SIZES,
                    torch.tensor(ps.mem.p2p_sizes, dtype=torch.long),
                )
                write(ShmKeys.MEM_FETCH, ps.mem.fetch_data)
                write(ShmKeys.MEM_WB, ps.mem.writeback_data)

                # [应用修复] 使用 pack_cts
                write(ShmKeys.MEM_REC_CTS, pack_cts(ps.mem.rec_cts))
                write(ShmKeys.MEM_SEND_CTS, pack_cts(ps.mem.send_cts))

            # B. Box Plan
            if ps.box:
                write(ShmKeys.BOX_P2P, ps.box.p2p_data)
                write(
                    ShmKeys.BOX_P2P_SIZES,
                    torch.tensor(ps.box.p2p_sizes, dtype=torch.long),
                )
                write(ShmKeys.BOX_FETCH, ps.box.fetch_data)
                write(ShmKeys.BOX_WB, ps.box.writeback_data)

                # [应用修复] 使用 pack_cts
                write(ShmKeys.BOX_REC_CTS, pack_cts(ps.box.rec_cts))
                write(ShmKeys.BOX_SEND_CTS, pack_cts(ps.box.send_cts))

            # C. NFeat Plan
            if ps.nfeat:
                write(ShmKeys.NFEAT_CACHE, ps.nfeat.to_cache_data)
                write(ShmKeys.NFEAT_FETCH, ps.nfeat.fetch_data)

        return manifest

    def _get_control_signal_list(self, signal_code: int) -> list[torch.Tensor]:
        """生成发送给所有 Worker 的控制信号列表"""
        # 1. 创建一个 filled tensor
        t = self.control_tensor_template.clone()
        t[ShmKeys.BATCH_IDX] = signal_code

        # 2. 复制 world_size 份 (因为 dist.scatter 要求 list 长度必须匹配)
        return [t.clone() for _ in range(self.world_size)]

    def scatter_send(self, data: list[Meta] | None | Literal["end_of_epoch"]) -> None:
        assert self.local_rank == self.host_rank, "只有 Host 进程可以 scatter"
        scatter_list: list[Meta | None | Literal["end_of_epoch"]]

        if data == "end_of_epoch":
            scatter_list = ["end_of_epoch"] * self.world_size
        elif isinstance(data, list):
            assert len(data) == self.ngpus, (
                f"需要为 {self.ngpus} 个 GPU worker 准备 payload, 但你给了 {len(data)} 个"
            )
            scatter_list = [*data, None]  # Host 自己的槽位留 None
        else:
            scatter_list = [None] * self.world_size

        dist.scatter_object_list(
            [None],  # Host 自己的接收缓冲区（没用）
            scatter_list,
            src=self.host_rank,
            group=self.gloo,
        )

        # scatter.py

    def gather_receive(self) -> Result:
        assert self.local_rank == self.host_rank, "只有 Host 进程可以 gather"

        gathered_list: list[Result | None] = [None] * self.world_size

        # Host 也必须参与，并提供一个"占位"的发送值
        self._gather_seq += 1
        logger.info(
            "Scatter gather_object start: seq=%d rank=%d",
            self._gather_seq,
            self.local_rank,
        )
        dist.gather_object(
            None,
            gathered_list,
            dst=self.host_rank,  # Host 发送的占位数据
            group=self.gloo,
        )
        logger.info(
            "Scatter gather_object end: seq=%d rank=%d",
            self._gather_seq,
            self.local_rank,
        )

        data_0_result = gathered_list[0]
        assert data_0_result is not None, "Worker 返回的 Result 不能为 None"
        # 丢掉 Host 自己的槽位（最后一个），只返回 workers 的结果
        return data_0_result

    def flush_inflight(self):
        while self.inflight:
            works, tensors = self.inflight.popleft()
            for w in works:
                w.wait()

    def logic(self) -> None:
        batch_count = 0
        while not self.stop_ev.is_set():
            meta_2_scatter = self.in_q.get(block=True)

            if meta_2_scatter is None:
                self.flush_inflight()
                send_tensors = self._get_control_signal_list(-2)

                self._ctrl_send_seq += 1
                logger.info(
                    "Scatter ctrl send: seq=%d code=-2 rank=%d",
                    self._ctrl_send_seq,
                    self.local_rank,
                )
                works = []
                for i in range(self.n_gpu):
                    works.append(dist.isend(send_tensors[i], dst=i, group=self.gloo))
                for w in works:
                    w.wait()
                break
            elif meta_2_scatter == "end_of_epoch":
                # 【修正 1】发送 Tensor 列表
                self.flush_inflight()
                send_tensors = self._get_control_signal_list(-1)

                self._ctrl_send_seq += 1
                logger.info(
                    "Scatter ctrl send: seq=%d code=-1 rank=%d",
                    self._ctrl_send_seq,
                    self.local_rank,
                )

                works = []
                for i in range(self.n_gpu):
                    works.append(dist.isend(send_tensors[i], dst=i, group=self.gloo))
                for w in works:
                    w.wait()
                # 等待收集结果
                data = self.gather_receive()
                self.result_q.put(data)
                continue

            else:
                batch_count += 1
                assert len(meta_2_scatter) == self.n_gpu

                mem_plans = mail_plans = nfeat_plans = None

                if self.mem_plan_out_q is not None:
                    mem_plans = self.mem_plan_out_q.get(block=True)

                if self.mail_plan_out_q is not None:
                    mail_plans = self.mail_plan_out_q.get(block=True)

                if self.nfeat_plan_out_q is not None:
                    nfeat_plans = self.nfeat_plan_out_q.get(block=True)

                # 3. 【关键修改】组装 Plan 并执行打包操作
                send_tensors = []
                for i in range(self.n_gpu):
                    worker_rank = i
                    mgr = self.shm_managers[worker_rank]
                    meta = meta_2_scatter[i]

                    # --- A. 处理 PlanSet ---
                    # 先构建原始 PlanSet
                    assert nfeat_plans is not None, (
                        "nfeat_plans 不能为空，因为 StAssembler 需要它来填充 nfeat_plan"
                    )
                    raw_plan_set = PlanSet(
                        mem=mem_plans[i] if mem_plans else None,
                        box=mail_plans[i] if mail_plans else None,
                        nfeat=nfeat_plans[i] if nfeat_plans else None,
                    )

                    # 执行 Pack：将 list[TensorSet] 压扁为大 Tensor
                    # 注意：假设你在 PlanSet 中实现了 pack() 方法 (参考上一轮对话)
                    meta.packed_plan_set = raw_plan_set.pack()

                    # 【重要】设为 None，确保不会传输原始 heavy 对象
                    meta.plan_set = None

                    # --- B. 处理 n2r_meta (List[DGLBlockMeta]) ---
                    if meta.n2r_meta is not None:
                        # 使用静态方法将列表打包成 Structure of Arrays
                        meta.n2r_meta_packed = DGLBlockMeta.pack_list(meta.n2r_meta)
                        # 【重要】清空原始列表
                        meta.n2r_meta = None

                    # --- C. 处理 r2n_meta (Single DGLBlockMeta) ---
                    if meta.r2n_meta is not None:
                        # 即使是单个对象，为了复用 pack_list 的逻辑，包装成列表传入
                        # 这样接收端 unpack 出来也是列表，取 [0] 即可，或者你如果不嫌麻烦可以写个 pack_single
                        # 这里建议复用 pack_list，解包时注意一下即可
                        meta.r2n_meta_packed = DGLBlockMeta.pack_list([meta.r2n_meta])
                        # 【重要】清空原始对象
                        meta.r2n_meta = None

                    manifest = self._offload_to_shm_v2(
                        self.shm_managers[i], batch_count, meta
                    )
                    send_tensors.append(manifest)
                    # ... 写入其他 ...

                    # meta.batch_idx = self.  # 让 Worker 知道这是第几轮

                    # --- D. 其他清理 ---
                    meta.state = None  # 清理 state，节省传输开销

                # dummy_recv = torch.zeros(MANIFEST_SIZE, dtype=torch.long)
                # dist.scatter(
                #     dummy_recv,
                #     scatter_list=send_tensors,
                #     src=self.host_rank,
                #     group=self.group,
                # )
                works = []
                for i in range(self.n_gpu):
                    w = dist.isend(send_tensors[i], dst=i, group=self.gloo)
                    works.append(w)

                # 保存引用：确保 send_tensors 在发送完成前不被释放
                self.inflight.append((works, send_tensors))

                # 流控：最多允许 K 轮在飞
                if len(self.inflight) > self.inflight_K:
                    old_works, old_tensors = self.inflight.popleft()
                    for w in old_works:
                        w.wait()
