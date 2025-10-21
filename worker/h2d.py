import queue
import threading
from typing import Optional

import torch
import torch.distributed as dist

from memorys import MemoryCFG, MemoryMailbox
from schemas.datatypes import GPUData, PrevCache, TransferData
from utils import mfgs_to_cuda, prepare_mfgs
from worker.gate import GlobalMicrobatchGate


class h2d(threading.Thread):
    def __init__(
        self,
        dedup,
        host_rank,
        stm: torch.cuda.Stream,
        nbuf,
        ebuf,
        nfeat,
        efeat,
        mm: MemoryMailbox | None,
        mcfg: MemoryCFG,
        merge_src,
        max_inflight: int = 2,  # 目前未用；可扩展成多缓冲
        gate: Optional[GlobalMicrobatchGate] = None,
        comm_lock: threading.Lock = threading.Lock(),  # 用于保护 dist 通信
    ):
        super().__init__(daemon=True)
        self.stop_ev = threading.Event()
        self.stream = stm
        self.host_rank = host_rank
        self.dedup = dedup
        self.pinned_nfeat_buffs = nbuf
        self.pinned_efeat_buffs = ebuf
        self.node_feats = nfeat
        self.edge_feats = efeat
        self.mm = mm
        self.memory_param = mcfg
        self.deduplicate_mfg_src = merge_src
        self.gate: Optional[GlobalMicrobatchGate] = gate
        self.deliver_to = mcfg.deliver_to
        self.comm_lock = comm_lock

        # 复用缓存（上一批 uniqn/uniqe 及其 GPU 切片）
        self.prev_cache = PrevCache(None, None, None, None)

        # 接口队列
        self.submit_q = queue.Queue(maxsize=0)
        self.ready_q = queue.Queue(maxsize=1)

    def submit(self):
        """训练侧调用：请求预取下一批（由 mover 内部收主机的 TransferData 并做 H2D）"""
        self.submit_q.put(object(), block=True)

    def get_ready(self, block=True, timeout=None):
        """训练侧调用：拿到 (GPUData, done_ev, td)"""
        return self.ready_q.get(block=block, timeout=timeout)

    def _recv_transfer_data(self) -> Optional[TransferData]:
        def scatter_receive(host_rank):
            """
            Worker 端用 scatter_object_list 接收 host 分发的对象。
            注意：worker 只传 obj_list（长度=1），scatter_list 必须由 src 端提供。
            """
            recv = [None]
            dist.scatter_object_list(recv, None, src=host_rank)
            return recv[0]

        # 简单阻塞式接收即可；如需更高吞吐，可以改造为 irecv + 双缓冲
        td: Optional[TransferData] = scatter_receive(self.host_rank)
        return td

    def run(self):
        # 当前进程已将 CUDA_VISIBLE_DEVICES 收窄为 1，这里直接用当前 device
        torch.cuda.set_device(torch.cuda.current_device())

        while not self.stop_ev.is_set():
            # 等训练侧提交
            try:
                self.submit_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # 1) 从 host 接 TransferData（CPU 上）
            with self.comm_lock:
                td: Optional[TransferData] = self._recv_transfer_data()

            if td is None or (td.mfgss is None) or (len(td.mfgss) == 0):
                # 通知训练侧“无数据/结束”
                self.ready_q.put((None, None, None), block=True)
                continue

            mfgss_on_cpu = td.mfgss  # List[List[DGLBlock]] 分组后的 MFGs
            # 如果需要，可用 get_ids 做一些统计/调试
            # nids, eids = get_ids(mfgss_on_cpu, self.node_feats, self.edge_feats)

            # 2) 搬到 GPU + 准备特征（在独立的拷贝 stream 上）
            with torch.cuda.stream(self.stream), torch.no_grad():
                gpumfgss = mfgs_to_cuda(mfgss_on_cpu)

                # 使用（或不使用）去重缓存准备特征
                gpumfgss, gpunfeat, gpuefeat, uniqn, uniqe = prepare_mfgs(
                    self.dedup,
                    mfgss_on_cpu,
                    gpumfgss,
                    self.node_feats,
                    self.edge_feats,
                    deduplicate_mfg_src=self.deduplicate_mfg_src,
                    pinned=True,
                    nfeat_buffs=self.pinned_nfeat_buffs,
                    efeat_buffs=self.pinned_efeat_buffs,
                    prev_cache=self.prev_cache,
                )

                gpumem = gpumail = None
                if (
                    (self.mm is not None)
                    and (self.gate is not None)
                    and self.gate.wait_can_start(td.mid)
                ):
                    # 只有 gate 放行时，才准备 mailbox 的 GPU 侧片段
                    gpumem, gpumail = self.mm.prepmm(
                        self.dedup,
                        gpumfgss[0],  # 第一层/最近一跳的 block（按你的实现需要）
                        uniqn,
                        use_buf=True,
                    )

                # 记录一个事件，用于训练流等待
                done_ev = torch.cuda.Event(blocking=False)
                done_ev.record(self.stream)

            # 3) 更新复用缓存（给下一批使用）
            self.prev_cache = PrevCache(
                gpunfeat=gpunfeat,
                gpuefeat=gpuefeat,
                uniqn=uniqn,
                uniqe=uniqe,
            )

            # 4) 交付训练侧
            gpu_data = GPUData(
                mfgs=gpumfgss,
                nfeat=gpunfeat,
                efeat=gpuefeat,
                gpumem=gpumem,
                gpumail=gpumail,
            )
            self.ready_q.put((gpu_data, done_ev, td), block=True)

    def stop(self):
        self.stop_ev.set()
