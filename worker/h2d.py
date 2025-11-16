import queue
import threading
from typing import Optional

import torch

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
        rx_q: queue.Queue,
        ready_q: queue.Queue,
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
        self.rx_q = rx_q

        # 复用缓存（上一批 uniqn/uniqe 及其 GPU 切片）
        self.prev_cache = PrevCache(None, None, None, None)

        # 接口队列
        self.ready_q = queue.Queue(maxsize=1)

    def run(self):
        # 当前进程已将 CUDA_VISIBLE_DEVICES 收窄为 1，这里直接用当前 device
        torch.cuda.set_device(torch.cuda.current_device())

        while not self.stop_ev.is_set():
            # 等训练侧提交
            try:
                td: TransferData = self.rx_q.get(timeout=0.1)
            except queue.Empty:
                continue

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

    def prepmm(self, dedup, mfg, node_uniq, use_buf=False):
        mm: MemoryMailbox = self.mm
        # example: memory_mailbox.prep_input_mails(mfgs[0])
        # history = 2
        # mfgs = [[block1,block2],[block3,block4],[block5,block6]]
        # receive mfg = [block1,block2] if 2 history
        data_map = [
            {
                "name": "mem",
                "src": mm.node_memory,
                "pinned": "pinned_node_memory_buffs",
            },
            {
                "name": "mem_ts",
                "src": mm.node_memory_ts,
                "pinned": "pinned_node_memory_ts_buffs",
            },
            {
                "name": "mem_input",
                "src": mm.mails,
                "pinned": "pinned_mailbox_buffs",
            },
            {
                "name": "mail_ts",
                "src": mm.mail_ts,
                "pinned": "pinned_mailbox_ts_buffs",
            },
        ]
        deduplicated_mem = None
        deduplicated_mem_input = None
        for hist_id, hist_block in enumerate(mfg):
            idx = hist_block.srcdata["ID"].cpu().long()
            n_nodes = idx.shape[0]
            for item in data_map:
                name = item["name"]
                if not dedup or name not in ["mem", "mem_input"]:
                    src_tensor = item["src"]
                    pinned_list = getattr(self, item["pinned"], None)
                    can_use_pinned = use_buf and hist_id < len(pinned_list)
                    with Timer() as t:
                        if can_use_pinned:
                            # --- Use Pinned Memory Buffers ---
                            pinned_buffer = pinned_list[hist_id]
                            # Select data into the pinned buffer (CPU operation).
                            # we can see that the length of pinned_buffer[history_i] need to be larger than size of block.srcdata['ID']
                            torch.index_select(
                                src_tensor, 0, idx, out=pinned_buffer[:n_nodes]
                            )
                            # Copy the selected data from pinned buffer to GPU (non-blocking).
                            gpu_data = pinned_buffer[:n_nodes].cuda(non_blocking=True)
                        else:
                            # --- Standard Method (No Pinned Buffers) ---
                            # Select data and directly copy to GPU.
                            gpu_data = src_tensor[idx].cuda()
                            # Reshape mailbox data if needed.
                        if name == "mem_input":
                            gpu_data = gpu_data.reshape(n_nodes, -1)

                        # Assign the prepared data to the DGL block's source nodes.
                        hist_block.srcdata[name] = gpu_data

                    # if name == "mem":
                    # profiler.increase_time({"3_transfer_node_mem": t.last_duration})
                    # elif name == "mem_input":
                    #     profiler.increase_time(
                    #         {"4_transfer_node_mail": t.last_duration}
                    #     )
        if dedup:
            src_tensor = mm.node_memory
            pinned_buffer = getattr(self, "pinned_node_memory_buffs", None)
            with Timer() as t:
                if use_buf:
                    torch.index_select(
                        src_tensor,
                        0,
                        node_uniq,
                        out=pinned_buffer[: node_uniq.shape[0]],
                    )
                    deduplicated_mem = pinned_buffer[: node_uniq.shape[0]].cuda(
                        non_blocking=True
                    )
                else:
                    deduplicated_mem = src_tensor[node_uniq].to(
                        "cuda", non_blocking=True
                    )
            # profiler.increase_time({"3_transfer_node_mem": t.last_duration})

            src_tensor = self.mails
            pinned_buffer = getattr(self, "pinned_mailbox_buffs", None)
            with Timer() as t:
                if use_buf:
                    torch.index_select(
                        src_tensor,
                        0,
                        node_uniq,
                        out=pinned_buffer[: node_uniq.shape[0]],
                    )
                    deduplicated_mem_input = pinned_buffer[: node_uniq.shape[0]].cuda(
                        non_blocking=True
                    )
                else:
                    deduplicated_mem_input = src_tensor[node_uniq].to(
                        "cuda", non_blocking=True
                    )
                deduplicated_mem_input = deduplicated_mem_input.reshape(
                    node_uniq.shape[0], -1
                )
            # profiler.increase_time({"4_transfer_node_mail": t.last_duration})

        return deduplicated_mem, deduplicated_mem_input

    def stop(self):
        self.stop_ev.set()
