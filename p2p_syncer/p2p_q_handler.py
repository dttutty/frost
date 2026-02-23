import logging
import threading
from queue import Queue as Q
from typing import Any, Literal, TypeVar, cast

import torch
import torch.distributed as dist

from buffer import Buffers
from schemas.datatypes.env import Env
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.task import P2PTask
from schemas.protocol.misc import CudaEventLike
from thread_base import ThreadBase


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


class P2PQHandler(ThreadBase):
    def __init__(
        self,
        buf: Buffers,
        env: Env,
        mem_p2p_q: Q[tuple[P2PTask, CudaEventLike] | Literal["nothing_to_p2p"] | None],
        box_p2p_q: Q[tuple[P2PTask, CudaEventLike] | Literal["nothing_to_p2p"]],
        nccl_p2p: dist.ProcessGroup,
        p2p_sem: threading.Semaphore,
        stop_ev: threading.Event,
        mem_writer_q: Q[tuple[torch.Tensor | None, MemTsNid, CudaEventLike]],
        delta_q: Q[tuple[P2PTask, P2PTask, list[Any]] | Literal["nothing"]],
        device: torch.device,
    ) -> None:
        super().__init__(name="p2p_q_handler", stop_ev=stop_ev)
        self.buf = buf
        self.device = device
        self.mem_p2p_q = mem_p2p_q
        self.box_p2p_q = box_p2p_q
        self.p2p_sem = p2p_sem
        self.env = env
        self.stop_ev = stop_ev
        self.mem_writer_q = mem_writer_q
        self.delta_q = delta_q
        self.nccl_p2p = nccl_p2p
        self.handler_stream = cast(
            torch.cuda.Stream, torch.cuda.Stream(device=self.device, priority=-1)
        )
        self.count = 0

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            with torch.cuda.stream(self.handler_stream):
                # CPU 侧拿任务、计算时间
                mem_p2p_tuple = self.mem_p2p_q.get(block=True)
                if mem_p2p_tuple is None:
                    # logger.debug(
                    #     f"[Rank {self.env.local_rank}] Received shutdown signal in P2P handler."
                    # )
                    break
                # logger.debug(f"[Rank {self.env.local_rank}] Got Mem P2P task: {mem_p2p_tuple}")
                mem_wait_ms = 0.0

                # 快速路径：如果没有 P2P 任务，直接跳过
                if mem_p2p_tuple == "nothing_to_p2p":
                    box_p2p_tuple = self.box_p2p_q.get(block=True)
                    assert box_p2p_tuple == "nothing_to_p2p"
                    self.delta_q.put("nothing")
                    # Keep semaphore balanced so the next step doesn't block after no-op P2P.
                    # logger.debug(
                    #     "nothing_to_p2p received for both Mem and Box. Skipping P2P and sending 'nothing' to delta_q."
                    # )
                    self.p2p_sem.release()

                    # if mem_wait_ms > 5.0:
                    #     logger.debug(
                    #         f"[Rank {self.env.local_rank}] P2P Starvation (Nothing): Waited {mem_wait_ms:.2f}ms for empty signal"
                    # )

                    continue
                mem_p2p, mem_event = mem_p2p_tuple
                # 此时肯定有 box 任务
                box_p2p_tuple = self.box_p2p_q.get(block=True)
                box_wait_ms = 0.0
                total_wait = mem_wait_ms + box_wait_ms
                # if total_wait > 2.0:
                #     logger.debug(
                #         f"[Rank {self.env.local_rank}] P2P Blocked! Total: {total_wait:.2f}ms | Mem: {mem_wait_ms:.2f}ms | Box: {box_wait_ms:.2f}ms"
                # )
                assert box_p2p_tuple != "nothing_to_p2p"
                box_p2p, box_event = box_p2p_tuple
                torch.cuda.current_stream().wait_event(mem_event)
                torch.cuda.current_stream().wait_event(box_event)

                # 2. 准备 NCCL 通讯列表 (Flatten the Loop)
                # 顺序：Mem_NID -> Mem_Data -> Mem_TS -> Box_NID -> Box_Data -> Box_TS
                # 这样写的好处是 NCCL Kernel 会被密集地压入 CUDA Stream，没有任何 Python 逻辑打断
                p2p_ops = [
                    # 格式：(Name, Recv, Send, Rec_Cts, Send_Cts)
                    (
                        "Mem_NID",
                        mem_p2p.nid2recv,
                        mem_p2p.nid2send,
                        mem_p2p.rec_cts,
                        mem_p2p.send_cts,
                    ),
                    (
                        "Mem_Data",
                        mem_p2p.data2recv,
                        mem_p2p.data2send,
                        mem_p2p.rec_cts,
                        mem_p2p.send_cts,
                    ),
                    (
                        "Mem_TS",
                        mem_p2p.ts2recv,
                        mem_p2p.ts2send,
                        mem_p2p.rec_cts,
                        mem_p2p.send_cts,
                    ),
                    (
                        "Box_NID",
                        box_p2p.nid2recv,
                        box_p2p.nid2send,
                        box_p2p.rec_cts,
                        box_p2p.send_cts,
                    ),
                    (
                        "Box_Data",
                        box_p2p.data2recv,
                        box_p2p.data2send,
                        box_p2p.rec_cts,
                        box_p2p.send_cts,
                    ),
                    (
                        "Box_TS",
                        box_p2p.ts2recv,
                        box_p2p.ts2send,
                        box_p2p.rec_cts,
                        box_p2p.send_cts,
                    ),
                ]

                # if torch.equal(mem_p2p.nid2recv, box_p2p.nid2recv) == False:
                #     logger.debug("Error: Mem P2P NIDs and Box P2P NIDs to send do not match.")
                #     logger.debug("Mem NIDs to send: %s", mem_p2p.nid2send)
                #     logger.debug("Box NIDs to send: %s", box_p2p.nid2send)
                #     raise ValueError("Mismatched NIDs in P2P send buffers.")

                # 3. 执行所有通信

                works = []
                logger.debug(
                    f"Rank {self.env.local_rank} starting P2P communication, count={self.count}"
                )
                self.count += 1
                for i, (name, recv_t, send_t, rec_cts, send_cts) in enumerate(p2p_ops):
                    # --- [DEBUG] 强力检查代码开始 ---
                    total_send = sum(send_cts)
                    real_shape = send_t.shape[0]
                    if total_send != real_shape:
                        import sys

                        # logger.debug(
                        #     f"\n[CRITICAL ERROR] P2P Mismatch on Rank {dist.get_rank()} | Op: {name}"
                        # )
                        # logger.debug(f"  > Sum of send_cts: {total_send}")
                        # logger.debug(f"  > Actual Tensor dim0: {real_shape}")
                        # logger.debug(f"  > Tensor Full Shape: {send_t.shape}")
                        # logger.debug(f"  > Send Cts List: {send_cts}")
                        sys.stdout.flush()
                        # 抛出异常让程序停在这里
                        raise RuntimeError(
                            f"Split size mismatch in {name}: planned {total_send} vs actual {real_shape}"
                        )
                    # --- [DEBUG] 强力检查代码结束 ---

                    work = dist.all_to_all_single(
                        output=recv_t,
                        input=send_t,
                        output_split_sizes=rec_cts,
                        input_split_sizes=send_cts,
                        group=self.nccl_p2p,
                        async_op=True,
                    )
                    works.append(work)

                logger.debug(
                    f"[Rank {self.env.local_rank}] Posted all NCCL P2P ops for count {self.count}. Waiting for completion..."
                )
                self.p2p_sem.release()  # 释放信号量，允许下一轮 P2P 任务进入
                # # 4. 组装结果
                # delta_mem = MemTsNid(
                #     nid=mem_p2p.nid2recv, data=mem_p2p.data2recv, ts=mem_p2p.ts2recv
                # )
                # delta_box = BoxTsNid(
                #     nid=box_p2p.nid2recv, data=box_p2p.data2recv, ts=box_p2p.ts2recv
                # )

                # delta = MemNBox(mem=delta_mem, box=delta_box)

                # # 5. 记录 Event 并发送
                # ready_event: CudaEventLike = torch.cuda.Event()
                # ready_event.record(torch.cuda.current_stream())

                self.delta_q.put((mem_p2p, box_p2p, works))
                logger.debug(
                    f"Rank {self.env.local_rank} finished P2P communication, posted delta to delta_q, count={self.count}"
                )
