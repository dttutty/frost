import logging
import threading

import torch
import torch.distributed as dist

from frost.runtime.worker.types import BoxP2PQ, MemP2PQ
from frost.runtime.worker.types import DeltaQueue
from frost.datatypes.env import Env
from frost.runtime._thread_base import StreamThreadBase

logger = logging.getLogger(__name__)


def _copy_single_rank_p2p_round(mem_p2p, box_p2p) -> None:
    """Bypass NCCL when the P2P group has a single rank.

    ``all_to_all_single`` with world-size 1 is semantically just a self-copy.
    Executing it through NCCL in the single-worker pipeline has been observed to
    corrupt the next batch's fresh delta under overlap, so we copy the buffers
    directly on GPU instead.
    """

    mem_p2p.recv_nid.copy_(mem_p2p.send_nid)
    mem_p2p.data2recv.copy_(mem_p2p.data2send)
    mem_p2p.ts2recv.copy_(mem_p2p.ts2send)
    box_p2p.recv_nid.copy_(box_p2p.send_nid)
    box_p2p.data2recv.copy_(box_p2p.data2send)
    box_p2p.ts2recv.copy_(box_p2p.ts2send)


def _safe_p2p_world_size(group) -> int | None:
    try:
        return dist.get_world_size(group)
    except Exception:
        return None


class P2PQHandler(StreamThreadBase):
    def __init__(
        self,
        env: Env,
        mem_p2p_q: MemP2PQ,
        box_p2p_q: BoxP2PQ,
        nccl_p2p: dist.ProcessGroup,
        p2p_sem: threading.Semaphore,
        stop_ev: threading.Event,
        delta_queue: DeltaQueue,
    ) -> None:
        super().__init__(name="p2p_comm", stop_ev=stop_ev)
        self.mem_p2p_q = mem_p2p_q
        self.box_p2p_q = box_p2p_q
        self.p2p_sem = p2p_sem
        self.env = env
        self.stop_ev = stop_ev
        self.delta_queue = delta_queue
        self.nccl_p2p = nccl_p2p

        self.count = 0

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            mem_task_done = False
            box_task_done = False
            # Dequeue next P2P task on the CPU side
            try:
                mem_p2p_tuple = self.mem_p2p_q.get(block=True)
                mem_task_done = True
                if mem_p2p_tuple is None:
                    # logger.debug(
                    #     f"[Rank {self.env.local_rank}] Received shutdown signal in P2P handler."
                    # )
                    break
                # logger.debug(f"[Rank {self.env.local_rank}] Got Mem P2P task: {mem_p2p_tuple}")
                # Fast path: no P2P work this step, pass through
                if mem_p2p_tuple == "nothing_to_p2p":
                    box_p2p_tuple = self.box_p2p_q.get(block=True)
                    box_task_done = True
                    if box_p2p_tuple is None:
                        break
                    assert box_p2p_tuple == "nothing_to_p2p"
                    self.delta_queue.put("nothing")
                    # Keep semaphore balanced so the next step doesn't block after no-op P2P.
                    self.p2p_sem.release()
                    continue
                mem_p2p, mem_event = mem_p2p_tuple
                # A mem task implies a corresponding box task
                box_p2p_tuple = self.box_p2p_q.get(block=True)
                box_task_done = True
                if box_p2p_tuple is None:
                    break
                assert box_p2p_tuple != "nothing_to_p2p"
                box_p2p, box_event = box_p2p_tuple
                torch.cuda.current_stream().wait_event(mem_event)
                torch.cuda.current_stream().wait_event(box_event)

                p2p_ops = [
                    (
                        "Mem_NID",
                        mem_p2p.recv_nid,
                        mem_p2p.send_nid,
                        mem_p2p.recv_counts,
                        mem_p2p.send_counts,
                    ),
                    (
                        "Mem_Data",
                        mem_p2p.data2recv,
                        mem_p2p.data2send,
                        mem_p2p.recv_counts,
                        mem_p2p.send_counts,
                    ),
                    (
                        "Mem_TS",
                        mem_p2p.ts2recv,
                        mem_p2p.ts2send,
                        mem_p2p.recv_counts,
                        mem_p2p.send_counts,
                    ),
                    (
                        "Box_NID",
                        box_p2p.recv_nid,
                        box_p2p.send_nid,
                        box_p2p.recv_counts,
                        box_p2p.send_counts,
                    ),
                    (
                        "Box_Data",
                        box_p2p.data2recv,
                        box_p2p.data2send,
                        box_p2p.recv_counts,
                        box_p2p.send_counts,
                    ),
                    (
                        "Box_TS",
                        box_p2p.ts2recv,
                        box_p2p.ts2send,
                        box_p2p.recv_counts,
                        box_p2p.send_counts,
                    ),
                ]

                works = []
                logger.debug(
                    f"Rank {self.env.local_rank} starting P2P communication, count={self.count}"
                )
                self.count += 1
                world_size = _safe_p2p_world_size(self.nccl_p2p)
                if world_size == 1:
                    _copy_single_rank_p2p_round(mem_p2p, box_p2p)
                    # Single-rank bypass still launches GPU copy kernels on this
                    # thread's current stream. Unlike the NCCL path there is no
                    # work handle for DynamicAssembler to wait on, so we must
                    # synchronize here before publishing the delta payload.
                    torch.cuda.current_stream().synchronize()
                    self.p2p_sem.release()
                    self.delta_queue.put((mem_p2p, box_p2p, []))
                    logger.debug(
                        "Rank %s bypassed single-rank NCCL P2P with direct GPU copies, count=%s",
                        self.env.local_rank,
                        self.count,
                    )
                    continue

                for _, (name, recv_t, send_t, recv_counts, send_counts) in enumerate(
                    p2p_ops
                ):
                    total_send = sum(send_counts)
                    real_shape = send_t.shape[0]
                    if total_send != real_shape:
                        import sys

                        sys.stdout.flush()
                        raise RuntimeError(
                            f"Split size mismatch in {name}: planned {total_send} vs actual {real_shape}"
                        )

                    work = dist.all_to_all_single(
                        output=recv_t,
                        input=send_t,
                        output_split_sizes=recv_counts,
                        input_split_sizes=send_counts,
                        group=self.nccl_p2p,
                        async_op=True,
                    )
                    works.append(work)

                logger.debug(
                    f"[Rank {self.env.local_rank}] Posted all NCCL P2P ops for count {self.count}. Waiting for completion..."
                )
                self.p2p_sem.release()  # Allow the next P2P round to proceed
                self.delta_queue.put((mem_p2p, box_p2p, works))
                logger.debug(
                    f"Rank {self.env.local_rank} finished P2P communication, posted delta to delta_queue, count={self.count}"
                )
            finally:
                if mem_task_done:
                    self.mem_p2p_q.task_done()
                if box_task_done:
                    self.box_p2p_q.task_done()
