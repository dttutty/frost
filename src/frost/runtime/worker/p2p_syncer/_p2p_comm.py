import logging
import threading

import torch
import torch.distributed as dist

from frost.aliases.syncer import BoxP2PQ, MemP2PQ
from frost.aliases.worker import DeltaQ
from frost.datatypes.env import Env
from frost.runtime._thread_base import StreamThreadBase

logger = logging.getLogger(__name__)


class P2PQHandler(StreamThreadBase):
    def __init__(
        self,
        env: Env,
        mem_p2p_q: MemP2PQ,
        box_p2p_q: BoxP2PQ,
        nccl_p2p: dist.ProcessGroup,
        p2p_sem: threading.Semaphore,
        stop_ev: threading.Event,
        delta_q: DeltaQ,
        device: torch.device,
        use_shm_instead: bool = False,
    ) -> None:
        super().__init__(name="p2p_comm", stop_ev=stop_ev)
        self.device = device
        self.mem_p2p_q = mem_p2p_q
        self.box_p2p_q = box_p2p_q
        self.p2p_sem = p2p_sem
        self.env = env
        self.stop_ev = stop_ev
        self.delta_q = delta_q
        self.nccl_p2p = nccl_p2p
        self.use_shm_instead = use_shm_instead

        self.count = 0

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            # Dequeue next P2P task on the CPU side
            mem_p2p_tuple = self.mem_p2p_q.get(block=True)
            if mem_p2p_tuple is None:
                # logger.debug(
                #     f"[Rank {self.env.local_rank}] Received shutdown signal in P2P handler."
                # )
                break
            # logger.debug(f"[Rank {self.env.local_rank}] Got Mem P2P task: {mem_p2p_tuple}")
            # Fast path: no P2P work this step, pass through
            if mem_p2p_tuple == "nothing_to_p2p":
                box_p2p_tuple = self.box_p2p_q.get(block=True)
                if box_p2p_tuple is None:
                    break
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
            # A mem task implies a corresponding box task
            box_p2p_tuple = self.box_p2p_q.get(block=True)
            if box_p2p_tuple is None:
                break
            # if total_wait > 2.0:
            #     logger.debug(
            #         f"[Rank {self.env.local_rank}] P2P Blocked! Total: {total_wait:.2f}ms | Mem: {mem_wait_ms:.2f}ms | Box: {box_wait_ms:.2f}ms"
            # )
            assert box_p2p_tuple != "nothing_to_p2p"
            box_p2p, box_event = box_p2p_tuple
            torch.cuda.current_stream().wait_event(mem_event)
            torch.cuda.current_stream().wait_event(box_event)

            # 2. Build the NCCL op list in a flat loop to submit all kernels
            #    back-to-back with no Python logic between them.
            #    Order: Mem_NID -> Mem_Data -> Mem_TS -> Box_NID -> Box_Data -> Box_TS
            p2p_ops = [
                # (Name, recv_tensor, send_tensor, recv_counts, send_counts)
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

            # if torch.equal(mem_p2p.nid2recv, box_p2p.nid2recv) == False:
            #     logger.debug("Error: Mem P2P NIDs and Box P2P NIDs to send do not match.")
            #     logger.debug("Mem NIDs to send: %s", mem_p2p.nid2send)
            #     logger.debug("Box NIDs to send: %s", box_p2p.nid2send)
            #     raise ValueError("Mismatched NIDs in P2P send buffers.")

            # 3. Submit all NCCL all_to_all operations

            works = []
            logger.debug(
                f"Rank {self.env.local_rank} starting P2P communication, count={self.count}"
            )
            self.count += 1
            if self.use_shm_instead:
                self.p2p_sem.release()
                self.delta_q.put((None, None, []))
                logger.debug(
                    "Rank %s skipped NCCL P2P and posted empty delta via shared memory, count=%s",
                    self.env.local_rank,
                    self.count,
                )
                continue

            for _, (name, recv_t, send_t, recv_counts, send_counts) in enumerate(
                p2p_ops
            ):
                # --- [DEBUG] sanity check start ---
                total_send = sum(send_counts)
                real_shape = send_t.shape[0]
                if total_send != real_shape:
                    import sys

                    # logger.debug(
                    #     f"\n[CRITICAL ERROR] P2P Mismatch on Rank {dist.get_rank()} | Op: {name}"
                    # )
                    # logger.debug(f"  > Sum of send_counts: {total_send}")
                    # logger.debug(f"  > Actual Tensor dim0: {real_shape}")
                    # logger.debug(f"  > Tensor Full Shape: {send_t.shape}")
                    # logger.debug(f"  > Send Cts List: {send_counts}")
                    sys.stdout.flush()
                    # Raise immediately so the bug is visible
                    raise RuntimeError(
                        f"Split size mismatch in {name}: planned {total_send} vs actual {real_shape}"
                    )
                # --- [DEBUG] sanity check end ---

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
            # # 4. Assemble results
            # delta_mem = MemEntry(
            #     nid=mem_p2p.nid2recv, data=mem_p2p.data2recv, ts=mem_p2p.ts2recv
            # )
            # delta_box = BoxEntry(
            #     nid=box_p2p.nid2recv, data=box_p2p.data2recv, ts=box_p2p.ts2recv
            # )

            # delta = MemNBox(mem=delta_mem, box=delta_box)

            # # 5. Record CUDA event and send to delta_q
            # ready_event: CudaEventLike = torch.cuda.Event()
            # ready_event.record(torch.cuda.current_stream())

            self.delta_q.put((mem_p2p, box_p2p, works))
            logger.debug(
                f"Rank {self.env.local_rank} finished P2P communication, posted delta to delta_q, count={self.count}"
            )
