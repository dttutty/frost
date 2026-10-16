"""Control loop for dynamic memory and mailbox assembly."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from queue import Queue
from typing import Callable, cast

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.models.contracts import RootExpansion
from frost.runtime._thread_base import StreamThreadBase, put_shutdown_sentinel
from frost.runtime.experimental.dynamic_state import (
    DynamicStatePolicy,
    PlannerMode,
    TransportMode,
    resolve_dynamic_state_policy,
)
from frost.runtime.protocols import CudaEvent
from frost.runtime.worker._dynamic_state import (
    DynamicPrepareProfiler,
    DynamicStateAssemblerMixin,
    MemAndBox,
)
from frost.runtime.worker._h2d_credit import CreditGate
from frost.runtime.worker.types import AssembledQ, DeltaQueue, H2DOutQ
from frost.shm.mem_mgr import MemBoxStore
from frost.shm.dynamic_state_sync import SharedDynamicStateSync

logger = logging.getLogger(__name__)


def _timing_enabled() -> bool:
    return os.environ.get("FROST_PIPELINE_TIMING_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class DynamicAssembler(DynamicStateAssemblerMixin, StreamThreadBase):
    """Merge local shared state and P2P deltas, then forward ready batches."""

    def __init__(
        self,
        h2d_out_q: H2DOutQ,
        delta_queue: DeltaQueue,
        full_q: AssembledQ,
        stop_ev: threading.Event,
        device: torch.device,
        h2d_sem: CreditGate,
        mem_box_store: MemBoxStore,  # Pass shared memory instance here
        n_neg: int = 1,
        root_expansion: RootExpansion = RootExpansion.ALL_ROOTS,
        mem_task_q: Queue | None = None,
        box_task_q: Queue | None = None,
        mem_wr_q: Queue | None = None,
        box_wr_q: Queue | None = None,
        dynamic_state_policy: DynamicStatePolicy | None = None,
        wait_for_resident_write: Callable[[torch.cuda.Stream], None] | None = None,
        state_sync: SharedDynamicStateSync | None = None,
    ) -> None:
        super().__init__(name="dy_assembler", stop_ev=stop_ev)
        self.in_q = h2d_out_q
        self.delta_queue = delta_queue
        self.full_q = full_q
        self.device = device
        self.h2d_sem = h2d_sem
        self.shared = mem_box_store
        self.stop_ev = stop_ev
        self.boxsize = mem_box_store.box.shape[1]
        self.d_mail = mem_box_store.box.shape[2]
        self.n_neg = n_neg
        self.root_expansion = root_expansion
        self.mem_task_q = mem_task_q
        self.box_task_q = box_task_q
        self.mem_wr_q = mem_wr_q
        self.box_wr_q = box_wr_q
        self.dynamic_state_policy = (
            dynamic_state_policy
            if dynamic_state_policy is not None
            else resolve_dynamic_state_policy("baseline")
        )
        self.wait_for_resident_write = wait_for_resident_write
        self.state_sync = state_sync
        self._timing_enabled = _timing_enabled()
        self._h2d_credit_wait_s_curve: list[float] = []
        self._prepare_shared_s_curve: list[float] = []
        self._delta_queue_wait_s_curve: list[float] = []
        self._p2p_work_wait_s_curve: list[float] = []
        self._apply_delta_s_curve: list[float] = []
        self._prepare_profiler = DynamicPrepareProfiler(enabled=self._timing_enabled)
        self._shared_gather_buffers = {}

    def logic(self) -> None:
        """Main assembly loop: merge shared + delta data and forward to full_q.

        For each incoming batch:

        1. Waits for the H2D-ready CUDA event.
        2. Acquires two H2D semaphore slots (one for mem write-back, one for
           box write-back) to prevent the pipeline from over-running the writers.
        3. Calls ``prepare_shared_features`` to fill local data from the store.
        4. Blocks on ``delta_queue`` for the P2P result, then calls
           ``apply_delta_features`` to fill remote data.
        5. Records a CUDA event and puts ``(meta, event)`` into ``full_q``.
        """
        epoch_count = getattr(self, "epoch_count", 0)
        dynamic_state_policy = getattr(
            self,
            "dynamic_state_policy",
            resolve_dynamic_state_policy("baseline"),
        )
        timing_enabled = getattr(self, "_timing_enabled", False)
        state_sync = getattr(self, "state_sync", None)
        last_macro = None
        epoch_idx = None

        while not self.stop_ev.is_set():
            in_data = self.in_q.get(block=True)

            if in_data is None:
                put_shutdown_sentinel(self.full_q, None)
                break
            elif in_data == "end_of_epoch":
                if state_sync is not None:
                    if last_macro is None:
                        raise RuntimeError("Shared-state epoch contains no batches")
                    state_sync.close_epoch(last_macro, epoch_id=epoch_idx)
                self.full_q.put("end_of_epoch", block=True)
                last_macro = None
                epoch_idx = None
                continue

            meta, meta_ready_event = in_data
            torch.cuda.current_stream().wait_event(meta_ready_event)

            # ==========================================================
            # OPTIMIZATION: Process local shared memory FIRST
            # We do this BEFORE waiting on the blocking delta_queue
            # ==========================================================
            assert meta.flat_graph is not None
            assert meta.plan_set is not None
            if state_sync is not None:
                state_sync.register_batch(
                    meta.mini_batch_idx, meta.macro_batch_idx, meta.epoch_idx
                )
                last_macro = meta.macro_batch_idx
                epoch_idx = meta.epoch_idx
            else:
                self._drain_prior_eval_writebacks(split=meta.split)
            wait_for_batch = getattr(self.h2d_sem, "wait_for_batch", None)
            h2d_wait_start = time.perf_counter()
            if callable(wait_for_batch):
                wait_for_batch(meta.mini_batch_idx)
            else:
                self.h2d_sem.acquire()
                self.h2d_sem.acquire()
            if state_sync is not None:
                state_sync.wait_before_snapshot(meta.macro_batch_idx)
            h2d_wait_end = time.perf_counter()

            if wait_for_resident_write := getattr(
                self, "wait_for_resident_write", None
            ):
                wait_for_resident_write(torch.cuda.current_stream(self.device))

            prepare_start = time.perf_counter()
            assembly_layout = self.prepare_shared_features(
                graph=meta.flat_graph,
                mem_plan=meta.plan_set.mem_plan,
                box_plan=meta.plan_set.box_plan,
                n_src=meta.n_src,
                n_neg=meta.n_neg,
                pin_memory=dynamic_state_policy.pin_gather_buffers,
                profiler=getattr(self, "_prepare_profiler", None),
            )
            prepare_end = time.perf_counter()
            if state_sync is not None:
                # prepare_shared_features copies all shared rows and APAN ring
                # metadata into private storage before publishing this fence.
                state_sync.snapshot_done(meta.macro_batch_idx)
                if meta.is_last_macro_of_epoch:
                    state_sync.close_epoch(
                        meta.macro_batch_idx, epoch_id=meta.epoch_idx
                    )

            if dynamic_state_policy.transport in {
                TransportMode.DIRECT_CPU,
                TransportMode.GPU_LOCAL,
            }:
                if dynamic_state_policy.planner is PlannerMode.NONE:
                    self._fill_all_required_from_shm(
                        meta.flat_graph, meta.n_src, meta.n_neg
                    )
                if timing_enabled:
                    self._delta_queue_wait_s_curve.append(0.0)
                    self._p2p_work_wait_s_curve.append(0.0)
                    self._apply_delta_s_curve.append(0.0)
                ready_event = cast(CudaEvent, torch.cuda.Event())
                ready_event.record(torch.cuda.current_stream())
                self.full_q.put((meta, ready_event), block=True)
                continue

            # Now block and wait for network data
            delta_wait_start = time.perf_counter()
            delta_tuple = self.delta_queue.get(block=True)
            delta_wait_end = time.perf_counter()
            if delta_tuple is None:
                put_shutdown_sentinel(self.full_q, None)
                break

            logger.debug(
                "DynamicAssembler received delta for batch %s, epoch %s",
                meta.mini_batch_idx,
                epoch_count,
            )

            if delta_tuple != "nothing":
                p2p_wait_s = 0.0
                mem, box, works = delta_tuple
                assert mem is not None
                assert box is not None

                # Wait for P2P network streams to finish receiving
                p2p_wait_start = time.perf_counter()
                for work in works:
                    work.wait()
                p2p_wait_s = time.perf_counter() - p2p_wait_start

                box_data = box.data2recv
                delta_mem = MemEntry(
                    nid=mem.recv_nid, data=mem.data2recv, ts=mem.ts2recv
                )
                delta_box = BoxEntry(nid=box.recv_nid, data=box_data, ts=box.ts2recv)
                delta = MemAndBox(mem=delta_mem, box=delta_box)
                shared_dynamic_plan = meta.plan_set.mem_plan is meta.plan_set.box_plan
                delta = delta.sort_mem_and_box(shared_nid_order=shared_dynamic_plan)

                # ==========================================================
                # Apply the received delta to the already partially-filled graph
                # ==========================================================
                apply_delta_start = time.perf_counter()
                self.apply_delta_features(
                    graph=meta.flat_graph,
                    delta=delta,
                    mem_plan=meta.plan_set.mem_plan,
                    box_plan=meta.plan_set.box_plan,
                    n_src=meta.n_src,
                    n_neg=meta.n_neg,
                    layout=assembly_layout,
                )
                apply_delta_end = time.perf_counter()
            else:
                p2p_wait_s = 0.0
                apply_delta_start = None
                apply_delta_end = None

            if timing_enabled:
                self._h2d_credit_wait_s_curve.append(h2d_wait_end - h2d_wait_start)
                self._prepare_shared_s_curve.append(prepare_end - prepare_start)
                self._delta_queue_wait_s_curve.append(delta_wait_end - delta_wait_start)
                self._p2p_work_wait_s_curve.append(p2p_wait_s)
                self._apply_delta_s_curve.append(
                    0.0
                    if apply_delta_start is None or apply_delta_end is None
                    else apply_delta_end - apply_delta_start
                )

            # Push to the next queue
            ready_event = cast(CudaEvent, torch.cuda.Event())
            ready_event.record(torch.cuda.current_stream())
            self.full_q.put((meta, ready_event), block=True)
            logger.debug(
                f"DynamicAssembler put full meta for batch {meta.mini_batch_idx}, epoch {epoch_count} into full_q"
            )

        if timing_enabled and self._delta_queue_wait_s_curve:
            payload = {
                "system": "frost",
                "rank": getattr(self.device, "index", -1),
                "batch_count": len(self._delta_queue_wait_s_curve),
                "h2d_credit_wait_s_curve": [
                    round(v, 6) for v in self._h2d_credit_wait_s_curve
                ],
                "prepare_shared_s_curve": [
                    round(v, 6) for v in self._prepare_shared_s_curve
                ],
                "delta_queue_wait_s_curve": [
                    round(v, 6) for v in self._delta_queue_wait_s_curve
                ],
                "p2p_work_wait_s_curve": [
                    round(v, 6) for v in self._p2p_work_wait_s_curve
                ],
                "apply_delta_s_curve": [round(v, 6) for v in self._apply_delta_s_curve],
            }
            print(
                "APPENDIX_DYNAMIC_ASSEMBLER_TIMING_CURVE "
                f"{json.dumps(payload, sort_keys=True)}"
            )
        prepare_profiler = getattr(self, "_prepare_profiler", None)
        if timing_enabled and prepare_profiler is not None:
            torch.cuda.current_stream(self.device).synchronize()
            payload = {
                "system": "frost",
                "rank": getattr(self.device, "index", -1),
                **prepare_profiler.payload(),
            }
            print(
                "APPENDIX_DYNAMIC_PREPARE_STAGE_TIMING_CURVE "
                f"{json.dumps(payload, sort_keys=True)}"
            )
