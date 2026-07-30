import json
import logging
import os
import threading
import time
from queue import Queue as Q
from typing import Literal, cast

import torch.distributed as dist

from frost.runtime.host.types import MetaBundle
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, PlanSet, StaticPlan
from frost.datatypes.result import Result
from frost.runtime._thread_base import ThreadBase

logger = logging.getLogger(__name__)


def _timing_enabled() -> bool:
    return os.environ.get("FROST_HOST_TIMING_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class Scatter(ThreadBase):
    """Background thread that distributes batches to Worker ranks via collective ops.

    Consumes MetaBundles (lists of ``Meta`` objects, one per GPU) from ``in_q``,
    attaches per-GPU ``PlanSet`` data from the planner output queues, and uses
    ``dist.scatter_object_list`` to push each ``Meta`` to its assigned Worker rank.
    At end-of-epoch it also collects results from Workers via ``dist.gather_object``
    and forwards them to ``result_q`` for the Host to consume.

    The thread terminates cleanly when it receives a ``None`` sentinel from ``in_q``.

    Attributes:
        n_gpu: Number of Worker ranks (= world_size - 1).
        in_q: Source queue fed by the Host main thread.
        mem_plan_out_q: Optional queue of per-GPU DynamicPlan lists for memory.
        mail_plan_out_q: Optional queue of per-GPU DynamicPlan lists for mailbox.
        nfeat_plan_out_q: Optional queue of per-GPU StaticPlan lists for nfeat.
        result_q: Queue where gathered epoch Results are placed for the Host.
    """

    def __init__(
        self,
        n_gpu: int,
        in_q: Q[MetaBundle | None | Literal["end_of_epoch"]],
        mem_plan_out_q: Q[list[DynamicPlan] | None] | None,
        mail_plan_out_q: Q[list[DynamicPlan] | None] | None,
        nfeat_plan_out_q: Q[list[StaticPlan] | None] | None,
        env: Env,
        result_q: Q[Result],
        stop_ev: threading.Event,
    ) -> None:
        super().__init__(name="scatter", stop_ev=stop_ev)
        self.n_gpu = n_gpu
        self.in_q = in_q
        self.mem_plan_out_q = mem_plan_out_q
        self.mail_plan_out_q = mail_plan_out_q
        self.nfeat_plan_out_q = nfeat_plan_out_q
        self.stop_ev = stop_ev
        self.result_q = result_q

        self.world_size: int = env.world_size
        self.local_rank: int = env.local_rank
        self.host_rank: int = env.host_rank
        self._gather_seq = 0
        self._timing_enabled = _timing_enabled()
        self._mem_plan_wait_s_curve: list[float] = []
        self._mail_plan_wait_s_curve: list[float] = []
        self._nfeat_plan_wait_s_curve: list[float] = []
        self._scatter_call_s_curve: list[float] = []

        assert self.local_rank == self.host_rank

    def _scatter_to_workers(self, per_worker: list[Meta | str | None]) -> None:
        """Scatter one object to each rank via collective scatter.

        per_worker is indexed by *worker order* (len == n_gpu).
        We build a world_size-length list indexed by rank.
        """
        scatter_input: list[Meta | str | None] = [None] * self.world_size
        worker_ranks = [r for r in range(self.world_size) if r != self.host_rank]
        for rank, obj in zip(worker_ranks, per_worker):
            scatter_input[rank] = obj
        # Host's own slot (scatter_input[host_rank]) stays None
        output: list[None] = [None]
        scatter_start = time.perf_counter()
        dist.scatter_object_list(
            output, scatter_object_input_list=scatter_input, src=self.host_rank
        )
        scatter_end = time.perf_counter()
        if self._timing_enabled:
            self._scatter_call_s_curve.append(scatter_end - scatter_start)

    def _gather_from_workers(self) -> Result:
        """Gather results from all workers via collective gather."""
        self._gather_seq += 1
        gather_list: list[Result | None] = [None] * self.world_size
        dist.gather_object(None, gather_list, dst=self.host_rank)
        # Return result from worker rank 0
        worker_ranks = [r for r in range(self.world_size) if r != self.host_rank]
        result = gather_list[worker_ranks[0]]
        assert isinstance(result, Result), (
            "Object gathered from worker must be a Result"
        )
        return result

    def logic(self) -> None:
        """Main loop: dequeue MetaBundles, attach plans, and scatter to Workers.

        Handles three sentinel values from ``in_q``:
        - ``None``: send shutdown signal to all Workers and exit.
        - ``"end_of_epoch"``: broadcast the sentinel to Workers, gather results.
        - A normal ``MetaBundle``: attach plan data and scatter per-GPU Metas.
        """
        while not self.stop_ev.is_set():
            meta_2_scatter = self.in_q.get(block=True)

            if meta_2_scatter is None:
                self._scatter_to_workers([None] * self.n_gpu)
                # raise RuntimeError("Scatter received None, shutting down...")
                break
            elif meta_2_scatter == "end_of_epoch":
                logger.warning(
                    "end_of_epoch received by Scatter, sending control signal to Workers..."
                )
                self._scatter_to_workers(["end_of_epoch"] * self.n_gpu)
                data = self._gather_from_workers()
                self.result_q.put(data)
                continue

            else:
                mem_plan: list[DynamicPlan] | None = None
                mail_plan: list[DynamicPlan] | None = None
                nfeat_plan: list[StaticPlan] | None = None
                if self.mem_plan_out_q is not None:
                    mem_wait_start = time.perf_counter()
                    mem_plan = self.mem_plan_out_q.get(block=True)
                    mem_wait_end = time.perf_counter()
                    if self._timing_enabled:
                        self._mem_plan_wait_s_curve.append(
                            mem_wait_end - mem_wait_start
                        )
                    if mem_plan is None and self.stop_ev.is_set():
                        self._scatter_to_workers([None] * self.n_gpu)
                        break
                if self.mail_plan_out_q is not None:
                    mail_wait_start = time.perf_counter()
                    mail_plan = self.mail_plan_out_q.get(block=True)
                    mail_wait_end = time.perf_counter()
                    if self._timing_enabled:
                        self._mail_plan_wait_s_curve.append(
                            mail_wait_end - mail_wait_start
                        )
                    if mail_plan is None and self.stop_ev.is_set():
                        self._scatter_to_workers([None] * self.n_gpu)
                        break
                if self.nfeat_plan_out_q is not None:
                    nfeat_wait_start = time.perf_counter()
                    nfeat_plan = self.nfeat_plan_out_q.get(block=True)
                    nfeat_wait_end = time.perf_counter()
                    if self._timing_enabled:
                        self._nfeat_plan_wait_s_curve.append(
                            nfeat_wait_end - nfeat_wait_start
                        )
                    if nfeat_plan is None and self.stop_ev.is_set():
                        self._scatter_to_workers([None] * self.n_gpu)
                        break
                if self.stop_ev.is_set():
                    self._scatter_to_workers([None] * self.n_gpu)
                    break
                for i, m in enumerate(meta_2_scatter):
                    safe_mem = mem_plan[i] if mem_plan is not None else None
                    safe_mail = mail_plan[i] if mail_plan is not None else None
                    safe_nfeat = nfeat_plan[i] if nfeat_plan is not None else None
                    m.plan_set = PlanSet(
                        mem_plan=safe_mem,
                        box_plan=safe_mail,
                        nfeat_plan=safe_nfeat,
                    )
                self._scatter_to_workers(cast(list[Meta | str | None], meta_2_scatter))

        if self._timing_enabled and self._scatter_call_s_curve:
            payload = {
                "system": "frost",
                "rank": self.local_rank,
                "batch_count": len(self._scatter_call_s_curve),
                "mem_plan_wait_s_curve": [
                    round(v, 6) for v in self._mem_plan_wait_s_curve
                ],
                "mail_plan_wait_s_curve": [
                    round(v, 6) for v in self._mail_plan_wait_s_curve
                ],
                "nfeat_plan_wait_s_curve": [
                    round(v, 6) for v in self._nfeat_plan_wait_s_curve
                ],
                "scatter_call_s_curve": [
                    round(v, 6) for v in self._scatter_call_s_curve
                ],
            }
            print(
                "APPENDIX_HOST_SCATTER_TIMING_CURVE "
                f"{json.dumps(payload, sort_keys=True)}"
            )
