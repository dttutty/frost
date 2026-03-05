import logging
import threading
from queue import Queue as Q
from typing import Literal, cast

import torch
import torch.distributed as dist

from frost.aliases.host import MetaBundle
from frost.datatypes.env import Env
from frost.datatypes.meta import Meta
from frost.datatypes.plan import DyPlan, PlanSet, StPlan
from frost.datatypes.result import Result
from frost.runtime._thread_base import ThreadBase
from frost.runtime._wire import ControlWire, MetaWire, meta_to_wire, pack_wire_message

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
        stop_ev: threading.Event,
        ring_buf=None,
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

        # 2. 将所有全局变量变为实例属性
        self.host_rank: int = env.host_rank
        self.ngpus: int = env.n_gpu
        self._gather_seq = 0
        self._ctrl_send_seq = 0

        assert self.local_rank == self.host_rank

    def gather_receive(self) -> Result:
        assert self.local_rank == self.host_rank, "只有 Host 进程可以 gather"

        gathered_list: list[Result | None] = [None] * self.world_size

        # Host 也必须参与，并提供一个"占位"的发送值
        self._gather_seq += 1
        logger.debug(
            "Scatter gather_object start: seq=%d rank=%d",
            self._gather_seq,
            self.local_rank,
        )
        dist.gather_object(
            None,
            gathered_list,
            dst=self.host_rank,  # Host 发送的占位数据
        )
        logger.debug(
            "Scatter gather_object end: seq=%d rank=%d",
            self._gather_seq,
            self.local_rank,
        )

        data_0_result = gathered_list[0]
        assert data_0_result is not None, "Worker 返回的 Result 不能为 None"
        # 丢掉 Host 自己的槽位（最后一个），只返回 workers 的结果
        return data_0_result

    def logic(self) -> None:
        def _send_wire(dst_rank: int, msg: ControlWire) -> None:
            payload = pack_wire_message(msg)
            nbytes = torch.tensor([len(payload)], dtype=torch.long)
            dist.send(nbytes, dst=dst_rank)
            if nbytes.item() > 0:
                data = torch.tensor(list(payload), dtype=torch.uint8)
                dist.send(data, dst=dst_rank)

        worker_ranks = [r for r in range(self.world_size) if r != self.host_rank]

        def _broadcast_control(kind: Literal["end_of_epoch", "stop"]) -> None:
            for dst_rank in worker_ranks:
                _send_wire(dst_rank, {"kind": kind, "meta": None})

        while not self.stop_ev.is_set():
            meta_2_scatter = self.in_q.get(block=True)

            if meta_2_scatter is None:
                _broadcast_control("stop")
                raise RuntimeError(
                    "Scatter received None, shutting down..."
                )  # 这里抛异常是为了
                break
            elif meta_2_scatter == "end_of_epoch":
                logger.warning(
                    "end_of_epoch received by Scatter, sending control signal to Workers..."
                )
                _broadcast_control("end_of_epoch")
                data = self.gather_receive()
                self.result_q.put(data)
                continue

            else:
                if self.mem_plan_out_q is not None:
                    mem_plan = self.mem_plan_out_q.get(block=True)
                if self.mail_plan_out_q is not None:
                    mail_plan = self.mail_plan_out_q.get(block=True)
                if self.nfeat_plan_out_q is not None:
                    nfeat_plan = self.nfeat_plan_out_q.get(block=True)
                for i, m in enumerate(meta_2_scatter):
                    safe_mem = mem_plan[i].clone() if self.mem_plan_out_q else None
                    safe_mail = mail_plan[i].clone() if self.mail_plan_out_q else None
                    safe_nfeat = (
                        nfeat_plan[i].clone() if self.nfeat_plan_out_q else None
                    )
                    m.plan_set = PlanSet(
                        mem=safe_mem,
                        box=safe_mail,
                        nfeat=safe_nfeat,
                    )
                meta_2_scatter = cast(list[Meta | None], meta_2_scatter)
                wire_payload: list[MetaWire | None] = [
                    meta_to_wire(cast(Meta, m)) if m is not None else None
                    for m in meta_2_scatter
                ]
                try:
                    for dst_rank, per_rank_meta in zip(worker_ranks, wire_payload, strict=True):
                        _send_wire(
                            dst_rank,
                            {"kind": "meta", "meta": per_rank_meta},
                        )
                except Exception as e:
                    logger.error(f"Scatter point-to-point send failed: {e}")
                    raise RuntimeError(
                        f"Scatter point-to-point send failed: {e}"
                    ) from e
