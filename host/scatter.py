import logging
import threading
from queue import Queue as Q
from typing import Literal

import torch.distributed as dist

from schemas.alias import MetaBundle
from schemas.datatypes.env import Env
from schemas.datatypes.misc import Meta, Result
from schemas.datatypes.plan import DyPlan, PlanSet, StPlan
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
        self.group: dist.ProcessGroup = group

        assert self.local_rank == self.host_rank

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
            group=self.group,
        )

    def gather_receive(self) -> Result:
        assert self.local_rank == self.host_rank, "只有 Host 进程可以 gather"

        gathered_list: list[Result | None] = [None] * self.world_size

        # Host 也必须参与，并提供一个“占位”的发送值
        dist.gather_object(
            float(0),
            gathered_list,
            dst=self.host_rank,  # Host 发送的占位数据
        )
        data_0_result = gathered_list[0]
        assert data_0_result is not None, "Worker 返回的 Result 不能为 None
        # 丢掉 Host 自己的槽位（最后一个），只返回 workers 的结果
        return data_0_result

    def logic(self) -> None:
        while not self.stop_ev.is_set():
            meta_2_scatter = self.in_q.get(block=True)
            if meta_2_scatter is None:
                self.scatter_send(None)
                break
            elif meta_2_scatter == "end_of_epoch":
                self.scatter_send("end_of_epoch")
                data = self.gather_receive()
                self.result_q.put(data)
            else:
                assert len(meta_2_scatter) == self.n_gpu
                mem_plans = mail_plans = nfeat_plans = None
                if self.mem_plan_out_q is not None:
                    mem_plans = self.mem_plan_out_q.get(block=True)
                if self.mail_plan_out_q is not None:
                    mail_plans = self.mail_plan_out_q.get(block=True)
                if self.nfeat_plan_out_q is not None:
                    nfeat_plans = self.nfeat_plan_out_q.get(block=True)
                for i in range(self.n_gpu):
                    plan_set = PlanSet(
                        mem=mem_plans[i] if mem_plans is not None else None,
                        box=mail_plans[i] if mail_plans is not None else None,
                        nfeat=nfeat_plans[i] if nfeat_plans is not None else None,
                    )
                    plan_set = plan_set.tensorfy()
                    plan_set = plan_set.sort()
                    meta_2_scatter[i].plan_set = plan_set
                    del meta_2_scatter[i].state

                self.scatter_send(meta_2_scatter)
