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
        logger.debug(
            f"Scatter: initializing with n_gpu={n_gpu}, world_size={env.world_size}"
        )
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
        logger.debug(f"Scatter: initialized on host_rank={self.host_rank}")

    def scatter_send(self, data: list[Meta] | None | Literal["end_of_epoch"]) -> None:
        assert self.local_rank == self.host_rank, "只有 Host 进程可以 scatter"
        scatter_list: list[Meta | None | Literal["end_of_epoch"]]

        if data == "end_of_epoch":
            logger.debug(
                f"Scatter: preparing to scatter end_of_epoch to {self.world_size} processes"
            )
            scatter_list = ["end_of_epoch"] * self.world_size
        elif isinstance(data, list):
            logger.debug(
                f"Scatter: preparing to scatter {len(data)} meta objects to {self.ngpus} GPUs"
            )
            assert len(data) == self.ngpus, (
                f"需要为 {self.ngpus} 个 GPU worker 准备 payload, 但你给了 {len(data)} 个"
            )
            scatter_list = [*data, None]  # Host 自己的槽位留 None
        else:
            logger.debug(
                f"Scatter: preparing to scatter None (stop signal) to {self.world_size} processes"
            )
            scatter_list = [None] * self.world_size

        logger.debug(
            f"Scatter: executing scatter_object_list, scatter_list length={len(scatter_list)}"
        )
        dist.scatter_object_list(
            [None],  # Host 自己的接收缓冲区（没用）
            scatter_list,
            src=self.host_rank,
            group=self.group,
        )
        logger.debug("Scatter: scatter_object_list complete")

    def gather_receive(self) -> Result:
        logger.debug("Scatter: preparing to gather results from workers")
        assert self.local_rank == self.host_rank, "只有 Host 进程可以 gather"

        gathered_list: list[Result | None] = [None] * self.world_size

        # Host 也必须参与，并提供一个"占位"的发送值
        logger.debug(f"Scatter: executing gather_object, world_size={self.world_size}")
        dist.gather_object(
            float(0),
            gathered_list,
            dst=self.host_rank,  # Host 发送的占位数据
        )
        logger.debug("Scatter: gather_object complete")

        data_0_result = gathered_list[0]
        assert data_0_result is not None, "Worker 返回的 Result 不能为 None"
        logger.debug(
            f"Scatter: received result from worker 0: val_auc={data_0_result.val_auc:.4f}, val_ap={data_0_result.val_ap:.4f}"
        )
        # 丢掉 Host 自己的槽位（最后一个），只返回 workers 的结果
        return data_0_result

    def logic(self) -> None:
        batch_count = 0
        logger.debug("Scatter: starting logic loop")

        while not self.stop_ev.is_set():
            logger.debug("Scatter: waiting for data from in_q")
            meta_2_scatter = self.in_q.get(block=True)
            logger.debug(f"Scatter: received data type={type(meta_2_scatter).__name__}")

            if meta_2_scatter is None:
                logger.debug("Scatter: received None, sending stop signal and exiting")
                self.scatter_send(None)
                break
            elif meta_2_scatter == "end_of_epoch":
                logger.debug(
                    "Scatter: received end_of_epoch, scattering and gathering results"
                )
                self.scatter_send("end_of_epoch")
                data = self.gather_receive()
                logger.debug("Scatter: putting result to result_q")
                self.result_q.put(data)
            else:
                batch_count += 1
                logger.debug(
                    f"Scatter: processing batch #{batch_count}, n_gpu={len(meta_2_scatter)}"
                )
                assert len(meta_2_scatter) == self.n_gpu

                mem_plans = mail_plans = nfeat_plans = None

                if self.mem_plan_out_q is not None:
                    logger.debug("Scatter: getting mem plans from mem_plan_out_q")
                    mem_plans = self.mem_plan_out_q.get(block=True)
                    logger.debug(
                        f"Scatter: received {len(mem_plans) if mem_plans else 0} mem plans"
                    )

                if self.mail_plan_out_q is not None:
                    logger.debug("Scatter: getting mail plans from mail_plan_out_q")
                    mail_plans = self.mail_plan_out_q.get(block=True)
                    logger.debug(
                        f"Scatter: received {len(mail_plans) if mail_plans else 0} mail plans"
                    )

                if self.nfeat_plan_out_q is not None:
                    logger.debug("Scatter: getting nfeat plans from nfeat_plan_out_q")
                    nfeat_plans = self.nfeat_plan_out_q.get(block=True)
                    logger.debug(
                        f"Scatter: received {len(nfeat_plans) if nfeat_plans else 0} nfeat plans"
                    )

                logger.debug(f"Scatter: attaching plans to {self.n_gpu} meta objects")
                for i in range(self.n_gpu):
                    plan_set = PlanSet(
                        mem=mem_plans[i] if mem_plans is not None else None,
                        box=mail_plans[i] if mail_plans is not None else None,
                        nfeat=nfeat_plans[i] if nfeat_plans is not None else None,
                    )
                    logger.debug(
                        f"Scatter: tensorifying and sorting plan_set for GPU {i}"
                    )
                    meta_2_scatter[i].plan_set = plan_set
                    meta_2_scatter[i].state = None  # 清理 state，节省传输开销

                logger.debug(f"Scatter: scattering batch #{batch_count} to workers")
                self.scatter_send(meta_2_scatter)
                logger.debug(f"Scatter: batch #{batch_count} scattered successfully")

        logger.debug(f"Scatter: exiting logic loop, processed {batch_count} batches")
