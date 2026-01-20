from typing import Any, List

import torch
import torch.distributed as dist

from schemas.datatypes import Env, Meta


class Distributor:
    """
    一个封装了 Host-Worker 模式下 torch.distributed 通信逻辑的管理器。

    假设:
    1. Host 进程是 world_size - 1。
    2. Worker 进程是 0 到 world_size - 2。
    """

    def __init__(self, env: Env, group: dist.ProcessGroup):
        # 1. 在构造时才获取分布式状态，而不是在全局
        self.world_size: int = env.world_size
        self.local_rank: int = env.local_rank

        # 2. 将所有全局变量变为实例属性
        self.host_rank: int = env.host_rank
        self.ngpus: int = env.n_gpu
        self.group: dist.ProcessGroup = group

        assert self.local_rank == self.host_rank
        # 3. 状态缓冲区也由类实例管理
        self.state_buf: torch.Tensor = torch.empty(1, dtype=torch.int32, device="cpu")

    def scatter_send(self, metas: List[Meta]):
        assert self.local_rank == self.host_rank, "只有 Host 进程可以 scatter"
        assert len(metas) == self.ngpus, (
            f"需要为 {self.ngpus} 个 GPU worker 准备 payload, 但你给了 {len(metas)} 个"
        )

        scatter_list = metas + [None]

        dist.scatter_object_list(
            [None],  # Host 自己的接收缓冲区（没用）
            scatter_list,
            src=self.host_rank,
            group=self.group,
        )

    def gather_receive(self) -> List[Any]:
        assert self.local_rank == self.host_rank, "只有 Host 进程可以 gather"

        gathered_list = [None] * self.world_size

        # Host 也必须参与，并提供一个“占位”的发送值
        dist.gather_object(
            float(0),
            gathered_list,
            dst=self.host_rank,  # Host 发送的占位数据
        )

        # 丢掉 Host 自己的槽位（最后一个），只返回 workers 的结果
        return gathered_list[:-1]
