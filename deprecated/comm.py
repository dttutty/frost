import logging
from typing import Literal

import torch.distributed as dist

from schemas.datatypes.env import Env
from schemas.datatypes.misc import Meta


logger = logging.getLogger(__name__)


class Distributor:
    """
    一个封装了 Host-Workerw 模式下 torch.distributed 通信逻辑的管理器。

    假设:
    1. Host 进程是 world_size - 1。
    2. Worker 进程是 0 到 world_size - 2。
    """

    def __init__(self, env: Env, group: dist.ProcessGroup) -> None:
        # 1. 在构造时才获取分布式状态，而不是在全局
        self.world_size: int = env.world_size
        self.local_rank: int = env.local_rank

        # 2. 将所有全局变量变为实例属性
        self.host_rank: int = env.host_rank
        self.ngpus: int = env.n_gpu
        self.group: dist.ProcessGroup = group

        assert self.local_rank == self.host_rank
        # 3. 状态缓冲区也由类实例管理

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
