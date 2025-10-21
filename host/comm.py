from typing import Any, List

import torch
import torch.distributed as dist

from common.state import WorkerState


class Distributor:
    """
    一个封装了 Host-Worker 模式下 torch.distributed 通信逻辑的管理器。

    假设:
    1. Host 进程是 world_size - 1。
    2. Worker 进程是 0 到 world_size - 2。
    """

    def __init__(self):
        # 1. 在构造时才获取分布式状态，而不是在全局
        self.world_size: int = dist.get_world_size()
        self.local_rank: int = dist.get_rank()

        # 2. 将所有全局变量变为实例属性
        self.HOST_RANK: int = self.world_size - 1
        self.ngpus: int = self.world_size - 1

        assert self.local_rank == self.HOST_RANK
        # 3. 状态缓冲区也由类实例管理
        self.state_buf: torch.Tensor = torch.empty(1, dtype=torch.int32, device="cpu")

    def broadcast_state(self, state: WorkerState):
        """
        Host (self) 向所有进程广播一个状态。

        Args:
            state (WorkerState): 要广播的枚举状态。
        """
        assert self.local_rank == self.HOST_RANK, "只有 Host 进程可以广播状态"

        # [修改] 不再需要字典查找。
        # 直接从 Enum 成员获取其整数值。
        if not isinstance(state, WorkerState):
            raise TypeError(
                f"broadcast_state 期望一个 WorkerState 枚举成员，但收到了 {type(state)}"
            )

        val = state.value

        print(
            f"Host (Rank {self.HOST_RANK}): Broadcasting state {state.name} ({val}) ..."
        )
        self.state_buf.fill_(val)

        dist.broadcast(self.state_buf, src=self.HOST_RANK)

    def scatter_send(self, batch_payloads: List[Any]):
        """
        Host (self) 向所有 Workers scatter 数据。

        Args:
            batch_payloads: 长度 == ngpus 的 list。
                          payloads[0] 发给 Rank 0
                          payloads[1] 发给 Rank 1
                          ...
        """
        assert self.local_rank == self.HOST_RANK, "只有 Host 进程可以 scatter"
        assert len(batch_payloads) == self.ngpus, (
            f"需要为 {self.ngpus} 个 GPU worker 准备 payload, 但你给了 {len(batch_payloads)} 个"
        )

        print(f"Host (Rank {self.HOST_RANK}): Scattering to {self.ngpus} workers...")

        # 构造长度 world_size 的列表
        scatter_list = list(batch_payloads) + [None]  # Host 自己的槽位放 None

        # Host 自己也必须参与 collective
        dist.scatter_object_list(
            [None],  # Host 自己的接收缓冲区（没用）
            scatter_list,
            src=self.HOST_RANK,
        )

    def gather_receive(self) -> List[Any]:
        """
        Host (self) 从所有进程（包括自己）收集数据，并只返回 Workers 的数据。

        Returns:
            一个 list，长度 == ngpus。
            list[0] 是 Rank 0 的结果
            list[1] 是 Rank 1 的结果
            ...
        """
        assert self.local_rank == self.HOST_RANK, "只有 Host 进程可以 gather"

        gathered_list = [None] * self.world_size

        # Host 也必须参与，并提供一个“占位”的发送值
        dist.gather_object(
            float(0),
            gathered_list,
            dst=self.HOST_RANK,  # Host 发送的占位数据
        )

        # 丢掉 Host 自己的槽位（最后一个），只返回 workers 的结果
        return gathered_list[:-1]
