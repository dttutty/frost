from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist


Role = Literal["worker", "host"]


@dataclass(frozen=True, kw_only=True)
class Env:
    local_rank: int
    world_size: int
    host_rank: int  # 约定为 world_size - 1
    role: Role  # "worker" / "host"
    n_gpu: int  # = world_size - 1（每 worker 1 张）

    @property
    def is_host(self) -> bool:
        return self.role == "host"

    @classmethod
    def initial(cls) -> "Env":
        assert dist.is_initialized()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if rank != world_size - 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            torch.cuda.set_device(0)
        host_rank = world_size - 1  # the last rank is host
        role: Role = "host" if rank == host_rank else "worker"

        n_gpu = world_size - 1

        return cls(
            local_rank=rank,
            world_size=world_size,
            host_rank=host_rank,
            role=role,
            n_gpu=n_gpu,
        )
