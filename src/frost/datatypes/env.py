from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

Role = Literal["worker", "host"]


@dataclass(frozen=True, kw_only=True)
class Env:
    local_rank: int
    world_size: int
    role: Literal["worker", "host"]
    is_host: bool
    is_worker: bool
    n_gpu: int
    host_rank: int

    @classmethod
    def initial(cls) -> Env:
        assert dist.is_initialized()
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        if local_rank != world_size - 1:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
            # torch.cuda.set_device(0)
            torch.cuda.set_device(local_rank)
        role = "host" if local_rank == world_size - 1 else "worker"
        is_host = role == "host"
        is_worker = role == "worker"
        n_gpu = world_size - 1
        host_rank = world_size - 1

        return cls(
            local_rank=local_rank,
            world_size=world_size,
            role=role,
            is_host=is_host,
            is_worker=is_worker,
            n_gpu=n_gpu,
            host_rank=host_rank,
        )
