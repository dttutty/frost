from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import torch
import torch.distributed as dist


Role = Literal["worker", "host"]


@dataclass(frozen=True, kw_only=True)
class Env:
    local_rank: int
    world_size: int

    @cached_property
    def n_gpu(self) -> int:
        return self.world_size - 1

    @cached_property
    def is_host(self) -> bool:
        return self.role == "host"

    @cached_property
    def is_worker(self) -> bool:
        return self.role == "worker"

    @cached_property
    def role(self) -> Role:
        return "host" if self.local_rank == self.host_rank else "worker"

    @cached_property
    def host_rank(self) -> int:
        return self.world_size - 1

    @classmethod
    def initial(cls) -> Env:
        assert dist.is_initialized()
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        if local_rank != world_size - 1:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
            # torch.cuda.set_device(0)
            torch.cuda.set_device(local_rank)

        return cls(
            local_rank=local_rank,
            world_size=world_size,
        )
