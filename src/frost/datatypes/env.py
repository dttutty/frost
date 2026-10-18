"""Distributed process-group environment descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

Role = Literal["worker", "host"]


@dataclass(frozen=True, kw_only=True)
class Env:
    """Immutable snapshot of the distributed environment for the current process.

    Created once via ``Env.initial()`` after ``dist.init_process_group`` and
    shared throughout the process lifetime.

    Attributes:
        local_rank: Rank of the current process within the process group.
        world_size: Total number of processes in the group.
        role: Either ``"worker"`` (GPU trainer) or ``"host"`` (data pipeline).
        is_host: ``True`` iff this process is the host/master.
        is_worker: ``True`` iff this process is a GPU worker.
        n_gpu: Number of GPU worker processes (``world_size - 1``).
        host_rank: Rank of the host process (always ``world_size - 1``).
    """

    local_rank: int
    world_size: int
    role: Role
    is_host: bool
    is_worker: bool
    n_gpu: int
    host_rank: int

    @classmethod
    def initial(cls) -> Env:
        """Construct an ``Env`` from the already-initialised process group.

        Assigns the highest rank as the host and all lower ranks as workers.
        Also calls ``torch.cuda.set_device`` for worker processes.

        Returns:
            A fully populated, frozen ``Env`` instance.

        Raises:
            AssertionError: If ``dist.init_process_group`` has not been called.
        """
        assert dist.is_initialized()
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        if local_rank != world_size - 1:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
            # torch.cuda.set_device(0)
            torch.cuda.set_device(local_rank)
        role: Role = "host" if local_rank == world_size - 1 else "worker"
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
