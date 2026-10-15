"""Cross-worker ordering for direct CPU dynamic-state writeback."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch.distributed as dist


class DirectWritebackSynchronizer:
    """Coordinate one direct writeback per worker in each macro-batch.

    The group must contain only the workers and use Gloo. The first barrier
    prevents any worker from changing shared state while a peer is still
    reading that macro-batch. The last barrier completes every worker's
    writeback before any caller releases its next-batch H2D credit.
    """

    def __init__(self, group: dist.ProcessGroup) -> None:
        if dist.get_backend(group) != dist.Backend.GLOO:
            raise ValueError("Direct writeback synchronization requires a Gloo group.")
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        if rank < 0 or world_size < 1:
            raise ValueError("The current process must belong to the worker group.")
        self.group = group
        self.rank = rank
        self.world_size = world_size

    @contextmanager
    def transaction(self, *, serialize_writes: bool) -> Iterator[None]:
        """Protect reads, write memory and mailbox, then release the next batch.

        Raw C1/C2 updates may target the same node on multiple workers, so
        serialize those writes in worker-rank order. C3 has a unique owner per
        updated node and can write in parallel. A failed write must propagate;
        peers will fail through the process group's timeout rather than advance
        past an incomplete transaction.
        """
        if self.world_size == 1:
            yield
            return

        # All workers have finished reading this macro-batch before writes start.
        dist.barrier(group=self.group)

        if serialize_writes:
            for _ in range(self.rank):
                dist.barrier(group=self.group)
            yield
            for _ in range(self.rank, self.world_size):
                dist.barrier(group=self.group)
        else:
            yield
            dist.barrier(group=self.group)
