"""CPU-only regressions for direct writeback across worker processes."""

from __future__ import annotations

import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from frost.runtime.worker._direct_writeback_sync import DirectWritebackSynchronizer


def _run_two_worker_transactions(
    rank: int,
    rendezvous_file: str,
    data: torch.Tensor,
    timestamps: torch.Tensor,
    read_flags: torch.Tensor,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        group = dist.new_group(ranks=[0, 1], backend="gloo")
        sync = DirectWritebackSynchronizer(group)
        assert sync.rank == rank
        assert sync.world_size == 2

        # Rank 1 is still reading the current batch while rank 0 is ready to
        # write. The entry barrier must keep the old data and timestamp intact.
        if rank == 0:
            read_flags[0] = 1
        else:
            deadline = time.monotonic() + 5
            while not bool(read_flags[0]):
                assert time.monotonic() < deadline
                time.sleep(0.001)
            time.sleep(0.1)
            assert (int(data[0]), int(timestamps[0])) == (0, 0)
            read_flags[1] = 1

        # C1/C2: both workers update the same node. Rank 1 must see a complete
        # rank-0 pair before writing its later event, even if rank 0 pauses
        # between the data and timestamp stores.
        with sync.transaction(serialize_writes=True):
            if rank == 0:
                assert int(read_flags[1]) == 1
                data[0] = 10
                time.sleep(0.1)
                timestamps[0] = 10
            else:
                assert (int(data[0]), int(timestamps[0])) == (10, 10)
                data[0] = 20
                timestamps[0] = 20
        assert (int(data[0]), int(timestamps[0])) == (20, 20)

        # C3: ownership makes writes disjoint. Rank 0 must not read the next
        # batch until the deliberately slow rank-1 write has finished.
        with sync.transaction(serialize_writes=False):
            if rank == 1:
                time.sleep(0.1)
            data[rank + 1] = rank + 1
            timestamps[rank + 1] = rank + 1
        assert data.tolist() == [20, 1, 2]
        assert timestamps.tolist() == [20, 1, 2]
    finally:
        dist.destroy_process_group()


def test_direct_writeback_sync_orders_raw_and_waits_for_all_owners(tmp_path) -> None:
    data = torch.zeros(3, dtype=torch.int64).share_memory_()
    timestamps = torch.zeros(3, dtype=torch.int64).share_memory_()
    read_flags = torch.zeros(2, dtype=torch.int64).share_memory_()
    mp.spawn(
        _run_two_worker_transactions,
        args=(str(tmp_path / "gloo-rendezvous"), data, timestamps, read_flags),
        nprocs=2,
        join=True,
    )
