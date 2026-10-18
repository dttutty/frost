import threading

import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.tasks import DirectWritebackRequest
from frost.runtime.worker._direct_writeback import DirectDynamicStateWriter
from frost.shm.mem_mgr import MemBoxStore


def _make_store() -> MemBoxStore:
    return MemBoxStore(
        mem=torch.zeros((8, 2), dtype=torch.float32),
        mem_ts=torch.zeros(8, dtype=torch.float32),
        box=torch.zeros((8, 1, 3), dtype=torch.float32),
        box_ts=torch.zeros((8, 1), dtype=torch.float32),
        box_pos=torch.zeros(8, dtype=torch.int64),
    )


def test_direct_writeback_preserves_row_order_after_filtering():
    store = _make_store()
    writer = DirectDynamicStateWriter(
        mem_box_store=store,
        h2d_sem=threading.Semaphore(0),
    )
    request = DirectWritebackRequest(
        mini_batch_idx=0,
        mem_entry=MemEntry(
            nid=torch.tensor([1, 2, 1], dtype=torch.long),
            data=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            ts=torch.tensor([5.0, 5.0, 5.0]),
        ),
        box_entry=BoxEntry(
            nid=torch.tensor([1, 2, 1], dtype=torch.long),
            data=torch.tensor(
                [
                    [[10.0, 11.0, 12.0]],
                    [[20.0, 21.0, 22.0]],
                    [[30.0, 31.0, 32.0]],
                ]
            ),
            ts=torch.tensor([[5.0], [5.0], [5.0]]),
        ),
        mem_filter_nids=torch.tensor([1], dtype=torch.long),
        box_filter_nids=torch.tensor([1], dtype=torch.long),
    )

    writer.write(request, release_h2d_credits=False)

    assert torch.equal(store.mem[1], torch.tensor([3.0, 3.0]))
    assert torch.equal(store.mem[2], torch.tensor([0.0, 0.0]))
    assert torch.equal(store.box[1, 0], torch.tensor([30.0, 31.0, 32.0]))
    assert torch.equal(store.box[2, 0], torch.tensor([0.0, 0.0, 0.0]))


def test_direct_writeback_keeps_fresher_existing_rows():
    store = _make_store()
    store.mem[1] = torch.tensor([7.0, 8.0])
    store.mem_ts[1] = torch.tensor(10.0)
    store.box[1, 0] = torch.tensor([9.0, 8.0, 7.0])
    store.box_ts[1, 0] = torch.tensor(10.0)

    writer = DirectDynamicStateWriter(
        mem_box_store=store,
        h2d_sem=threading.Semaphore(0),
    )
    request = DirectWritebackRequest(
        mini_batch_idx=0,
        mem_entry=MemEntry(
            nid=torch.tensor([1], dtype=torch.long),
            data=torch.tensor([[1.0, 1.0]]),
            ts=torch.tensor([5.0]),
        ),
        box_entry=BoxEntry(
            nid=torch.tensor([1], dtype=torch.long),
            data=torch.tensor([[[1.0, 2.0, 3.0]]]),
            ts=torch.tensor([[5.0]]),
        ),
        mem_filter_nids=None,
        box_filter_nids=None,
    )

    writer.write(request, release_h2d_credits=False)

    assert torch.equal(store.mem[1], torch.tensor([7.0, 8.0]))
    assert store.mem_ts[1].item() == 10.0
    assert torch.equal(store.box[1, 0], torch.tensor([9.0, 8.0, 7.0]))
    assert store.box_ts[1, 0].item() == 10.0
