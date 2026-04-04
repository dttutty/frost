import torch

from frost.datatypes.mem_mail_box.box import BoxEntry
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.runtime.worker.p2p_syncer._writer import BoxMailWriter, SharedMemWriter
from frost.shm.mem_mgr import MemBoxStore


class _ExplodingEntry:
    @property
    def nid(self):
        raise AssertionError("empty writeback should not touch entry.nid")

    @property
    def data(self):
        raise AssertionError("empty writeback should not touch entry.data")

    @property
    def ts(self):
        raise AssertionError("empty writeback should not touch entry.ts")


def test_shared_mem_writer_empty_writeback_does_not_touch_entry():
    writer = SharedMemWriter.__new__(SharedMemWriter)
    SharedMemWriter.write_back(
        writer,
        torch.empty(0, dtype=torch.long),
        _ExplodingEntry(),
    )


def test_box_mail_writer_empty_writeback_does_not_touch_entry():
    writer = BoxMailWriter.__new__(BoxMailWriter)
    BoxMailWriter.write_back(
        writer,
        torch.empty(0, dtype=torch.long),
        _ExplodingEntry(),
    )


def test_shared_mem_writer_skips_stale_timestamp_overwrite():
    writer = SharedMemWriter.__new__(SharedMemWriter)
    writer.shared_data = torch.zeros(4, 2, dtype=torch.float32)
    writer.shared_ts = torch.tensor([0, 5, 0, 0], dtype=torch.int64)
    writer.shared_data[1] = torch.tensor([9.0, 9.0])

    entry = MemEntry(
        nid=torch.tensor([1, 2], dtype=torch.long),
        data=torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
        ts=torch.tensor([3, 7], dtype=torch.int64),
        is_sorted=True,
        is_unique=True,
    )

    SharedMemWriter.write_back(
        writer,
        torch.tensor([1, 2], dtype=torch.long),
        entry,
    )

    assert torch.equal(writer.shared_data[1], torch.tensor([9.0, 9.0]))
    assert writer.shared_ts[1].item() == 5
    assert torch.equal(writer.shared_data[2], torch.tensor([2.0, 2.0]))
    assert writer.shared_ts[2].item() == 7


def test_box_mail_writer_boxsize1_skips_stale_timestamp_overwrite():
    writer = BoxMailWriter.__new__(BoxMailWriter)
    writer.mem_box_store = MemBoxStore(
        mem=torch.zeros(4, 2, dtype=torch.float32),
        mem_ts=torch.zeros(4, dtype=torch.int64),
        box=torch.zeros(4, 1, 2, dtype=torch.float32),
        box_ts=torch.zeros(4, 1, dtype=torch.int64),
        box_pos=torch.zeros(4, dtype=torch.int64),
    )
    writer.mem_box_store.box[1, 0] = torch.tensor([9.0, 9.0])
    writer.mem_box_store.box_ts[1, 0] = 5

    entry = BoxEntry(
        nid=torch.tensor([1, 2], dtype=torch.long),
        data=torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]], dtype=torch.float32),
        ts=torch.tensor([[3], [7]], dtype=torch.int64),
        is_sorted=True,
        is_unique=True,
    )

    BoxMailWriter.write_back(
        writer,
        torch.tensor([1, 2], dtype=torch.long),
        entry,
    )

    assert torch.equal(writer.mem_box_store.box[1, 0], torch.tensor([9.0, 9.0]))
    assert writer.mem_box_store.box_ts[1, 0].item() == 5
    assert torch.equal(writer.mem_box_store.box[2, 0], torch.tensor([2.0, 2.0]))
    assert writer.mem_box_store.box_ts[2, 0].item() == 7
