import threading
from contextlib import contextmanager

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


def _copy_store(store: MemBoxStore) -> MemBoxStore:
    return MemBoxStore(
        mem=store.mem.clone(),
        mem_ts=store.mem_ts.clone(),
        box=store.box.clone(),
        box_ts=store.box_ts.clone(),
        box_pos=store.box_pos.clone(),
    )


def _reference_write(store: MemBoxStore, request: DirectWritebackRequest) -> None:
    """Original sequential CPU writer, including conversion to store dtypes."""
    if request.mem_entry is not None:
        entry = request.mem_entry
        nids = entry.nid.cpu().long()
        data = entry.data.cpu().to(store.mem.dtype)
        ts = entry.ts.cpu().to(store.mem_ts.dtype)
        allowed = (
            None
            if request.mem_filter_nids is None
            else set(request.mem_filter_nids.cpu().tolist())
        )
        for row, nid in enumerate(nids.tolist()):
            if allowed is not None and nid not in allowed:
                continue
            if ts[row] >= store.mem_ts[nid]:
                store.mem[nid] = data[row]
                store.mem_ts[nid] = ts[row]

    if request.box_entry is not None:
        entry = request.box_entry
        nids = entry.nid.cpu().long()
        data = entry.data.cpu().to(store.box.dtype)
        ts = entry.ts.cpu().to(store.box_ts.dtype)
        allowed = (
            None
            if request.box_filter_nids is None
            else set(request.box_filter_nids.cpu().tolist())
        )
        for row, nid in enumerate(nids.tolist()):
            if allowed is not None and nid not in allowed:
                continue
            if ts[row, 0] >= store.box_ts[nid, 0]:
                store.box[nid, 0] = data[row, 0]
                store.box_ts[nid, 0] = ts[row, 0]
                store.box_pos[nid] = 0


def test_direct_writeback_matches_sequential_writer_with_duplicates_and_nan():
    generator = torch.Generator().manual_seed(428)
    for _ in range(40):
        store = _make_store()
        store.mem = torch.randn((8, 2), generator=generator)
        store.mem_ts = torch.randint(-3, 8, (8,), generator=generator).float()
        store.box = torch.randn((8, 1, 3), generator=generator)
        store.box_ts = torch.randint(-3, 8, (8, 1), generator=generator).float()
        store.box_pos = torch.randint(0, 4, (8,), generator=generator)
        expected = _copy_store(store)

        mem_count = int(torch.randint(0, 25, (1,), generator=generator))
        box_count = int(torch.randint(0, 25, (1,), generator=generator))
        mem_ts = torch.randint(-3, 8, (mem_count,), generator=generator).double()
        box_ts = torch.randint(-3, 8, (box_count, 1), generator=generator).double()
        if mem_count:
            mem_ts[::7] = float("nan")
        if box_count:
            box_ts[::5] = float("nan")
        request = DirectWritebackRequest(
            mini_batch_idx=0,
            mem_entry=MemEntry(
                nid=torch.randint(0, 8, (mem_count,), generator=generator),
                data=torch.randn((mem_count, 2), generator=generator),
                ts=mem_ts,
            ),
            box_entry=BoxEntry(
                nid=torch.randint(0, 8, (box_count,), generator=generator),
                data=torch.randn((box_count, 1, 3), generator=generator),
                ts=box_ts,
            ),
            mem_filter_nids=torch.randperm(8, generator=generator)[:4],
            box_filter_nids=torch.randperm(8, generator=generator)[:5],
        )

        _reference_write(expected, request)
        DirectDynamicStateWriter(
            mem_box_store=store, h2d_sem=threading.Semaphore(0)
        ).write(request, release_h2d_credits=False)

        for name in ("mem", "mem_ts", "box", "box_ts", "box_pos"):
            assert torch.equal(getattr(store, name), getattr(expected, name)), name


def test_direct_writeback_ties_after_timestamp_dtype_conversion():
    store = _make_store()
    times = torch.tensor([1 + 2**-25, 1 + 2**-26], dtype=torch.float64)
    request = DirectWritebackRequest(
        mini_batch_idx=0,
        mem_entry=MemEntry(
            nid=torch.tensor([1, 1]),
            data=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            ts=times,
        ),
        box_entry=BoxEntry(
            nid=torch.tensor([1, 1]),
            data=torch.tensor([[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]]),
            ts=times[:, None],
        ),
        mem_filter_nids=None,
        box_filter_nids=None,
    )

    DirectDynamicStateWriter(mem_box_store=store, h2d_sem=threading.Semaphore(0)).write(
        request, release_h2d_credits=False
    )

    assert torch.equal(store.mem[1], torch.tensor([2.0, 2.0]))
    assert torch.equal(store.box[1, 0], torch.tensor([2.0, 2.0, 2.0]))


def test_direct_writeback_prepares_before_transaction_and_releases_after(monkeypatch):
    events = []

    class Synchronizer:
        @contextmanager
        def transaction(self, *, serialize_writes: bool):
            assert serialize_writes
            events.append("enter")
            yield
            events.append("exit")

    class Gate:
        def writer_done(self, *, batch_idx: int, kind: str):
            assert batch_idx == 4
            events.append(f"credit_{kind}")

    writer = DirectDynamicStateWriter(
        mem_box_store=_make_store(),
        h2d_sem=Gate(),
        synchronizer=Synchronizer(),
    )
    for name in ("_prepare_mem", "_prepare_box", "_apply_mem", "_apply_box"):
        original = getattr(writer, name)

        def record(*args, _name=name, _original=original):
            events.append(_name)
            return _original(*args)

        monkeypatch.setattr(writer, name, record)

    request = DirectWritebackRequest(
        mini_batch_idx=4,
        mem_entry=MemEntry(
            nid=torch.tensor([1]),
            data=torch.tensor([[1.0, 2.0]]),
            ts=torch.tensor([1.0]),
        ),
        box_entry=BoxEntry(
            nid=torch.tensor([1]),
            data=torch.tensor([[[1.0, 2.0, 3.0]]]),
            ts=torch.tensor([[1.0]]),
        ),
        mem_filter_nids=None,
        box_filter_nids=None,
    )

    writer.write(request, release_h2d_credits=True)

    assert events == [
        "_prepare_mem",
        "_prepare_box",
        "enter",
        "_apply_mem",
        "_apply_box",
        "exit",
        "credit_mem",
        "credit_box",
    ]
