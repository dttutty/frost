"""Tests for RingBuf shared-memory circular buffer."""

from types import SimpleNamespace

import torch

from frost.models.contracts import RootExpansion
from frost.shm import ring_buf
from frost.shm.ring_buf import (
    cal_max_buf_size,
    cal_max_buf_size_for_layouts,
)


class TestCalMaxBufSize:
    def test_no_fanout(self):
        n_src, n_neg = 100, 1
        size = cal_max_buf_size(n_src, fanout=None, n_neg=n_neg)
        expected = n_src * (2 + n_neg)
        assert size == expected

    def test_single_fanout(self):
        n_src, n_neg, fanout = 100, 1, [10]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg)
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        assert size == n_root + n_nbr

    def test_multi_hop_fanout(self):
        n_src, n_neg, fanout = 50, 2, [10, 5]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg)
        n_root = n_src * (2 + n_neg)
        n_1hop = n_root * fanout[0]
        n_2hop = n_1hop * fanout[1]
        assert size == n_root + n_1hop + n_2hop

    def test_positive_root_expansion(self):
        n_src, n_neg, fanout = 100, 1, [10]
        size = cal_max_buf_size(
            n_src,
            fanout=fanout,
            n_neg=n_neg,
            root_expansion=RootExpansion.POSITIVE_ROOTS,
        )
        n_root = n_src * (2 + n_neg)
        expected = n_root + n_src * 2 * fanout[0]
        assert size == expected

    def test_zero_neg(self):
        n_src, n_neg, fanout = 100, 0, [5]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg)
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        assert size == n_root + n_nbr

    def test_large_fanout(self):
        n_src, n_neg, fanout = 10, 1, [100]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg)
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        assert size == n_root + n_nbr

    def test_empty_fanout_list(self):
        n_src, n_neg = 100, 1
        size = cal_max_buf_size(n_src, fanout=[], n_neg=n_neg)
        expected = n_src * (2 + n_neg)
        assert size == expected

    def test_multiple_layouts_use_largest_actual_slot(self):
        size = cal_max_buf_size_for_layouts([(2_000, 1), (4, 999)], fanout=[10])

        assert size == cal_max_buf_size(2_000, fanout=[10], n_neg=1)


def test_ring_slots_are_bound_per_gpu(monkeypatch) -> None:
    created: list[tuple[str, tuple[int, ...], int | None]] = []

    def fake_create(
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        numa_node: int | None = None,
    ) -> torch.Tensor:
        created.append((name, shape, numa_node))
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(ring_buf, "create_shared_mem_array", fake_create)
    monkeypatch.setattr(ring_buf.dist, "barrier", lambda: None)
    env = SimpleNamespace(n_gpu=2, is_host=True, local_rank=2)

    buffer = ring_buf.RingBuf(
        env=env,
        run_id="test",
        max_batch_size=2,
        fanout=[1],
        n_neg=1,
        root_expansion=RootExpansion.ALL_ROOTS,
        max_size=3,
        ndtype=torch.int64,
        edtype=torch.int64,
        tsdtype=torch.float64,
        slot_capacity=7,
        gpu_numa_nodes=[0, 1],
    )

    data_segments = [item for item in created if "RING_BUF" in item[0]]
    assert len(data_segments) == 18
    assert all(shape == (7,) for _, shape, _ in data_segments)
    assert [node for _, _, node in data_segments] == [0] * 9 + [1] * 9
    assert buffer.buffers[0][2]["NID"].shape == (7,)


def test_ring_rejects_wrong_numa_node_count() -> None:
    env = SimpleNamespace(n_gpu=2, is_host=True, local_rank=2)

    try:
        ring_buf.RingBuf(
            env=env,
            run_id="test",
            max_batch_size=2,
            fanout=[1],
            n_neg=1,
            root_expansion=RootExpansion.ALL_ROOTS,
            max_size=3,
            ndtype=torch.int64,
            edtype=torch.int64,
            tsdtype=torch.float64,
            gpu_numa_nodes=[0],
        )
    except ValueError as exc:
        assert "one entry per GPU" in str(exc)
    else:
        raise AssertionError("RingBuf accepted an incomplete NUMA node list")
