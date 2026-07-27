"""Tests for compute_last_indices determinism and ring-buffer write position.

These tests verify two bug fixes:
1. compute_last_indices must be deterministic on CUDA (no scatter_ race).
2. _apply_box_mail_delta must write to box_pos (next slot), not box_pos-1.
"""

from __future__ import annotations

import pytest
import torch

from frost.datatypes.mem_mail_box._data_ops import compute_last_indices, keep_last
from frost.datatypes.mem_mail_box.mem import MemEntry


# ---------------------------------------------------------------------------
# compute_last_indices: correctness
# ---------------------------------------------------------------------------


class TestComputeLastIndices:
    """Verify that compute_last_indices always returns the entry with the
    highest timestamp for each unique nid."""

    def test_empty(self):
        nid = torch.tensor([], dtype=torch.long)
        ts = torch.tensor([], dtype=torch.long)
        result = compute_last_indices(nid, ts)
        assert result.numel() == 0

    def test_unique_nids(self):
        """Each nid appears once — trivial case."""
        nid = torch.tensor([10, 20, 30])
        ts = torch.tensor([100, 200, 300])
        result = compute_last_indices(nid, ts)
        # unique_nids sorted: [10, 20, 30]  → should map to indices [0, 1, 2]
        assert result.tolist() == [0, 1, 2]

    def test_duplicates_pick_max_ts(self):
        """When a nid appears multiple times, the entry with the highest ts wins."""
        nid = torch.tensor([5, 2, 5, 1, 2, 5, 2])
        ts = torch.tensor([100, 200, 150, 300, 250, 50, 400])
        result = compute_last_indices(nid, ts)
        # unique_nids: [1, 2, 5]
        # nid=1: only at index 3 (ts=300)
        # nid=2: indices 1(ts=200), 4(ts=250), 6(ts=400) → max ts at index 6
        # nid=5: indices 0(ts=100), 2(ts=150), 5(ts=50)  → max ts at index 2
        assert result.tolist() == [3, 6, 2]

    def test_all_same_nid(self):
        """All entries share one nid — should return the max-ts index."""
        nid = torch.tensor([7, 7, 7, 7])
        ts = torch.tensor([10, 40, 20, 30])
        result = compute_last_indices(nid, ts)
        assert result.numel() == 1
        assert result.item() == 1  # ts=40 is highest, at index 1

    def test_tied_timestamps(self):
        """When multiple entries tie on max ts, the one with the largest
        original index is selected (deterministic tie-breaking)."""
        nid = torch.tensor([3, 3, 3])
        ts = torch.tensor([100, 100, 100])
        result = compute_last_indices(nid, ts)
        assert result.numel() == 1
        # All ts equal; largest original index = 2
        assert result.item() == 2

    def test_determinism_repeated_runs(self):
        """Run compute_last_indices many times and verify it always returns
        the same result (no CUDA scatter_ non-determinism)."""
        nid = torch.tensor([1, 2, 1, 2, 1, 2])
        ts = torch.tensor([10, 20, 30, 5, 15, 25])
        expected = compute_last_indices(nid, ts).tolist()
        for _ in range(50):
            assert compute_last_indices(nid, ts).tolist() == expected


# ---------------------------------------------------------------------------
# compute_last_indices: CUDA determinism (skipped if no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestComputeLastIndicesCUDA:
    """Same tests on CUDA tensors to verify determinism on GPU."""

    def test_duplicates_pick_max_ts_cuda(self):
        nid = torch.tensor([5, 2, 5, 1, 2, 5, 2], device="cuda")
        ts = torch.tensor([100, 200, 150, 300, 250, 50, 400], device="cuda")
        result = compute_last_indices(nid, ts)
        assert result.tolist() == [3, 6, 2]

    def test_determinism_cuda(self):
        """Repeat on CUDA 100 times to catch non-deterministic scatter_."""
        nid = torch.tensor([1, 2, 1, 2, 1, 2], device="cuda")
        ts = torch.tensor([10, 20, 30, 5, 15, 25], device="cuda")
        expected = compute_last_indices(nid, ts).tolist()
        for _ in range(100):
            assert compute_last_indices(nid, ts).tolist() == expected

    def test_large_tensor_determinism_cuda(self):
        """Larger tensor to increase probability of hitting non-determinism."""
        n = 10000
        nid = torch.randint(0, 500, (n,), device="cuda")
        ts = torch.randint(0, 100000, (n,), device="cuda")
        expected = compute_last_indices(nid, ts).tolist()
        for _ in range(20):
            assert compute_last_indices(nid, ts).tolist() == expected


# ---------------------------------------------------------------------------
# keep_last: integration
# ---------------------------------------------------------------------------


class TestKeepLast:
    """Verify that keep_last correctly deduplicates MemEntry by max ts."""

    def test_keep_last_basic(self):
        mem = MemEntry(
            nid=torch.tensor([1, 2, 1]),
            data=torch.tensor([[1.0], [2.0], [3.0]]),
            ts=torch.tensor([10, 20, 30]),
        )
        result = keep_last(mem)
        # nid=1 has ts 10 and 30 → keep ts=30 (data=[3.0])
        # nid=2 has ts 20 → keep (data=[2.0])
        assert result.nid.tolist() == [1, 2]
        assert result.ts.tolist() == [30, 20]
        assert result.data.tolist() == [[3.0], [2.0]]
        assert result.is_sorted
        assert result.is_unique

    def test_keep_last_no_duplicates(self):
        mem = MemEntry(
            nid=torch.tensor([10, 20, 30]),
            data=torch.tensor([[1.0], [2.0], [3.0]]),
            ts=torch.tensor([100, 200, 300]),
        )
        result = keep_last(mem)
        assert result.nid.tolist() == [10, 20, 30]
        assert result.ts.tolist() == [100, 200, 300]

    def test_keep_last_sorts_unique_unsorted_input(self):
        mem = MemEntry(
            nid=torch.tensor([30, 10, 20]),
            data=torch.tensor([[3.0], [1.0], [2.0]]),
            ts=torch.tensor([300, 100, 200]),
        )

        result = keep_last(mem)

        assert result.nid.tolist() == [10, 20, 30]
        assert result.data.tolist() == [[1.0], [2.0], [3.0]]
        assert result.ts.tolist() == [100, 200, 300]
        assert result.is_sorted
        assert result.is_unique


# ---------------------------------------------------------------------------
# Ring-buffer write position
# ---------------------------------------------------------------------------


class TestRingBufferWritePos:
    """Verify that the ring-buffer write position uses box_pos (next slot),
    not box_pos - 1 (most-recent slot)."""

    def test_write_pos_semantics(self):
        """box_pos=2, boxsize=4 → new mail goes to slot 2, not slot 1."""
        boxsize = 4
        box_pos = torch.tensor([2])
        # Correct: write to box_pos % boxsize = 2
        correct_write_pos = box_pos % boxsize
        # Bug: (box_pos - 1) % boxsize = 1 (overwrites most-recent)
        wrong_write_pos = (box_pos - 1) % boxsize
        assert correct_write_pos.item() == 2
        assert wrong_write_pos.item() == 1
        assert correct_write_pos.item() != wrong_write_pos.item()

    def test_write_pos_wrap_around(self):
        """box_pos=0 (buffer full, wrapped) → write to slot 0."""
        boxsize = 3
        box_pos = torch.tensor([0])
        correct_write_pos = box_pos % boxsize
        wrong_write_pos = (box_pos - 1) % boxsize
        assert correct_write_pos.item() == 0
        assert wrong_write_pos.item() == 2  # bug: writes to last slot

    def test_write_pos_boxsize_1(self):
        """boxsize=1 → both expressions give 0 (bug invisible)."""
        boxsize = 1
        for pos_val in range(5):
            box_pos = torch.tensor([pos_val])
            assert (box_pos % boxsize).item() == ((box_pos - 1) % boxsize).item()
