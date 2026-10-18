"""Tests for TensorSet set-like operations on 1D tensors."""

import torch

from frost.utils.tensor_set import TensorSet, isin_sorted


def _ts(vals: list[int]) -> TensorSet:
    return TensorSet(torch.tensor(vals, dtype=torch.long))


def test_sorted_membership_matches_torch_isin():
    query = torch.tensor([9, -1, 3, 5, 3, 12, 1], dtype=torch.long)
    candidates = torch.tensor([1, 3, 9], dtype=torch.long)

    actual = isin_sorted(query, candidates)

    assert torch.equal(actual, torch.isin(query, candidates))


def test_sorted_membership_handles_empty_candidates():
    query = torch.tensor([1, 2, 3], dtype=torch.long)

    actual = isin_sorted(query, torch.empty(0, dtype=torch.long))

    assert torch.equal(actual, torch.zeros_like(query, dtype=torch.bool))


class TestTensorSetBasic:
    def test_dedup_and_sort(self):
        ts = _ts([3, 1, 2, 2, 1])
        assert ts.data.tolist() == [1, 2, 3]

    def test_empty(self):
        ts = TensorSet.empty_like(torch.tensor([1], dtype=torch.long))
        assert len(ts) == 0
        assert ts.numel() == 0

    def test_len_and_numel(self):
        ts = _ts([10, 20, 30])
        assert len(ts) == 3
        assert ts.numel() == 3

    def test_getitem(self):
        ts = _ts([5, 10, 15])
        assert ts[1].item() == 10

    def test_2d_input_flattened(self):
        t = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        ts = TensorSet(t)
        assert ts.data.tolist() == [1, 2, 3, 4]


class TestTensorSetOps:
    def test_intersection(self):
        a = _ts([1, 2, 3, 4, 5])
        b = _ts([4, 5, 6, 7])
        result = a & b
        assert result.data.tolist() == [4, 5]

    def test_intersection_empty(self):
        a = _ts([1, 2])
        b = _ts([3, 4])
        result = a & b
        assert len(result) == 0

    def test_union(self):
        a = _ts([1, 2, 3])
        b = _ts([3, 4, 5])
        result = a | b
        assert result.data.tolist() == [1, 2, 3, 4, 5]

    def test_union_via_add(self):
        a = _ts([1, 2])
        b = _ts([2, 3])
        result = a + b
        assert result.data.tolist() == [1, 2, 3]

    def test_difference(self):
        a = _ts([1, 2, 3, 4, 5])
        b = _ts([4, 5, 6])
        result = a - b
        assert result.data.tolist() == [1, 2, 3]

    def test_symmetric_difference(self):
        a = _ts([1, 2, 3])
        b = _ts([2, 3, 4])
        result = a ^ b
        assert result.data.tolist() == [1, 4]

    def test_mixed_tensor_operand(self):
        a = _ts([1, 2, 3, 4])
        raw = torch.tensor([2, 4], dtype=torch.long)
        result = a - raw
        assert result.data.tolist() == [1, 3]

    def test_dtype_coercion(self):
        a = _ts([1, 2, 3])
        b = TensorSet(torch.tensor([2, 3], dtype=torch.int32))
        result = a & b
        assert result.data.tolist() == [2, 3]
