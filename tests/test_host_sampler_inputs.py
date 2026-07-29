"""Tests for host-side sampler input preparation."""

import torch

from frost.models.contracts import RootExpansion
from frost.runtime.host._engine import _prepare_sampler_inputs


def test_prepare_sampler_inputs_full_sampling_keeps_all_roots():
    roots = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    ts = torch.tensor([10.0, 20.0], dtype=torch.float32)

    sample_nid, sample_ts = _prepare_sampler_inputs(
        roots=roots,
        ts=ts,
        n_neg=1,
        root_expansion=RootExpansion.ALL_ROOTS,
    )

    assert sample_nid.tolist() == roots.tolist()
    assert sample_ts.tolist() == [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]


def test_prepare_sampler_inputs_positive_roots_drops_negative_roots():
    roots = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    ts = torch.tensor([10.0, 20.0], dtype=torch.float32)

    sample_nid, sample_ts = _prepare_sampler_inputs(
        roots=roots,
        ts=ts,
        n_neg=1,
        root_expansion=RootExpansion.POSITIVE_ROOTS,
    )

    assert sample_nid.tolist() == [0, 1, 2, 3]
    assert sample_ts.tolist() == [10.0, 20.0, 10.0, 20.0]


def test_prepare_sampler_inputs_multi_neg_uses_per_edge_negative_order():
    # roots = [src0, src1 | dst0, dst1 | neg00, neg01, neg10, neg11]
    roots = torch.arange(8)
    ts = torch.tensor([10.0, 20.0])

    _, sample_ts = _prepare_sampler_inputs(
        roots=roots,
        ts=ts,
        n_neg=2,
        root_expansion=RootExpansion.ALL_ROOTS,
    )

    assert sample_ts.tolist() == [10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 20.0, 20.0]
