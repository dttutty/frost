"""Tests for the host-side sampled-layout view."""

import pytest
import torch

from frost.datatypes.sampled_layout import SampledLayoutView


def test_full_sampling_layout_slices_roots_and_hop():
    layout = SampledLayoutView(
        nid=torch.tensor(
            [0, 1, 2, 3, 4, 5, 10, 11, -1, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            dtype=torch.long,
        ),
        n_src=2,
        n_neg=1,
        fanout=[2],
        sample_4_neg=True,
    )

    assert layout.n_root == 6
    assert layout.src.tolist() == [0, 1]
    assert layout.pos_dst.tolist() == [2, 3]
    assert layout.pos_neg.tolist() == [4, 5]
    assert layout.pos_roots.tolist() == [0, 1, 2, 3]
    assert layout.valid_hop(0).tolist() == [10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    assert -1 not in layout.valid_nid.tolist()


def test_apan_style_layout_uses_only_positive_roots_for_hop_width():
    layout = SampledLayoutView(
        nid=torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                100,
                101,
                -1,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
            ],
            dtype=torch.long,
        ),
        n_src=2,
        n_neg=1,
        fanout=[3],
        sample_4_neg=False,
    )

    assert layout.n_root == 6
    assert layout.apan_pos_ngh.tolist() == [
        100,
        101,
        -1,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
    ]
    assert layout.filter_valid(layout.apan_pos_ngh).tolist() == [
        100,
        101,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
    ]
    with pytest.raises(AssertionError):
        layout.valid_hop(0)


def test_zero_hop_layout_has_no_hops():
    layout = SampledLayoutView(
        nid=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        n_src=1,
        n_neg=2,
        fanout=None,
        sample_4_neg=False,
    )

    with pytest.raises(IndexError):
        layout.valid_hop(0)
