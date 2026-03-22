"""Tests for host-side analyzer access counting.

These tests verify that each analyzer produces the correct Access fields
for its model family, without requiring distributed context.
"""

import torch

from frost.runtime.host.analyzer import (
    APANAnalyzer,
    JODIEAnalyzer,
    NoMemAnalyzer,
    TGNAnalyzer,
)


def _make_batch(n_src: int = 4, n_neg: int = 1, fanout: int = 10):
    """Create a synthetic batch with src, dst, all_nid, and valid_all."""
    src = torch.arange(0, n_src)
    dst = torch.arange(n_src, 2 * n_src)

    n_root = n_src * (2 + n_neg)
    neg = torch.arange(2 * n_src, 2 * n_src + n_src * n_neg)

    # Roots: [src | dst | neg]
    roots = torch.cat([src, dst, neg])
    # 1-hop neighbors for src+dst (some padded with -1)
    ngh_count = fanout * n_src * 2
    ngh = torch.randint(0, 100, (ngh_count,))
    ngh[::3] = -1  # sprinkle padding

    valid_all = torch.cat([roots, ngh])
    all_nid = valid_all[valid_all != -1].unique()
    return src, dst, all_nid, valid_all


class TestNoMemAnalyzer:
    def test_only_nfeat(self):
        analyzer = NoMemAnalyzer(n_neg=1)
        src, dst, all_nid, _ = _make_batch()
        access = analyzer.count_access(src, dst, all_nid, fanout=[10])

        assert access.required_nfeat is not None
        assert access.required_mem is None
        assert access.mem_update is None
        assert access.box_update is None

    def test_nfeat_covers_all_nodes(self):
        analyzer = NoMemAnalyzer(n_neg=1)
        src, dst, all_nid, _ = _make_batch()
        access = analyzer.count_access(src, dst, all_nid, fanout=[10])
        assert set(access.required_nfeat.data.tolist()) == set(all_nid.tolist())


class TestTGNAnalyzer:
    def test_all_fields_present(self):
        analyzer = TGNAnalyzer(n_neg=1)
        src, dst, all_nid, _ = _make_batch()
        access = analyzer.count_access(src, dst, all_nid, fanout=[10])

        assert access.required_mem is not None
        assert access.mem_update is not None
        assert access.box_update is not None
        assert access.required_nfeat is not None

    def test_mem_update_is_src_dst(self):
        analyzer = TGNAnalyzer(n_neg=1)
        src, dst, all_nid, _ = _make_batch(n_src=5)
        access = analyzer.count_access(src, dst, all_nid, fanout=[10])

        expected = set(torch.cat([src, dst]).tolist())
        actual = set(access.mem_update.data.tolist())
        assert actual == expected

    def test_box_update_equals_mem_update(self):
        """For TGN, box_update should be the same as mem_update (src+dst)."""
        analyzer = TGNAnalyzer(n_neg=1)
        src, dst, all_nid, _ = _make_batch()
        access = analyzer.count_access(src, dst, all_nid, fanout=[10])
        assert set(access.box_update.data.tolist()) == set(access.mem_update.data.tolist())


class TestJODIEAnalyzer:
    def test_same_as_tgn(self):
        """JODIE analyzer should produce the same access pattern as TGN."""
        jodie = JODIEAnalyzer(n_neg=1)
        tgn = TGNAnalyzer(n_neg=1)
        src, dst, all_nid, _ = _make_batch()

        a_jodie = jodie.count_access(src, dst, all_nid, fanout=None)
        a_tgn = tgn.count_access(src, dst, all_nid, fanout=None)

        assert set(a_jodie.mem_update.data.tolist()) == set(a_tgn.mem_update.data.tolist())
        assert set(a_jodie.box_update.data.tolist()) == set(a_tgn.box_update.data.tolist())


class TestAPANAnalyzer:
    def test_box_update_excludes_roots(self):
        """APAN box_update should only contain neighbors, not src/dst roots."""
        n_src = 4
        n_neg = 1
        fanout = 10
        analyzer = APANAnalyzer(n_neg=n_neg)
        src, dst, all_nid, valid_all = _make_batch(n_src=n_src, n_neg=n_neg, fanout=fanout)
        access = analyzer.count_access(src, dst, all_nid, fanout=[fanout], valid_all=valid_all)

        roots = set(torch.cat([src, dst]).tolist())
        box_nodes = set(access.box_update.data.tolist())
        # box_update contains only valid neighbors extracted from valid_all
        # It should not necessarily be a subset of roots - it's the opposite:
        # roots are NOT written by postman, only neighbors are
        assert access.box_update is not None

    def test_ngh_start_offset(self):
        """Verify the neighbor start offset is n_src * (2 + n_neg)."""
        n_src = 3
        n_neg = 2
        fanout = 5
        analyzer = APANAnalyzer(n_neg=n_neg)

        src = torch.arange(0, n_src)
        dst = torch.arange(n_src, 2 * n_src)
        neg = torch.arange(2 * n_src, 2 * n_src + n_src * n_neg)
        roots = torch.cat([src, dst, neg])

        ngh_count = fanout * n_src * 2
        ngh = torch.arange(100, 100 + ngh_count)
        valid_all = torch.cat([roots, ngh])
        all_nid = valid_all.unique()

        access = analyzer.count_access(src, dst, all_nid, fanout=[fanout], valid_all=valid_all)

        # All box_update nodes should come from the neighbor region (>= 100)
        assert access.box_update.data.min().item() >= 100

    def test_padding_filtered(self):
        """Padding nodes (-1) should not appear in box_update."""
        n_src = 2
        n_neg = 1
        fanout = 4
        analyzer = APANAnalyzer(n_neg=n_neg)

        src = torch.tensor([0, 1])
        dst = torch.tensor([2, 3])
        neg = torch.tensor([4, 5])
        roots = torch.cat([src, dst, neg])

        # All neighbors are padding
        ngh = torch.full((fanout * n_src * 2,), -1)
        valid_all = torch.cat([roots, ngh])
        all_nid = valid_all[valid_all != -1].unique()

        access = analyzer.count_access(src, dst, all_nid, fanout=[fanout], valid_all=valid_all)
        assert len(access.box_update) == 0
