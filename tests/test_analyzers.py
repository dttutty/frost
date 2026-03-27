"""Tests for host-side analyzer planner inputs."""

import torch

from frost.datatypes.sampled_layout import SampledLayoutView
from frost.runtime.host.analyzer import (
    APANAnalyzer,
    JODIEAnalyzer,
    NoMemAnalyzer,
    TGNAnalyzer,
)


def _make_layout(
    n_src: int = 4,
    n_neg: int = 1,
    fanout: int = 10,
    sample_4_neg: bool = True,
) -> SampledLayoutView:
    """Create a synthetic host-side sampled-layout view."""
    src = torch.arange(0, n_src)
    dst = torch.arange(n_src, 2 * n_src)

    neg = torch.arange(2 * n_src, 2 * n_src + n_src * n_neg)

    # Roots: [src | dst | neg]
    roots = torch.cat([src, dst, neg])
    hop_width = n_src * (2 + n_neg) if sample_4_neg else n_src * 2
    ngh_count = fanout * hop_width
    ngh = torch.randint(0, 100, (ngh_count,))
    ngh[::3] = -1  # sprinkle padding

    return SampledLayoutView(
        nid=torch.cat([roots, ngh]),
        n_src=n_src,
        n_neg=n_neg,
        fanout=[fanout],
        sample_4_neg=sample_4_neg,
    )


def _single_mem_inputs(analyzer, layout: SampledLayoutView):
    needs, updates = analyzer.get_mem_plan_inputs([layout])
    assert len(needs) == 1
    assert len(updates) == 1
    return needs[0], updates[0]


def _single_mail_inputs(analyzer, layout: SampledLayoutView):
    needs, updates = analyzer.get_mail_plan_inputs([layout])
    assert len(needs) == 1
    assert len(updates) == 1
    return needs[0], updates[0]


def _single_feat_need(analyzer, layout: SampledLayoutView):
    needs = analyzer.get_feat_plan_inputs([layout])
    assert len(needs) == 1
    return needs[0]


class TestNoMemAnalyzer:
    def test_only_nfeat(self):
        analyzer = NoMemAnalyzer(n_neg=1)
        layout = _make_layout()
        mem_needs, mem_updates = analyzer.get_mem_plan_inputs([layout])
        box_needs, box_updates = analyzer.get_mail_plan_inputs([layout])
        nfeat_need = _single_feat_need(analyzer, layout)

        assert len(mem_needs) == 0
        assert len(mem_updates) == 0
        assert len(box_needs) == 0
        assert len(box_updates) == 0
        assert nfeat_need is not None

    def test_nfeat_covers_all_nodes(self):
        analyzer = NoMemAnalyzer(n_neg=1)
        layout = _make_layout()
        nfeat_need = _single_feat_need(analyzer, layout)
        expected = set(layout.valid_nid.tolist())
        assert set(nfeat_need.data.tolist()) == expected
        assert -1 not in nfeat_need.data.tolist()


class TestTGNAnalyzer:
    def test_all_fields_present(self):
        analyzer = TGNAnalyzer(n_neg=1)
        layout = _make_layout()
        mem_need, mem_update = _single_mem_inputs(analyzer, layout)
        box_need, box_update = _single_mail_inputs(analyzer, layout)
        nfeat_need = _single_feat_need(analyzer, layout)

        assert mem_need is not None
        assert mem_update is not None
        assert box_need is not None
        assert box_update is not None
        assert nfeat_need is not None

    def test_mem_update_is_src_dst(self):
        analyzer = TGNAnalyzer(n_neg=1)
        layout = _make_layout(n_src=5)
        _, mem_update = _single_mem_inputs(analyzer, layout)

        expected = set(layout.pos_roots.tolist())
        actual = set(mem_update.data.tolist())
        assert actual == expected

    def test_box_update_equals_mem_update(self):
        """For TGN, box_update should be the same as mem_update (src+dst)."""
        analyzer = TGNAnalyzer(n_neg=1)
        layout = _make_layout()
        _, mem_update = _single_mem_inputs(analyzer, layout)
        _, box_update = _single_mail_inputs(analyzer, layout)
        assert set(box_update.data.tolist()) == set(mem_update.data.tolist())

    def test_padding_filtered_from_full_node_sets(self):
        analyzer = TGNAnalyzer(n_neg=1)
        layout = _make_layout()
        mem_need, _ = _single_mem_inputs(analyzer, layout)
        box_need, _ = _single_mail_inputs(analyzer, layout)
        nfeat_need = _single_feat_need(analyzer, layout)

        assert -1 not in mem_need.data.tolist()
        assert -1 not in nfeat_need.data.tolist()
        assert -1 not in box_need.data.tolist()


class TestJODIEAnalyzer:
    def test_same_as_tgn(self):
        """JODIE analyzer should produce the same access pattern as TGN."""
        jodie = JODIEAnalyzer(n_neg=1)
        tgn = TGNAnalyzer(n_neg=1)
        layout = _make_layout()

        jodie_mem_need, jodie_mem_update = _single_mem_inputs(jodie, layout)
        jodie_box_need, jodie_box_update = _single_mail_inputs(jodie, layout)
        jodie_nfeat_need = _single_feat_need(jodie, layout)
        tgn_mem_need, tgn_mem_update = _single_mem_inputs(tgn, layout)
        _, tgn_box_update = _single_mail_inputs(tgn, layout)

        assert set(jodie_mem_update.data.tolist()) == set(tgn_mem_update.data.tolist())
        assert set(jodie_box_update.data.tolist()) == set(tgn_box_update.data.tolist())
        assert set(jodie_mem_need.data.tolist()) == set(tgn_mem_need.data.tolist())
        assert -1 not in jodie_mem_need.data.tolist()
        assert -1 not in jodie_nfeat_need.data.tolist()
        assert -1 not in jodie_box_need.data.tolist()


class TestAPANAnalyzer:
    def test_box_update_excludes_roots(self):
        """APAN box_update should only contain neighbors, not src/dst roots."""
        n_src = 4
        n_neg = 1
        fanout = 10
        analyzer = APANAnalyzer(n_neg=n_neg)
        src = torch.arange(0, n_src)
        dst = torch.arange(n_src, 2 * n_src)
        neg = torch.arange(2 * n_src, 2 * n_src + n_src * n_neg)
        roots_and_neg = torch.cat([src, dst, neg])
        ngh = torch.arange(100, 100 + fanout * n_src * 2)
        layout = SampledLayoutView(
            nid=torch.cat([roots_and_neg, ngh]),
            n_src=n_src,
            n_neg=n_neg,
            fanout=[fanout],
            sample_4_neg=False,
        )
        box_need, box_update = _single_mail_inputs(analyzer, layout)

        roots = set(layout.pos_roots.tolist())
        box_nodes = set(box_update.data.tolist())
        # box_update contains only valid neighbors extracted from the APAN hop-0 slice.
        # It should not necessarily be a subset of roots - it's the opposite:
        # roots are NOT written by postman, only neighbors are
        assert box_need is not None
        assert box_update is not None
        assert roots.isdisjoint(box_nodes)
        assert -1 not in box_update.data.tolist()

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
        layout = SampledLayoutView(
            nid=torch.cat([roots, ngh]),
            n_src=n_src,
            n_neg=n_neg,
            fanout=[fanout],
            sample_4_neg=False,
        )

        _, box_update = _single_mail_inputs(analyzer, layout)

        # All box_update nodes should come from the neighbor region (>= 100)
        assert box_update.data.min().item() >= 100

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
        layout = SampledLayoutView(
            nid=torch.cat([roots, ngh]),
            n_src=n_src,
            n_neg=n_neg,
            fanout=[fanout],
            sample_4_neg=False,
        )

        _, box_update = _single_mail_inputs(analyzer, layout)
        assert len(box_update) == 0
