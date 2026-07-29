"""Tests for FlatGraph, HierGraph, and NeighborOnHop datatypes."""

import torch

from frost.models.contracts import RootExpansion

from frost.datatypes.graph import FlatGraph, HierGraph, NeighborOnHop


class TestNeighborOnHop:
    def test_basic_creation(self):
        nbr = NeighborOnHop(
            nid=torch.tensor([1, 2, 3]),
            eid=torch.tensor([10, 11, 12]),
            ts=torch.tensor([100, 101, 102]),
        )
        assert nbr.nid.tolist() == [1, 2, 3]
        assert nbr.eid.tolist() == [10, 11, 12]
        assert nbr.ts.tolist() == [100, 101, 102]

    def test_optional_fields_default_to_none(self):
        nbr = NeighborOnHop(
            nid=torch.tensor([1, 2]),
            eid=torch.tensor([10, 11]),
            ts=torch.tensor([100, 101]),
        )
        assert nbr.nfeat is None
        assert nbr.efeat is None
        assert nbr.mem is None
        assert nbr.mem_ts is None
        assert nbr.box is None
        assert nbr.box_ts is None


class TestHierGraph:
    def test_basic_creation(self):
        graph = HierGraph(
            root_nid=torch.tensor([1, 2, 3]),
            root_ts=torch.tensor([100, 101, 102]),
        )
        assert graph.root_nid.tolist() == [1, 2, 3]
        assert graph.root_ts.tolist() == [100, 101, 102]
        assert graph.root_nfeat is None
        assert graph.root_mem is None
        assert graph.neighbors is None

    def test_with_neighbors(self):
        graph = HierGraph(
            root_nid=torch.tensor([1, 2]),
            root_ts=torch.tensor([100, 101]),
            neighbors=[
                NeighborOnHop(
                    nid=torch.tensor([10, 20, 30, 40]),
                    eid=torch.tensor([1, 2, 3, 4]),
                    ts=torch.tensor([50, 51, 52, 53]),
                )
            ],
            fanout=[2],
        )
        assert len(graph.neighbors) == 1
        assert graph.neighbors[0].nid.tolist() == [10, 20, 30, 40]


class TestFlatGraph:
    def test_basic_creation(self):
        fg = FlatGraph(
            nid=torch.tensor([1, 2, 3]),
            eid=torch.tensor([10, 11, 12]),
            ts=torch.tensor([100, 101, 102]),
        )
        assert fg.nid.tolist() == [1, 2, 3]
        assert fg.eid.tolist() == [10, 11, 12]
        assert fg.ts.tolist() == [100, 101, 102]

    def test_optional_fields_default_to_none(self):
        fg = FlatGraph(
            nid=torch.tensor([1, 2]),
            eid=torch.tensor([10, 11]),
            ts=torch.tensor([100, 101]),
        )
        assert fg.nfeat is None
        assert fg.efeat is None
        assert fg.mem is None
        assert fg.mem_ts is None
        assert fg.box is None
        assert fg.box_ts is None
        assert fg.detached_mem is None
        assert fg.detached_ts is None
        assert fg.detached_nid is None

    def test_to_device_cpu(self):
        fg = FlatGraph(
            nid=torch.tensor([1, 2, 3]),
            eid=torch.tensor([10, 11, 12]),
            ts=torch.tensor([100, 101, 102]),
            nfeat=torch.tensor([[1.0], [2.0], [3.0]]),
        )
        fg_cpu = fg.to(torch.device("cpu"))
        assert fg_cpu.nid.device.type == "cpu"
        assert fg_cpu.nfeat is not None
        assert fg_cpu.nfeat.device.type == "cpu"

    def test_to_device_with_none_fields(self):
        fg = FlatGraph(
            nid=torch.tensor([1, 2]),
            eid=torch.tensor([10, 11]),
            ts=torch.tensor([100, 101]),
        )
        fg_moved = fg.to(torch.device("cpu"))
        assert fg_moved.nfeat is None
        assert fg_moved.mem is None

    def test_to_device_preserves_detached_fields(self):
        fg = FlatGraph(
            nid=torch.tensor([1, 2]),
            eid=torch.tensor([10, 11]),
            ts=torch.tensor([100, 101]),
            detached_mem=torch.tensor([[1.0, 2.0]]),
            detached_ts=torch.tensor([100]),
            detached_nid=torch.tensor([1]),
        )
        fg_moved = fg.to(torch.device("cpu"))
        assert fg_moved.detached_mem is not None
        assert fg_moved.detached_ts is not None
        assert fg_moved.detached_nid is not None


class TestFlatGraphUnflatten:
    def test_unflatten_no_fanout_0hop(self):
        n_src, n_neg = 4, 1
        n_root = n_src * (2 + n_neg)
        fg = FlatGraph(
            nid=torch.arange(n_root),
            eid=torch.arange(n_root),
            ts=torch.arange(n_root),
        )
        hg = fg.unflatten(n_src=n_src, n_neg=n_neg, fanout=[])
        assert hg.root_nid.tolist() == list(range(n_root))
        assert hg.neighbors is None
        assert hg.fanout == []

    def test_unflatten_1hop_standard(self):
        n_src, n_neg, fanout = 2, 1, [3]
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        total = n_root + n_nbr
        fg = FlatGraph(
            nid=torch.arange(total),
            eid=torch.arange(total),
            ts=torch.arange(total),
        )
        hg = fg.unflatten(n_src=n_src, n_neg=n_neg, fanout=fanout)
        assert hg.root_nid.tolist() == list(range(n_root))
        assert len(hg.neighbors) == 1
        assert hg.neighbors[0].nid.tolist() == list(range(n_root, total))

    def test_unflatten_2hop(self):
        n_src, n_neg, fanout = 2, 1, [2, 3]
        n_root = n_src * (2 + n_neg)
        n_1hop = n_root * fanout[0]
        n_2hop = n_1hop * fanout[1]
        total = n_root + n_1hop + n_2hop
        fg = FlatGraph(
            nid=torch.arange(total),
            eid=torch.arange(total),
            ts=torch.arange(total),
        )
        hg = fg.unflatten(n_src=n_src, n_neg=n_neg, fanout=fanout)
        assert hg.root_nid.tolist() == list(range(n_root))
        assert len(hg.neighbors) == 2
        assert hg.neighbors[0].nid.tolist() == list(range(n_root, n_root + n_1hop))
        assert hg.neighbors[1].nid.tolist() == list(
            range(n_root + n_1hop, n_root + n_1hop + n_2hop)
        )

    def test_unflatten_positive_root_mode_single_hop(self):
        n_src, n_neg, fanout = 4, 1, [5]
        n_root = n_src * (2 + n_neg)
        n_nbr = n_src * 2 * fanout[0]
        total = n_root + n_nbr
        fg = FlatGraph(
            nid=torch.arange(total),
            eid=torch.arange(total),
            ts=torch.arange(total),
        )
        hg = fg.unflatten(
            n_src=n_src,
            n_neg=n_neg,
            fanout=fanout,
            root_expansion=RootExpansion.POSITIVE_ROOTS,
        )
        assert hg.root_nid.tolist() == list(range(n_root))
        assert len(hg.neighbors) == 1
        assert hg.neighbors[0].nid.tolist() == list(range(n_root, total))

    def test_unflatten_positive_root_mode_multi_hop_raises(self):
        n_src, n_neg, fanout = 2, 1, [2, 3]
        n_root = n_src * (2 + n_neg)
        total = n_root + 100
        fg = FlatGraph(
            nid=torch.arange(total),
            eid=torch.arange(total),
            ts=torch.arange(total),
        )
        try:
            fg.unflatten(
                n_src=n_src,
                n_neg=n_neg,
                fanout=fanout,
                root_expansion=RootExpansion.POSITIVE_ROOTS,
            )
        except ValueError as e:
            assert "one hop" in str(e).lower()
        else:
            raise AssertionError(
                "Expected ValueError for multi-hop positive-root expansion"
            )

    def test_unflatten_with_features(self):
        n_src, n_neg, fanout = 2, 1, [2]
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        total = n_root + n_nbr
        fg = FlatGraph(
            nid=torch.arange(total),
            eid=torch.arange(total),
            ts=torch.arange(total),
            nfeat=torch.randn(total, 4),
            mem=torch.randn(total, 8),
        )
        hg = fg.unflatten(n_src=n_src, n_neg=n_neg, fanout=fanout)
        assert hg.n_neg == n_neg
        assert hg.root_nfeat.shape == (n_root, 4)
        assert hg.root_mem.shape == (n_root, 8)
        assert hg.neighbors[0].nfeat.shape == (n_nbr, 4)
        assert hg.neighbors[0].mem.shape == (n_nbr, 8)

    def test_unflatten_none_features_preserved(self):
        n_src, n_neg, fanout = 2, 1, [2]
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        total = n_root + n_nbr
        fg = FlatGraph(
            nid=torch.arange(total),
            eid=torch.arange(total),
            ts=torch.arange(total),
        )
        hg = fg.unflatten(n_src=n_src, n_neg=n_neg, fanout=fanout)
        assert hg.root_nfeat is None
        assert hg.root_mem is None
        assert hg.neighbors[0].nfeat is None
        assert hg.neighbors[0].mem is None
