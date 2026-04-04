"""Tests for TGN and TGAT encoder modules."""

import torch

from frost.datatypes.graph import HierGraph, NeighborOnHop
from frost.models.encoder.tgn import TGN
from frost.models.encoder.tgat import TGAT


class TestTGN:
    def test_forward_output_shape(self):
        ndim, edim, d_mem, d_time, d_hid, heads = 32, 16, 64, 32, 64, 2
        model = TGN(
            ndim=ndim,
            edim=edim,
            d_mem=d_mem,
            d_time=d_time,
            d_hid=d_hid,
            heads=heads,
            dropout=0.1,
        )
        n_src, n_neg, fanout = 4, 1, [10]
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_mem=torch.randn(n_root, d_mem),
            root_nfeat=torch.randn(n_root, ndim),
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 1000, (n_root * S,)),
                    eid=torch.randint(0, 1000, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=torch.randn(n_root * S, ndim),
                    efeat=torch.randn(n_root * S, edim),
                    mem=torch.randn(n_root * S, d_mem),
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        assert out.shape == (n_root, d_hid)

    def test_forward_without_node_features(self):
        ndim, edim, d_mem, d_time, d_hid, heads = 0, 8, 32, 16, 32, 2
        model = TGN(
            ndim=ndim,
            edim=edim,
            d_mem=d_mem,
            d_time=d_time,
            d_hid=d_hid,
            heads=heads,
            dropout=0.1,
        )
        n_src, n_neg, fanout = 3, 1, [5]
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_mem=torch.randn(n_root, d_mem),
            root_nfeat=None,
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 100, (n_root * S,)),
                    eid=torch.randint(0, 100, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=None,
                    efeat=torch.randn(n_root * S, edim),
                    mem=torch.randn(n_root * S, d_mem),
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        assert out.shape == (n_root, d_hid)

    def test_gradient_flow(self):
        ndim, edim, d_mem, d_time, d_hid, heads = 16, 8, 32, 16, 32, 2
        model = TGN(
            ndim=ndim,
            edim=edim,
            d_mem=d_mem,
            d_time=d_time,
            d_hid=d_hid,
            heads=heads,
            dropout=0.0,
        )
        n_src, n_neg, fanout = 2, 1, [4]
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        root_mem = torch.randn(n_root, d_mem, requires_grad=True)
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_mem=root_mem,
            root_nfeat=torch.randn(n_root, ndim),
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 50, (n_root * S,)),
                    eid=torch.randint(0, 50, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=torch.randn(n_root * S, ndim),
                    efeat=torch.randn(n_root * S, edim),
                    mem=torch.randn(n_root * S, d_mem),
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        out.sum().backward()
        assert root_mem.grad is not None


class TestTGAT:
    def test_single_layer_output_shape(self):
        ndim, edim, d_time, d_hid, heads = 32, 16, 32, 64, 2
        fanout = [10]
        model = TGAT(
            ndim=ndim,
            edim=edim,
            d_time=d_time,
            d_hid=d_hid,
            fanout=fanout,
            heads=heads,
            dropout=0.1,
        )
        n_src, n_neg = 4, 1
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_nfeat=torch.randn(n_root, ndim),
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 1000, (n_root * S,)),
                    eid=torch.randint(0, 1000, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=torch.randn(n_root * S, ndim),
                    efeat=torch.randn(n_root * S, edim),
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        assert out.shape == (n_root, d_hid)

    def test_two_layer_output_shape(self):
        ndim, edim, d_time, d_hid, heads = 32, 16, 32, 64, 2
        fanout = [10, 5]
        model = TGAT(
            ndim=ndim,
            edim=edim,
            d_time=d_time,
            d_hid=d_hid,
            fanout=fanout,
            heads=heads,
            dropout=0.1,
        )
        n_src, n_neg = 3, 1
        n_root = n_src * (2 + n_neg)
        S0, S1 = fanout[0], fanout[1]
        n_1hop = n_root * S0
        n_2hop = n_1hop * S1
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_nfeat=torch.randn(n_root, ndim),
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 1000, (n_1hop,)),
                    eid=torch.randint(0, 1000, (n_1hop,)),
                    ts=torch.randn(n_1hop),
                    nfeat=torch.randn(n_1hop, ndim),
                    efeat=torch.randn(n_1hop, edim),
                ),
                NeighborOnHop(
                    nid=torch.randint(0, 1000, (n_2hop,)),
                    eid=torch.randint(0, 1000, (n_2hop,)),
                    ts=torch.randn(n_2hop),
                    nfeat=torch.randn(n_2hop, ndim),
                    efeat=torch.randn(n_2hop, edim),
                ),
            ],
            fanout=fanout,
        )
        out = model(graph)
        assert out.shape == (n_root, d_hid)

    def test_with_memory(self):
        ndim, edim, d_time, d_hid, heads = 32, 16, 32, 64, 2
        d_mem = ndim
        fanout = [8]
        model = TGAT(
            ndim=ndim,
            edim=edim,
            d_time=d_time,
            d_hid=d_hid,
            fanout=fanout,
            heads=heads,
            dropout=0.1,
        )
        n_src, n_neg = 4, 1
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_nfeat=torch.randn(n_root, ndim),
            root_mem=torch.randn(n_root, d_mem),
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 500, (n_root * S,)),
                    eid=torch.randint(0, 500, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=torch.randn(n_root * S, ndim),
                    efeat=torch.randn(n_root * S, edim),
                    mem=torch.randn(n_root * S, d_mem),
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        assert out.shape == (n_root, d_hid)

    def test_gradient_flow(self):
        ndim, edim, d_time, d_hid, heads = 16, 8, 16, 32, 2
        fanout = [5]
        model = TGAT(
            ndim=ndim,
            edim=edim,
            d_time=d_time,
            d_hid=d_hid,
            fanout=fanout,
            heads=heads,
            dropout=0.0,
        )
        n_src, n_neg = 2, 1
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        root_nfeat = torch.randn(n_root, ndim, requires_grad=True)
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_nfeat=root_nfeat,
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 100, (n_root * S,)),
                    eid=torch.randint(0, 100, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=torch.randn(n_root * S, ndim),
                    efeat=torch.randn(n_root * S, edim),
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        out.sum().backward()
        assert root_nfeat.grad is not None

    def test_with_padding(self):
        ndim, edim, d_time, d_hid, heads = 16, 8, 16, 32, 2
        fanout = [5]
        model = TGAT(
            ndim=ndim,
            edim=edim,
            d_time=d_time,
            d_hid=d_hid,
            fanout=fanout,
            heads=heads,
            dropout=0.1,
        )
        n_src, n_neg = 2, 1
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        nbr_nid = torch.randint(0, 100, (n_root * S,))
        nbr_nid[:S] = -1
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_nfeat=torch.randn(n_root, ndim),
            neighbors=[
                NeighborOnHop(
                    nid=nbr_nid,
                    eid=torch.randint(0, 100, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=torch.randn(n_root * S, ndim),
                    efeat=torch.randn(n_root * S, edim),
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        assert out.shape == (n_root, d_hid)
