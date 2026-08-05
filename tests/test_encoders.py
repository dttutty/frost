"""Tests for TGN and TGAT encoder modules."""

import pytest
import torch

from frost.datatypes.graph import HierGraph, NeighborOnHop
from frost.models.encoder.tgn import TGN
from frost.models.encoder.tgat import TGAT


class TestTGN:
    def test_rejects_unknown_attention_layout(self):
        with pytest.raises(ValueError, match="attention_layout"):
            TGN(
                ndim=0,
                edim=0,
                d_mem=16,
                d_time=8,
                d_hid=16,
                heads=2,
                dropout=0.0,
                attention_layout="unknown",
            )

    def test_rejects_non_positive_auto_pack_threshold(self):
        with pytest.raises(ValueError, match="auto_pack_min_projected_elements"):
            TGN(
                ndim=0,
                edim=0,
                d_mem=16,
                d_time=8,
                d_hid=16,
                heads=2,
                dropout=0.0,
                attention_layout="auto",
                auto_pack_min_projected_elements=0,
            )

    def test_auto_attention_layout_uses_projected_work_threshold(self):
        model = TGN(
            ndim=0,
            edim=8,
            d_mem=16,
            d_time=8,
            d_hid=16,
            heads=2,
            dropout=0.0,
            attention_layout="auto",
            auto_pack_min_projected_elements=320,
        )

        # Projected width is 16 memory + 8 edge + 8 time = 32.
        assert not model._use_packed_attention(9)
        assert model._use_packed_attention(10)

    @pytest.mark.parametrize(
        ("layout", "expected"), (("dense", False), ("packed", True))
    )
    def test_explicit_attention_layout_bypasses_auto_threshold(
        self, layout: str, expected: bool
    ):
        model = TGN(
            ndim=0,
            edim=0,
            d_mem=16,
            d_time=8,
            d_hid=16,
            heads=2,
            dropout=0.0,
            attention_layout=layout,
        )

        assert model._use_packed_attention(1) is expected

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
        assert out.embeddings.shape == (n_root, d_hid)

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
        assert out.embeddings.shape == (n_root, d_hid)

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
        out.embeddings.sum().backward()
        assert root_mem.grad is not None

    def test_feature_projection_skips_padded_neighbors(self):
        ndim, d_mem = 8, 16
        model = TGN(
            ndim=ndim,
            edim=0,
            d_mem=d_mem,
            d_time=8,
            d_hid=d_mem,
            heads=2,
            dropout=0.0,
        )
        assert model.n_feat_mapper is not None
        projected_rows: list[int] = []
        handle = model.n_feat_mapper.register_forward_hook(
            lambda _module, inputs, _output: projected_rows.append(inputs[0].shape[0])
        )
        n_root, fanout = 3, [4]
        neighbor_nid = torch.tensor([1, -1, 2, -1] * n_root)
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_mem=torch.randn(n_root, d_mem),
            root_nfeat=torch.randn(n_root, ndim),
            neighbors=[
                NeighborOnHop(
                    nid=neighbor_nid,
                    eid=torch.full_like(neighbor_nid, -1),
                    ts=torch.zeros(n_root * fanout[0]),
                    nfeat=torch.randn(n_root * fanout[0], ndim),
                    efeat=None,
                    mem=torch.randn(n_root * fanout[0], d_mem),
                )
            ],
            fanout=fanout,
        )

        out = model(graph)
        handle.remove()

        assert out.embeddings.shape == (n_root, d_mem)
        assert projected_rows == [n_root, int((neighbor_nid != -1).sum())]
        dense_target = graph.root_mem + model.n_feat_mapper(graph.root_nfeat)
        neighbor = graph.neighbors[0]
        assert neighbor.nfeat is not None
        assert neighbor.mem is not None
        dense_neighbor = neighbor.mem + model.n_feat_mapper(neighbor.nfeat)
        dense_reference = model.encoder(
            target_feat=dense_target,
            neighbor_feat=dense_neighbor.view(n_root, fanout[0], -1),
            edge_feat=None,
            dt=(graph.root_ts.unsqueeze(1) - neighbor.ts.view(n_root, fanout[0])),
            mask=neighbor_nid.view(n_root, fanout[0]) != -1,
        )
        torch.testing.assert_close(out.embeddings, dense_reference)


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
        assert out.embeddings.shape == (n_root, d_hid)

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
        layer_call_counts = [0, 0]

        def count_layer_call(layer_index):
            def hook(*_):
                layer_call_counts[layer_index] += 1

            return hook

        hooks = [
            layer.register_forward_hook(count_layer_call(layer_index))
            for layer_index, layer in enumerate(model.layers)
        ]
        out = model(graph)
        for hook in hooks:
            hook.remove()

        assert out.embeddings.shape == (n_root, d_hid)
        assert layer_call_counts == [2, 1]

    def test_two_layer_parameter_count_matches_tgl_parity(self):
        model = TGAT(
            ndim=1,
            edim=172,
            d_time=128,
            d_hid=128,
            fanout=[10, 10],
            heads=2,
            dropout=0.1,
        )

        assert sum(parameter.numel() for parameter in model.parameters()) == 287232

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
        assert out.embeddings.shape == (n_root, d_hid)

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
        out.embeddings.sum().backward()
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
        assert out.embeddings.shape == (n_root, d_hid)

    def test_synthesizes_dummy_zero_edge_features_when_edim_is_zero(self):
        ndim, d_time, d_hid, heads = 16, 16, 32, 2
        fanout = [5]
        model = TGAT(
            ndim=ndim,
            edim=0,
            d_time=d_time,
            d_hid=d_hid,
            fanout=fanout,
            heads=heads,
            dropout=0.1,
            dummy_edge_dim=4,
        )
        n_src, n_neg = 2, 1
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_nfeat=torch.randn(n_root, ndim),
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 100, (n_root * S,)),
                    eid=torch.randint(0, 100, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=torch.randn(n_root * S, ndim),
                    efeat=None,
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        assert out.embeddings.shape == (n_root, d_hid)

    def test_synthesizes_dummy_zero_node_and_edge_features_when_dims_are_zero(self):
        d_time, d_hid, heads = 16, 32, 2
        fanout = [5]
        model = TGAT(
            ndim=0,
            edim=0,
            d_time=d_time,
            d_hid=d_hid,
            fanout=fanout,
            heads=heads,
            dropout=0.1,
            dummy_node_dim=4,
            dummy_edge_dim=6,
        )
        n_src, n_neg = 2, 1
        n_root = n_src * (2 + n_neg)
        S = fanout[0]
        graph = HierGraph(
            root_nid=torch.arange(n_root),
            root_ts=torch.arange(n_root, dtype=torch.float32),
            root_nfeat=None,
            neighbors=[
                NeighborOnHop(
                    nid=torch.randint(0, 100, (n_root * S,)),
                    eid=torch.randint(0, 100, (n_root * S,)),
                    ts=torch.randn(n_root * S),
                    nfeat=None,
                    efeat=None,
                )
            ],
            fanout=fanout,
        )
        out = model(graph)
        assert out.embeddings.shape == (n_root, d_hid)
