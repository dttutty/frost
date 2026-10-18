"""Tests for model components, structured heads, and time encoding."""

import torch
import pytest

from frost.datatypes.graph import HierGraph, NeighborOnHop
from frost.models.encoder.dygformer import DyGFormer
from frost.models.header import (
    AdditiveLinkHead,
    ConcatLinkHead,
)
from frost.models.contracts import RepresentationKind
from frost.models.outputs import EncoderResult, RepresentationAdapter
from frost.models.modules.time_encoder import TimeEncoder
from frost.models.encoder.jodie import JODIETimeEmbedding


def _adapt(h: torch.Tensor, *, n_src: int, n_neg: int, kind: RepresentationKind):
    return RepresentationAdapter(kind)(EncoderResult(h), n_src=n_src, n_neg=n_neg)


@pytest.mark.parametrize("n_neg", [1, 20, 100])
@pytest.mark.parametrize(
    ("kind", "head_type"),
    [
        (RepresentationKind.NODE_ALIGNED, AdditiveLinkHead),
        (RepresentationKind.NODE_ALIGNED, ConcatLinkHead),
        (RepresentationKind.PAIR_ALIGNED, AdditiveLinkHead),
        (RepresentationKind.PAIR_ALIGNED, ConcatLinkHead),
    ],
)
def test_structured_link_pipeline_shapes(n_neg, kind, head_type):
    n_src, d_in = 3, 8
    row_count = (
        n_src * (2 + n_neg)
        if kind is RepresentationKind.NODE_ALIGNED
        else 2 * n_src * (1 + n_neg)
    )
    result = EncoderResult(torch.randn(row_count, d_in))
    edges = RepresentationAdapter(kind)(result, n_src=n_src, n_neg=n_neg)

    assert edges.positive_source.shape == (n_src, d_in)
    assert edges.negative_source.shape == (n_src * n_neg, d_in)
    assert head_type(d_in=d_in)(edges).shape == (n_src * (1 + n_neg), 1)


class TestAdditiveLinkHead:
    def test_output_shape(self):
        d_in, n_src, n_neg = 64, 8, 1
        pred = AdditiveLinkHead(d_in=d_in)
        h = torch.randn(n_src * (2 + n_neg), d_in)
        out = pred(
            _adapt(h, n_src=n_src, n_neg=n_neg, kind=RepresentationKind.NODE_ALIGNED)
        )
        assert out.shape == (n_src * (1 + n_neg), 1)

    def test_output_shape_multi_neg(self):
        d_in, n_src, n_neg = 32, 5, 3
        pred = AdditiveLinkHead(d_in=d_in)
        h = torch.randn(n_src * (2 + n_neg), d_in)
        out = pred(
            _adapt(h, n_src=n_src, n_neg=n_neg, kind=RepresentationKind.NODE_ALIGNED)
        )
        assert out.shape == (n_src * (1 + n_neg), 1)

    def test_gradient_flow(self):
        d_in, n_src, n_neg = 16, 4, 1
        pred = AdditiveLinkHead(d_in=d_in)
        h = torch.randn(n_src * (2 + n_neg), d_in, requires_grad=True)
        out = pred(
            _adapt(h, n_src=n_src, n_neg=n_neg, kind=RepresentationKind.NODE_ALIGNED)
        )
        out.sum().backward()
        assert h.grad is not None
        assert h.grad.shape == h.shape

    def test_pos_neg_split(self):
        """First n_src scores are positive, rest are negative."""
        d_in, n_src, n_neg = 16, 3, 2
        pred = AdditiveLinkHead(d_in=d_in)
        h = torch.randn(n_src * (2 + n_neg), d_in)
        out = pred(
            _adapt(h, n_src=n_src, n_neg=n_neg, kind=RepresentationKind.NODE_ALIGNED)
        )
        pos_scores = out[:n_src]
        neg_scores = out[n_src:]
        assert pos_scores.shape == (n_src, 1)
        assert neg_scores.shape == (n_src * n_neg, 1)

    def test_multi_neg_pairs_each_source_with_its_own_row(self):
        pred = AdditiveLinkHead(d_in=1)
        with torch.no_grad():
            for layer in (pred.src_fc, pred.dst_fc, pred.out_fc):
                layer.weight.fill_(1.0)
                layer.bias.zero_()

        # [src | pos_dst | per-edge negatives]
        h = torch.tensor(
            [[1.0], [10.0], [0.0], [0.0], [100.0], [200.0], [300.0], [400.0]]
        )
        logits = pred(
            _adapt(h, n_src=2, n_neg=2, kind=RepresentationKind.NODE_ALIGNED)
        ).squeeze(1)

        assert torch.equal(logits[2:], torch.tensor([101.0, 201.0, 310.0, 410.0]))

    def test_adapter_uses_runtime_n_neg(self):
        pred = AdditiveLinkHead(d_in=4)
        h = torch.randn(2 * (2 + 3), 4)
        edges = _adapt(h, n_src=2, n_neg=3, kind=RepresentationKind.NODE_ALIGNED)
        assert pred(edges).shape == (2 * (1 + 3), 1)


class TestConcatLinkHead:
    def test_multi_neg_pairs_each_source_with_its_own_row(self):
        pred = ConcatLinkHead(d_in=1)
        with torch.no_grad():
            pred.fc1.weight.fill_(1.0)
            pred.fc1.bias.zero_()
            pred.fc2.weight.fill_(1.0)
            pred.fc2.bias.zero_()

        h = torch.tensor(
            [[1.0], [10.0], [0.0], [0.0], [100.0], [200.0], [300.0], [400.0]]
        )
        logits = pred(
            _adapt(h, n_src=2, n_neg=2, kind=RepresentationKind.NODE_ALIGNED)
        ).squeeze(1)

        assert torch.equal(logits[2:], torch.tensor([101.0, 201.0, 310.0, 410.0]))


class TestPairAlignedRepresentations:
    def test_output_shape(self):
        d_in, n_src, n_neg = 32, 5, 2
        pred = AdditiveLinkHead(d_in=d_in)
        h = torch.randn(2 * n_src * (1 + n_neg), d_in)
        out = pred(
            _adapt(h, n_src=n_src, n_neg=n_neg, kind=RepresentationKind.PAIR_ALIGNED)
        )
        assert out.shape == (n_src * (1 + n_neg), 1)

    def test_negative_src_segment_affects_logits(self):
        d_in, n_src, n_neg = 16, 3, 2
        pred = AdditiveLinkHead(d_in=d_in)
        h = torch.randn(2 * n_src * (1 + n_neg), d_in, requires_grad=True)
        out = pred(
            _adapt(h, n_src=n_src, n_neg=n_neg, kind=RepresentationKind.PAIR_ALIGNED)
        )
        out.sum().backward()

        neg_src_lo = 2 * n_src
        neg_src_hi = neg_src_lo + n_src * n_neg
        assert h.grad is not None
        assert torch.count_nonzero(h.grad[neg_src_lo:neg_src_hi]).item() > 0

    def test_adapter_uses_runtime_n_neg(self):
        pred = AdditiveLinkHead(d_in=8)
        h = torch.randn(2 * 2 * (1 + 3), 8)
        edges = _adapt(h, n_src=2, n_neg=3, kind=RepresentationKind.PAIR_ALIGNED)
        assert pred(edges).shape == (2 * (1 + 3), 1)


class TestTimeEncoder:
    def test_output_shape_1d(self):
        enc = TimeEncoder(dim=128)
        t = torch.rand(10)
        out = enc(t)
        assert out.shape == (10, 128)

    def test_output_shape_2d(self):
        enc = TimeEncoder(dim=64)
        t = torch.rand(5, 3)
        out = enc(t)
        assert out.shape == (5, 3, 64)

    def test_int64_input_casts_to_float32_features(self):
        enc = TimeEncoder(dim=32)
        t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        out = enc(t)
        assert out.dtype == torch.float32
        assert out.shape == (2, 3, 32)

    def test_output_range(self):
        """Cosine output should be in [-1, 1]."""
        enc = TimeEncoder(dim=32)
        t = torch.rand(100) * 1000
        out = enc(t)
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_deterministic(self):
        enc = TimeEncoder(dim=16)
        t = torch.tensor([1.0, 2.0, 3.0])
        out1 = enc(t)
        out2 = enc(t)
        assert torch.equal(out1, out2)

    def test_gradient_flow(self):
        enc = TimeEncoder(dim=16)
        t = torch.rand(5, requires_grad=True)
        out = enc(t)
        out.sum().backward()
        assert t.grad is not None


class TestJODIETimeEmbedding:
    def test_preserves_small_delta_at_large_absolute_timestamp(self):
        projection = JODIETimeEmbedding(dim_out=1)
        with torch.no_grad():
            projection.time_emb.weight.fill_(100_000_000.0)
            projection.time_emb.bias.zero_()

        h = torch.ones(1, 1)
        ts = torch.tensor([1_700_000_001], dtype=torch.int64)
        mem_ts = torch.tensor([1_700_000_000], dtype=torch.int64)

        out = projection(h, mem_ts, ts)

        expected = 1.0 + 100_000_000.0 / 1_700_000_002.0
        assert out.item() == pytest.approx(expected, rel=1e-6)


class TestDyGFormer:
    def test_build_sequences_left_aligns_recent_neighbors(self):
        model = DyGFormer(
            ndim=1,
            edim=0,
            d_time=4,
            d_out=8,
            channel_embedding_dim=2,
            patch_size=2,
            transformer_layers=1,
            num_heads=1,
            dropout=0.0,
            max_input_sequence_length=4,
            n_neg=1,
            enable_dummy_edge_features=False,
        )

        graph = HierGraph(
            root_nid=torch.tensor([100, 200], dtype=torch.int64),
            root_ts=torch.tensor([10, 20], dtype=torch.int64),
            root_nfeat=torch.tensor([[1000.0], [2000.0]]),
            root_mem=None,
            root_mem_ts=None,
            root_box=None,
            root_box_ts=None,
            neighbors=[
                NeighborOnHop(
                    nid=torch.tensor(
                        [-1, -1, -1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64
                    ),
                    eid=torch.full((10,), -1, dtype=torch.int64),
                    ts=torch.tensor(
                        [0, 0, 0, 8, 9, 14, 15, 16, 17, 18], dtype=torch.int64
                    ),
                    nfeat=torch.tensor(
                        [
                            [0.0],
                            [0.0],
                            [0.0],
                            [12.0],
                            [13.0],
                            [14.0],
                            [15.0],
                            [16.0],
                            [17.0],
                            [18.0],
                        ]
                    ),
                    efeat=None,
                    mem=None,
                    mem_ts=None,
                    box=None,
                    box_ts=None,
                )
            ],
            fanout=[5],
        )

        seq = model._build_sequences(graph, n_root=2, K=5)

        assert torch.equal(
            seq["nid"],
            torch.tensor([[100, 2, 3, -1], [200, 6, 7, 8]], dtype=torch.int64),
        )
        assert torch.equal(
            seq["ts"],
            torch.tensor([[10, 8, 9, 0], [20, 16, 17, 18]], dtype=torch.int64),
        )
        assert torch.equal(
            seq["nfeat"].squeeze(-1),
            torch.tensor([[1000.0, 12.0, 13.0, 0.0], [2000.0, 16.0, 17.0, 18.0]]),
        )

    def test_forward_returns_pair_specific_layout(self):
        model = DyGFormer(
            ndim=1,
            edim=0,
            d_time=4,
            d_out=8,
            channel_embedding_dim=2,
            patch_size=2,
            transformer_layers=1,
            num_heads=1,
            dropout=0.0,
            max_input_sequence_length=4,
            n_neg=1,
            enable_dummy_edge_features=False,
        )
        predictor = AdditiveLinkHead(d_in=8)

        graph = HierGraph(
            root_nid=torch.tensor([100, 200, 300], dtype=torch.int64),
            root_ts=torch.tensor([10, 10, 10], dtype=torch.int64),
            root_nfeat=torch.tensor([[1.0], [2.0], [3.0]]),
            root_mem=None,
            root_mem_ts=None,
            root_box=None,
            root_box_ts=None,
            neighbors=[
                NeighborOnHop(
                    nid=torch.tensor([-1, 11, -1, 21, -1, 31], dtype=torch.int64),
                    eid=torch.full((6,), -1, dtype=torch.int64),
                    ts=torch.tensor([0, 9, 0, 8, 0, 7], dtype=torch.int64),
                    nfeat=torch.tensor([[0.0], [11.0], [0.0], [21.0], [0.0], [31.0]]),
                    efeat=None,
                    mem=None,
                    mem_ts=None,
                    box=None,
                    box_ts=None,
                )
            ],
            fanout=[2],
        )

        result = model(graph)
        edges = RepresentationAdapter(RepresentationKind.PAIR_ALIGNED)(
            result, n_src=1, n_neg=1
        )
        logits = predictor(edges)

        assert result.embeddings.shape == (4, 8)
        assert logits.shape == (2, 1)
