"""Tests for model components: LinkPredictor, TimeEncoder, MemoryUpdater."""

import torch

from frost.models.header import LinkPredictor, NodeClassificationHeader
from frost.models.modules.time_encoder import TimeEncoder


class TestLinkPredictor:
    def test_output_shape(self):
        d_in, n_src, n_neg = 64, 8, 1
        pred = LinkPredictor(d_in=d_in, n_neg=n_neg)
        h = torch.randn(n_src * (2 + n_neg), d_in)
        out = pred(h)
        assert out.shape == (n_src * (1 + n_neg), 1)

    def test_output_shape_multi_neg(self):
        d_in, n_src, n_neg = 32, 5, 3
        pred = LinkPredictor(d_in=d_in, n_neg=n_neg)
        h = torch.randn(n_src * (2 + n_neg), d_in)
        out = pred(h)
        assert out.shape == (n_src * (1 + n_neg), 1)

    def test_gradient_flow(self):
        d_in, n_src, n_neg = 16, 4, 1
        pred = LinkPredictor(d_in=d_in, n_neg=n_neg)
        h = torch.randn(n_src * (2 + n_neg), d_in, requires_grad=True)
        out = pred(h)
        out.sum().backward()
        assert h.grad is not None
        assert h.grad.shape == h.shape

    def test_pos_neg_split(self):
        """First n_src scores are positive, rest are negative."""
        d_in, n_src, n_neg = 16, 3, 2
        pred = LinkPredictor(d_in=d_in, n_neg=n_neg)
        h = torch.randn(n_src * (2 + n_neg), d_in)
        out = pred(h)
        pos_scores = out[:n_src]
        neg_scores = out[n_src:]
        assert pos_scores.shape == (n_src, 1)
        assert neg_scores.shape == (n_src * n_neg, 1)


class TestNodeClassificationHeader:
    def test_output_shape(self):
        header = NodeClassificationHeader(dim_in=64, d_hid=32, num_class=5)
        x = torch.randn(10, 64)
        out = header(x)
        assert out.shape == (10, 5)

    def test_gradient_flow(self):
        header = NodeClassificationHeader(dim_in=32, d_hid=16, num_class=3)
        x = torch.randn(4, 32, requires_grad=True)
        out = header(x)
        out.sum().backward()
        assert x.grad is not None


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
