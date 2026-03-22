"""Tests for NegLinkSampler."""

import torch

from frost.sampling.neg_sampler import NegLinkSampler


class TestNegLinkSampler:
    def test_output_shape_single_neg(self):
        sampler = NegLinkSampler(n_nodes=100, n_neg=1)
        src = torch.arange(10)
        neg = sampler.sample(src)
        assert neg.shape == (10,)

    def test_output_shape_multi_neg(self):
        sampler = NegLinkSampler(n_nodes=100, n_neg=5)
        src = torch.arange(8)
        neg = sampler.sample(src)
        assert neg.shape == (40,)

    def test_value_range(self):
        n_nodes = 50
        sampler = NegLinkSampler(n_nodes=n_nodes, n_neg=3)
        src = torch.arange(20)
        neg = sampler.sample(src)
        assert neg.min().item() >= 0
        assert neg.max().item() < n_nodes

    def test_deterministic_with_seed(self):
        sampler = NegLinkSampler(n_nodes=1000, n_neg=2)
        src = torch.arange(16)
        torch.manual_seed(42)
        a = sampler.sample(src)
        torch.manual_seed(42)
        b = sampler.sample(src)
        assert torch.equal(a, b)

    def test_single_source(self):
        sampler = NegLinkSampler(n_nodes=10, n_neg=3)
        src = torch.tensor([5])
        neg = sampler.sample(src)
        assert neg.shape == (3,)
