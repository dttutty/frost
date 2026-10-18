"""Tests for TemporalGraphAttentionLayer."""

import torch
import torch.nn as nn

from frost.models.modules.temporal_graph_attention_layer import (
    TemporalGraphAttentionLayer,
)


class TestTemporalGraphAttentionLayer:
    def test_packed_matches_dense_with_padding(self):
        torch.manual_seed(7)
        layer = TemporalGraphAttentionLayer(
            dim_node=8,
            dim_edge=3,
            dim_time=4,
            dim_out=8,
            num_head=2,
            dropout=0.0,
        )
        target = torch.randn(5, 8)
        neighbor = torch.randn(5, 4, 8)
        edge = torch.randn(5, 4, 3)
        dt = torch.rand(5, 4)
        mask = torch.tensor(
            [
                [True, True, False, False],
                [False, False, False, False],
                [True, False, True, False],
                [True, True, True, True],
                [False, True, False, False],
            ]
        )

        dense = layer(target, neighbor, edge, dt, mask)
        packed = layer.forward_packed(target, neighbor, edge, dt, mask)

        torch.testing.assert_close(packed, dense, atol=1e-5, rtol=1e-5)

    def test_packed_gradients_match_dense_with_padding(self):
        torch.manual_seed(11)
        layer = TemporalGraphAttentionLayer(
            dim_node=6,
            dim_edge=2,
            dim_time=4,
            dim_out=6,
            num_head=2,
            dropout=0.0,
        )
        target = torch.randn(4, 6, requires_grad=True)
        neighbor = torch.randn(4, 5, 6, requires_grad=True)
        edge = torch.randn(4, 5, 2, requires_grad=True)
        dt = torch.rand(4, 5)
        mask = torch.tensor(
            [
                [True, False, True, False, False],
                [False, False, False, False, False],
                [True, True, True, False, False],
                [False, True, False, True, True],
            ]
        )
        differentiable = (target, neighbor, edge, *layer.parameters())

        dense_grad = torch.autograd.grad(
            layer(target, neighbor, edge, dt, mask).sum(),
            differentiable,
        )
        packed_grad = torch.autograd.grad(
            layer.forward_packed(target, neighbor, edge, dt, mask).sum(),
            differentiable,
        )

        for actual, expected in zip(packed_grad, dense_grad, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_packed_empty_batch_preserves_kv_gradients(self):
        layer = TemporalGraphAttentionLayer(
            dim_node=4,
            dim_edge=0,
            dim_time=2,
            dim_out=4,
            num_head=2,
            dropout=0.0,
        )
        target = torch.randn(3, 4, requires_grad=True)
        neighbor = torch.randn(3, 2, 4, requires_grad=True)
        dt = torch.zeros(3, 2)
        mask = torch.zeros(3, 2, dtype=torch.bool)

        layer.forward_packed(target, neighbor, None, dt, mask).sum().backward()

        assert layer.w_k.weight.grad is not None
        assert layer.w_v.weight.grad is not None

    def test_output_projection_concatenates_aggregate_before_target(self):
        layer = TemporalGraphAttentionLayer(
            dim_node=1,
            dim_edge=1,
            dim_time=0,
            dim_out=3,
            num_head=1,
            dropout=0.0,
        )
        layer.layer_norm = nn.Identity()
        with torch.no_grad():
            layer.w_q.weight.zero_()
            layer.w_q.bias.zero_()
            layer.w_k.weight.zero_()
            layer.w_k.bias.zero_()
            layer.w_v.weight.zero_()
            layer.w_v.bias.copy_(torch.tensor([2.0, 3.0, 5.0]))
            layer.w_out.weight.zero_()
            layer.w_out.bias.zero_()
            layer.w_out.weight[:, :3] = torch.eye(3)

        out = layer(
            target_feat=torch.tensor([[7.0]]),
            neighbor_feat=torch.tensor([[[11.0]]]),
            edge_feat=torch.tensor([[[13.0]]]),
            dt=torch.zeros(1, 1),
            mask=torch.ones(1, 1, dtype=torch.bool),
        )

        assert torch.equal(out, torch.tensor([[2.0, 3.0, 5.0]]))

    def test_output_shape(self):
        dim_node, dim_edge, dim_time, dim_out, num_head = 32, 16, 8, 64, 4
        layer = TemporalGraphAttentionLayer(
            dim_node=dim_node,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=num_head,
        )
        N, S = 10, 20
        target_feat = torch.randn(N, dim_node)
        neighbor_feat = torch.randn(N, S, dim_node)
        edge_feat = torch.randn(N, S, dim_edge)
        dt = torch.rand(N, S) * 100
        mask = torch.ones(N, S, dtype=torch.bool)
        out = layer(target_feat, neighbor_feat, edge_feat, dt, mask)
        assert out.shape == (N, dim_out)

    def test_output_shape_no_time_encoding(self):
        dim_node, dim_edge, dim_time, dim_out, num_head = 32, 16, 0, 64, 4
        layer = TemporalGraphAttentionLayer(
            dim_node=dim_node,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=num_head,
        )
        N, S = 5, 10
        target_feat = torch.randn(N, dim_node)
        neighbor_feat = torch.randn(N, S, dim_node)
        edge_feat = torch.randn(N, S, dim_edge)
        dt = torch.rand(N, S) * 100
        mask = torch.ones(N, S, dtype=torch.bool)
        out = layer(target_feat, neighbor_feat, edge_feat, dt, mask)
        assert out.shape == (N, dim_out)

    def test_single_head(self):
        dim_node, dim_edge, dim_time, dim_out, num_head = 32, 8, 4, 32, 1
        layer = TemporalGraphAttentionLayer(
            dim_node=dim_node,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=num_head,
        )
        N, S = 8, 15
        target_feat = torch.randn(N, dim_node)
        neighbor_feat = torch.randn(N, S, dim_node)
        edge_feat = torch.randn(N, S, dim_edge)
        dt = torch.rand(N, S) * 50
        mask = torch.ones(N, S, dtype=torch.bool)
        out = layer(target_feat, neighbor_feat, edge_feat, dt, mask)
        assert out.shape == (N, dim_out)

    def test_with_padding_mask(self):
        dim_node, dim_edge, dim_time, dim_out, num_head = 16, 8, 4, 32, 2
        layer = TemporalGraphAttentionLayer(
            dim_node=dim_node,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=num_head,
        )
        N, S = 4, 10
        target_feat = torch.randn(N, dim_node)
        neighbor_feat = torch.randn(N, S, dim_node)
        edge_feat = torch.randn(N, S, dim_edge)
        dt = torch.rand(N, S) * 100
        mask = torch.ones(N, S, dtype=torch.bool)
        mask[:, 5:] = False
        out = layer(target_feat, neighbor_feat, edge_feat, dt, mask)
        assert out.shape == (N, dim_out)

    def test_all_padding_neighbors(self):
        dim_node, dim_edge, dim_time, dim_out, num_head = 16, 4, 4, 16, 2
        layer = TemporalGraphAttentionLayer(
            dim_node=dim_node,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=num_head,
        )
        N, S = 3, 5
        target_feat = torch.randn(N, dim_node)
        neighbor_feat = torch.randn(N, S, dim_node)
        edge_feat = torch.randn(N, S, dim_edge)
        dt = torch.rand(N, S) * 100
        mask = torch.zeros(N, S, dtype=torch.bool)
        out = layer(target_feat, neighbor_feat, edge_feat, dt, mask)
        assert out.shape == (N, dim_out)

    def test_gradient_flow(self):
        dim_node, dim_edge, dim_time, dim_out, num_head = 16, 4, 4, 16, 2
        layer = TemporalGraphAttentionLayer(
            dim_node=dim_node,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=num_head,
        )
        N, S = 4, 8
        target_feat = torch.randn(N, dim_node, requires_grad=True)
        neighbor_feat = torch.randn(N, S, dim_node, requires_grad=True)
        edge_feat = torch.randn(N, S, dim_edge, requires_grad=True)
        dt = torch.rand(N, S) * 100
        mask = torch.ones(N, S, dtype=torch.bool)
        out = layer(target_feat, neighbor_feat, edge_feat, dt, mask)
        out.sum().backward()
        assert target_feat.grad is not None
        assert neighbor_feat.grad is not None
        assert edge_feat.grad is not None

    def test_different_batch_sizes(self):
        dim_node, dim_edge, dim_time, dim_out, num_head = 16, 4, 4, 32, 2
        layer = TemporalGraphAttentionLayer(
            dim_node=dim_node,
            dim_edge=dim_edge,
            dim_time=dim_time,
            dim_out=dim_out,
            num_head=num_head,
        )
        for N in [1, 4, 16, 64]:
            S = 10
            target_feat = torch.randn(N, dim_node)
            neighbor_feat = torch.randn(N, S, dim_node)
            edge_feat = torch.randn(N, S, dim_edge)
            dt = torch.rand(N, S) * 100
            mask = torch.ones(N, S, dtype=torch.bool)
            out = layer(target_feat, neighbor_feat, edge_feat, dt, mask)
            assert out.shape == (N, dim_out)
