"""Benchmark dense-tensor temporal attention (FROST-style HierGraph).

Reproduces FROST's TemporalGraphAttentionLayer on dense [N, S, D] tensors.
Measures forward + backward latency across varying (N, S) configurations.
Functionally equivalent to TGL's sparse temporal-attention layer.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json


class TimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(1, dim)
        self.w.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32)).reshape(dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        return torch.cos(self.w(t.unsqueeze(-1)))


class DenseTemporalAttention(nn.Module):
    """FROST's TemporalGraphAttentionLayer — pure dense-tensor ops."""

    def __init__(self, dim_node, dim_edge, dim_time, dim_out, num_head, dropout=0.1):
        super().__init__()
        self.num_head = num_head
        self.dim_out = dim_out
        self.dim_node = dim_node
        self.head_dim = dim_out // num_head
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.time_enc = TimeEncoder(dim_time) if dim_time > 0 else None
        self.w_q = nn.Linear(dim_node + dim_time, dim_out)
        self.w_k = nn.Linear(dim_node + dim_edge + dim_time, dim_out)
        self.w_v = nn.Linear(dim_node + dim_edge + dim_time, dim_out)
        self.w_out = nn.Linear(dim_node + dim_out, dim_out)

    def forward(self, target_feat, neighbor_feat, edge_feat, dt, mask):
        """
        Args:
            target_feat:   [N, dim_node]
            neighbor_feat: [N, S, dim_node]
            edge_feat:     [N, S, dim_edge]
            dt:            [N, S]
            mask:          [N, S] bool (True = valid)
        Returns:
            [N, dim_out]
        """
        N, S = dt.shape

        # 1. Time encoding
        time_feat = self.time_enc(dt)                    # [N, S, dim_time]
        zero_time_feat = self.time_enc(
            torch.zeros(N, device=dt.device, dtype=dt.dtype)
        )                                                 # [N, dim_time]

        # 2. Q from target nodes
        Q = self.w_q(torch.cat([target_feat, zero_time_feat], dim=-1))  # [N, dim_out]
        Q = Q.view(N, 1, self.num_head, self.head_dim)                 # [N, 1, H, D_h]

        # 3. K, V from neighbors
        kv_input = torch.cat([neighbor_feat, edge_feat, time_feat], dim=-1)
        K = self.w_k(kv_input).view(N, S, self.num_head, self.head_dim)
        V = self.w_v(kv_input).view(N, S, self.num_head, self.head_dim)

        # 4. Attention scores
        att = F.leaky_relu(torch.sum(Q * K, dim=-1), negative_slope=0.2)  # [N, S, H]
        att = att.masked_fill(~mask.unsqueeze(-1), -1e9)
        att = F.softmax(att, dim=1)
        att = self.dropout(att)

        # 5. Aggregate
        agg = torch.sum(att.unsqueeze(-1) * V, dim=1)  # [N, H, D_h]
        agg = agg.reshape(N, self.dim_out)

        # 6. Residual + output
        out = self.w_out(torch.cat([target_feat, agg], dim=1))
        out = F.relu(self.dropout(out))
        return self.layer_norm(out)


def make_dense_inputs(N, S, dim_node, dim_edge, device):
    target_feat = torch.randn(N, dim_node, device=device)
    neighbor_feat = torch.randn(N, S, dim_node, device=device)
    edge_feat = torch.randn(N, S, dim_edge, device=device)
    dt = torch.rand(N, S, device=device)
    mask = torch.ones(N, S, dtype=torch.bool, device=device)
    return target_feat, neighbor_feat, edge_feat, dt, mask


def benchmark(N, S, dim_node, dim_edge, dim_time, dim_out, num_head,
              warmup=10, repeat=100, device='cuda', use_compile=False):
    model = DenseTemporalAttention(dim_node, dim_edge, dim_time, dim_out, num_head).to(device)
    if use_compile:
        model = torch.compile(model)
    model.train()

    # Warmup
    for _ in range(warmup):
        inputs = make_dense_inputs(N, S, dim_node, dim_edge, device)
        out = model(*inputs)
        out.sum().backward()
    torch.cuda.synchronize()

    # Timed runs
    fwd_times = []
    bwd_times = []
    for _ in range(repeat):
        inputs = make_dense_inputs(N, S, dim_node, dim_edge, device)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = model(*inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        out.sum().backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    fwd_ms = np.array(fwd_times) * 1000
    bwd_ms = np.array(bwd_times) * 1000
    total_ms = fwd_ms + bwd_ms
    return {
        'N': N, 'S': S,
        'fwd_ms': float(np.mean(fwd_ms)),
        'bwd_ms': float(np.mean(bwd_ms)),
        'total_ms': float(np.mean(total_ms)),
        'fwd_std': float(np.std(fwd_ms)),
        'bwd_std': float(np.std(bwd_ms)),
        'total_std': float(np.std(total_ms)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dim_node', type=int, default=172)
    parser.add_argument('--dim_edge', type=int, default=172)
    parser.add_argument('--dim_time', type=int, default=100)
    parser.add_argument('--dim_out', type=int, default=100)
    parser.add_argument('--num_head', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--compile', action='store_true', help='enable torch.compile')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    configs = [
        (200, 10), (200, 20), (200, 30),
        (600, 10), (600, 20), (600, 30),
        (1000, 10), (1000, 20), (1000, 30),
    ]

    mode = "torch.compile" if args.compile else "Eager"
    print(f"{'='*70}")
    print(f"Dense Tensor Temporal Attention Benchmark ({mode})")
    print(f"dim_node={args.dim_node} dim_edge={args.dim_edge} dim_time={args.dim_time} dim_out={args.dim_out} heads={args.num_head}")
    print(f"{'='*70}")
    print(f"{'N':>6} {'S':>4} | {'Fwd (ms)':>10} {'Bwd (ms)':>10} {'Total (ms)':>12} {'± std':>8}")
    print(f"{'-'*70}")

    results = []
    for N, S in configs:
        r = benchmark(N, S, args.dim_node, args.dim_edge, args.dim_time,
                       args.dim_out, args.num_head, args.warmup, args.repeat,
                       device, args.compile)
        r['mode'] = mode
        results.append(r)
        print(f"{r['N']:>6} {r['S']:>4} | {r['fwd_ms']:>10.3f} {r['bwd_ms']:>10.3f} {r['total_ms']:>12.3f} {r['total_std']:>8.3f}")

    suffix = "compile" if args.compile else "eager"
    fname = f'bench_dense_{suffix}_results.json'
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {fname}")
