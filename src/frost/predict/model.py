"""A small, single-process TGAT trainer using FROST's existing encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from frost.datatypes.graph import HierGraph, NeighborOnHop
from frost.models.encoder.tgat import TGAT
from frost.predict.data import events, load_features


class History:
    def __init__(self, edges: np.ndarray, timestamps: np.ndarray, size: int):
        buckets: list[list[tuple[float, int, int]]] = [[] for _ in range(size)]
        for eid, ((src, dst), ts) in enumerate(zip(edges, timestamps)):
            buckets[src].append((float(ts), int(dst), eid))
            if src != dst:
                buckets[dst].append((float(ts), int(src), eid))
        self.rows = []
        for bucket in buckets:
            bucket.sort(key=lambda row: row[0])
            self.rows.append(
                (
                    np.array([r[0] for r in bucket], dtype=np.float64),
                    np.array([r[1] for r in bucket], dtype=np.int64),
                    np.array([r[2] for r in bucket], dtype=np.int64),
                )
            )

    def sample(self, nodes, timestamps, fanout, inclusive=False):
        shape = (len(nodes), fanout)
        neighbors = np.full(shape, -1, dtype=np.int64)
        edge_ids = np.full(shape, -1, dtype=np.int64)
        ts = np.repeat(np.asarray(timestamps)[:, None], fanout, axis=1)
        for row, (node, time) in enumerate(zip(nodes, timestamps)):
            history_ts, history_nodes, history_eids = self.rows[node]
            end = np.searchsorted(
                history_ts, time, side="right" if inclusive else "left"
            )
            start = max(0, end - fanout)
            count = end - start
            if count:
                neighbors[row, -count:] = history_nodes[start:end]
                edge_ids[row, -count:] = history_eids[start:end]
                ts[row, -count:] = history_ts[start:end]
        return neighbors, edge_ids, ts


class Predictor(nn.Module):
    def __init__(
        self, count: int, features: torch.Tensor, fanout: int = 20, hidden: int = 64
    ):
        super().__init__()
        self.fanout = fanout
        self.embedding = nn.Embedding(count, hidden)
        self.register_buffer("features", features)
        self.encoder = TGAT(
            ndim=hidden + features.shape[1],
            edim=0,
            d_time=16,
            d_hid=hidden,
            fanout=[fanout],
            dropout=0.1,
            enable_dummy_edge_features=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def node_features(self, nodes):
        safe = nodes.clamp_min(0)
        result = torch.cat((self.embedding(safe), self.features[safe]), dim=1)
        return result * (nodes >= 0).unsqueeze(1)

    def forward(self, history, src, dst, timestamps, inclusive=False):
        nodes = np.concatenate((src, dst))
        root_times = np.concatenate((timestamps, timestamps))
        neighbors, eids, neighbor_times = history.sample(
            nodes, root_times, self.fanout, inclusive
        )
        device = self.features.device
        root_ids = torch.as_tensor(nodes, dtype=torch.long, device=device)
        neighbor_ids = torch.as_tensor(neighbors.reshape(-1), device=device)
        graph = HierGraph(
            root_nid=root_ids,
            root_ts=torch.as_tensor(root_times, device=device),
            root_nfeat=self.node_features(root_ids),
            fanout=[self.fanout],
            neighbors=[
                NeighborOnHop(
                    nid=neighbor_ids,
                    eid=torch.as_tensor(eids.reshape(-1), device=device),
                    ts=torch.as_tensor(neighbor_times.reshape(-1), device=device),
                    nfeat=self.node_features(neighbor_ids),
                )
            ],
        )
        embedding = self.encoder(graph).embeddings
        left, right = embedding.chunk(2)
        return self.head(torch.cat((left, right), dim=1)).squeeze(1)


def train(
    csv_path,
    *,
    output,
    node_features=None,
    feature_ids=None,
    src_col=None,
    dst_col=None,
    time_col=None,
    epochs=10,
    batch_size=200,
    fanout=20,
    hidden=64,
    lr=0.001,
    seed=0,
    device="cpu",
):
    if (
        min(epochs, batch_size, fanout, hidden) < 1
        or hidden % 2
        or not np.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError(
            "Epochs, batch size and fanout must be positive; hidden must be positive/even; lr > 0."
        )
    target = Path(output)
    if target.exists():
        raise ValueError(f"Checkpoint already exists: {target}; choose a new --output.")
    src, dst, timestamps, kind = events(csv_path, src_col, dst_col, time_col)
    node_ids = list(dict.fromkeys(src + dst))
    mapping = {node: index for index, node in enumerate(node_ids)}
    edges = np.array(
        [(mapping[s], mapping[d]) for s, d in zip(src, dst)], dtype=np.int64
    )
    order = np.argsort(timestamps, kind="stable")
    edges, timestamps = edges[order], timestamps[order]
    features = torch.from_numpy(load_features(node_features, node_ids, feature_ids))
    destinations = np.unique(edges[:, 1])
    # Exclude every known positive at the same time, not only the current row.
    positives: dict[tuple[int, float], set[int]] = {}
    for (s, d), t in zip(edges, timestamps):
        positives.setdefault((int(s), float(t)), set()).add(int(d))
    for (s, _), observed in positives.items():
        if not any(int(d) not in observed and d != s for d in destinations):
            raise ValueError(
                "No negative destination available for an event; provide more destination diversity."
            )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Predictor(len(node_ids), features, fanout, hidden).to(device)
    history = History(edges, timestamps, len(node_ids))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    print(
        f"Training TGAT: {len(edges)} events, {len(node_ids)} nodes, time={kind}, device={device}.",
        file=sys.stderr,
    )
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for start in range(0, len(edges), batch_size):
            batch = edges[start : start + batch_size]
            ts = timestamps[start : start + batch_size]
            negatives = []
            for (s, _), time in zip(batch, ts):
                forbidden = positives[(int(s), float(time))] | {int(s)}
                candidate = int(rng.choice(destinations))
                for _ in range(32):
                    if candidate not in forbidden:
                        break
                    candidate = int(rng.choice(destinations))
                if candidate in forbidden:
                    candidate = int(
                        rng.choice([d for d in destinations if d not in forbidden])
                    )
                negatives.append(candidate)
            source = np.concatenate((batch[:, 0], batch[:, 0]))
            destination = np.concatenate((batch[:, 1], negatives))
            logits = model(history, source, destination, np.concatenate((ts, ts)))
            labels = torch.cat((torch.ones(len(batch)), torch.zeros(len(batch)))).to(
                device
            )
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError(
                    "Training produced a nonfinite loss; check feature and timestamp scales."
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(edges))
        print(f"Epoch {epoch + 1}/{epochs}: loss={losses[-1]:.5f}", file=sys.stderr)
    checkpoint = {
        "format_version": 1,
        "model": "tgat",
        "node_ids": node_ids,
        "edges": torch.from_numpy(edges),
        "timestamps": torch.from_numpy(timestamps),
        "time_kind": kind,
        "last_time": float(timestamps[-1]),
        "config": {"fanout": fanout, "hidden": hidden},
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "loss": losses,
            "negative_sampling": "uniform_destination_1",
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        torch.save(checkpoint, stream)
    print(f"Saved {target}", file=sys.stderr)
    return checkpoint


def load(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("model") != "tgat":
        raise ValueError("Unsupported FROST prediction checkpoint.")
    model = Predictor(
        len(checkpoint["node_ids"]),
        checkpoint["state_dict"]["features"],
        **checkpoint["config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    history = History(
        checkpoint["edges"].numpy(),
        checkpoint["timestamps"].numpy(),
        len(checkpoint["node_ids"]),
    )
    return checkpoint, model, history


def score(
    checkpoint, model, history, sources, destinations, timestamps, batch_size=200
):
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if not (len(sources) == len(destinations) == len(timestamps)) or not len(sources):
        raise ValueError(
            "Provide equally sized, nonempty source, destination and time arrays."
        )
    if not np.isfinite(timestamps).all() or np.any(
        np.asarray(timestamps) < checkpoint["last_time"]
    ):
        raise ValueError(
            "Query time must be at or after the training cutoff; train on earlier data to backtest."
        )
    mapping = {node: index for index, node in enumerate(checkpoint["node_ids"])}
    unknown = sorted((set(sources) | set(destinations)) - mapping.keys())
    if unknown:
        raise ValueError(
            f"Unknown node IDs: {unknown[:5]}; this predictor requires nodes seen in training."
        )
    src = np.array([mapping[node] for node in sources])
    dst = np.array([mapping[node] for node in destinations])
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(src), batch_size):
            end = start + batch_size
            result.extend(
                torch.sigmoid(
                    model(
                        history,
                        src[start:end],
                        dst[start:end],
                        timestamps[start:end],
                        inclusive=True,
                    )
                )
                .cpu()
                .tolist()
            )
    return result
