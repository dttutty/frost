"""Single-process temporal prediction using FROST encoders and memory updates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from frost.datatypes.graph import FlatGraph, HierGraph, NeighborOnHop
from frost.models.encoder.tgat import TGAT
from frost.models.encoder.tgn import TGN
from frost.models.header import AdditiveLinkHead
from frost.models.mem_updater import MemoryUpdater
from frost.models.outputs import EdgeRepresentations
from frost.predict.config import DEFAULTS, resolve
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
        self,
        count: int,
        features: torch.Tensor,
        fanout: int = 20,
        hidden: int = 64,
        time_dim: int = 16,
        heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fanout = fanout
        self.embedding = nn.Embedding(count, hidden)
        self.register_buffer("features", features)
        self.encoder: nn.Module = TGAT(
            ndim=hidden + features.shape[1],
            edim=0,
            d_time=time_dim,
            d_hid=hidden,
            fanout=[fanout],
            heads=heads,
            dropout=dropout,
            enable_dummy_edge_features=False,
        )
        self.head: nn.Module = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def node_features(self, nodes):
        safe = nodes.clamp_min(0)
        result = torch.cat((self.embedding(safe), self.features[safe]), dim=1)
        return result * (nodes >= 0).unsqueeze(1)

    def memory_values(self, nodes):
        return None

    def link_logits(self, left, right):
        return self.head(torch.cat((left, right), dim=1)).squeeze(1)

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
            root_mem=self.memory_values(root_ids),
            fanout=[self.fanout],
            neighbors=[
                NeighborOnHop(
                    nid=neighbor_ids,
                    eid=torch.as_tensor(eids.reshape(-1), device=device),
                    ts=torch.as_tensor(neighbor_times.reshape(-1), device=device),
                    nfeat=self.node_features(neighbor_ids),
                    mem=self.memory_values(neighbor_ids),
                )
            ],
        )
        embedding = self.encoder(graph).embeddings
        left, right = embedding.chunk(2)
        return self.link_logits(left, right)


class TGNPredictor(Predictor):
    def __init__(
        self, count, features, fanout=10, hidden=128, time_dim=128, heads=2, dropout=0.2
    ):
        nn.Module.__init__(self)
        self.fanout = fanout
        self.embedding = nn.Embedding(count, hidden)
        self.register_buffer("features", features)
        self.encoder = TGN(
            ndim=hidden + features.shape[1],
            edim=0,
            d_mem=hidden,
            d_time=time_dim,
            d_hid=hidden,
            heads=heads,
            dropout=dropout,
        )
        self.updater = MemoryUpdater(
            d_mail=2 * hidden, d_time=time_dim, d_mem=hidden, cell="gru"
        )
        self.head = AdditiveLinkHead(hidden)
        self.register_buffer("memory", torch.zeros(count, hidden))
        self.register_buffer("memory_time", torch.zeros(count, dtype=torch.float64))
        self.register_buffer("mail", torch.zeros(count, 2 * hidden))
        self.register_buffer("mail_time", torch.zeros(count, dtype=torch.float64))
        self.register_buffer("has_mail", torch.zeros(count, dtype=torch.bool))

    def reset_memory(self, start_time):
        self.memory.zero_()
        self.mail.zero_()
        self.memory_time.fill_(start_time)
        self.mail_time.fill_(start_time)
        self.has_mail.zero_()

    def memory_values(self, nodes):
        safe = nodes.clamp_min(0)
        graph = FlatGraph(
            nid=nodes,
            eid=torch.full_like(nodes, -1),
            ts=self.mail_time[safe],
            mem=self.memory[safe],
            mem_ts=self.memory_time[safe],
            box=self.mail[safe],
            box_ts=self.mail_time[safe],
        )
        updated = self.updater(graph, n_target_nodes=0).mem
        assert updated is not None
        result = torch.where(self.has_mail[safe, None], updated, self.memory[safe])
        return result * (nodes >= 0).unsqueeze(1)

    def link_logits(self, left, right):
        return self.head(EdgeRepresentations(left, right, left[:0], right[:0])).squeeze(
            1
        )

    @torch.no_grad()
    def observe(self, edges, timestamps):
        # One snapshot for the entire chronological batch; last message wins.
        # Only positive endpoints publish state. Negative/query pairs never do.
        endpoints = np.asarray(edges).reshape(-1)
        unique, inverse = np.unique(endpoints, return_inverse=True)
        device = self.memory.device
        nodes = torch.as_tensor(unique, device=device)
        updated = self.memory_values(nodes)
        last = np.full(len(unique), -1, dtype=np.int64)
        np.maximum.at(last, inverse, np.arange(len(endpoints)))
        self.memory[nodes] = updated
        self.memory_time[nodes] = torch.where(
            self.has_mail[nodes], self.mail_time[nodes], self.memory_time[nodes]
        )
        # Select each node's final peer before materializing messages, so repeated
        # endpoints in a large equal-time group do not duplicate feature tensors.
        peers = updated[torch.as_tensor(inverse[last ^ 1], device=device)]
        self.mail[nodes] = torch.cat((updated, peers), dim=1)
        self.mail_time[nodes] = torch.as_tensor(
            np.asarray(timestamps)[last // 2], device=device
        )
        self.has_mail[nodes] = True


def temporal_batches(timestamps, batch_size):
    """Extend batch boundaries to keep equal-time events on one state snapshot."""
    start = 0
    while start < len(timestamps):
        end = min(start + batch_size, len(timestamps))
        end = int(np.searchsorted(timestamps, timestamps[end - 1], side="right"))
        yield start, end
        start = end


def train(
    csv_path,
    *,
    output,
    node_features=None,
    feature_ids=None,
    src_col=None,
    dst_col=None,
    time_col=None,
    epochs=DEFAULTS["epochs"],
    batch_size=DEFAULTS["batch_size"],
    fanout=DEFAULTS["fanout"],
    hidden=DEFAULTS["hidden"],
    lr=DEFAULTS["lr"],
    seed=DEFAULTS["seed"],
    device=DEFAULTS["device"],
    model=DEFAULTS["model"],
    time_dim=DEFAULTS["time_dim"],
    heads=DEFAULTS["heads"],
    dropout=DEFAULTS["dropout"],
    threads=DEFAULTS["threads"],
):
    settings = resolve(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        fanout=fanout,
        hidden=hidden,
        lr=lr,
        seed=seed,
        device=device,
        time_dim=time_dim,
        heads=heads,
        dropout=dropout,
        threads=threads,
    )
    model_name = settings["model"]
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
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    architecture = dict(
        fanout=fanout, hidden=hidden, time_dim=time_dim, heads=heads, dropout=dropout
    )
    predictor = TGNPredictor if model_name == "tgn" else Predictor
    model = predictor(len(node_ids), features, **architecture).to(device)
    history = History(edges, timestamps, len(node_ids))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    print(
        f"Training {model_name.upper()}: {len(edges)} events, {len(node_ids)} nodes, time={kind}, device={device}.",
        file=sys.stderr,
    )
    for epoch in range(epochs):
        model.train()
        if isinstance(model, TGNPredictor):
            model.reset_memory(float(timestamps[0]))
        total = 0.0
        for group_start, group_end in temporal_batches(timestamps, batch_size):
            # Bound attention memory even for very large equal-time groups.
            for start in range(group_start, group_end, batch_size):
                end = min(start + batch_size, group_end)
                batch = edges[start:end]
                ts = timestamps[start:end]
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
                labels = torch.cat(
                    (torch.ones(len(batch)), torch.zeros(len(batch)))
                ).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Training produced a nonfinite loss; check feature and timestamp scales."
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            if isinstance(model, TGNPredictor):
                model.observe(
                    edges[group_start:group_end], timestamps[group_start:group_end]
                )
        losses.append(total / len(edges))
        print(f"Epoch {epoch + 1}/{epochs}: loss={losses[-1]:.5f}", file=sys.stderr)
    if isinstance(model, TGNPredictor):
        # Reconstruct state with the final weights, not intermediate epoch weights.
        model.eval()
        model.reset_memory(float(timestamps[0]))
        for start, end in temporal_batches(timestamps, batch_size):
            model.observe(edges[start:end], timestamps[start:end])
    checkpoint = {
        "format_version": 2,
        "model": model_name,
        "node_ids": node_ids,
        "edges": torch.from_numpy(edges),
        "timestamps": torch.from_numpy(timestamps),
        "time_kind": kind,
        "last_time": float(timestamps[-1]),
        "config": architecture,
        "resolved_config": settings,
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
    version, kind = checkpoint.get("format_version"), checkpoint.get("model")
    if not (
        version == 1 and kind == "tgat" or version == 2 and kind in ("tgn", "tgat")
    ):
        raise ValueError("Unsupported FROST prediction checkpoint.")
    predictor = TGNPredictor if kind == "tgn" else Predictor
    model = predictor(
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
