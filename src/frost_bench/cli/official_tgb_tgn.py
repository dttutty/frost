"""Run TGB's PyG TGN baseline under a reproducible MRR parity protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from frost.data.paths import tgb_data_root
from frost.runtime._early_stopping import is_significant_improvement


SUPPORTED_DATASETS = (
    "tgbl-wiki",
    "tgbl-coin",
    "tgbl-comment",
    "tgbl-flight",
    "tgbl-review",
)
OPTIMIZED_SPEED_ENV = "FROST_OPTIMIZED_SPEED"


@dataclass(frozen=True)
class _NegativeGroup:
    indices: torch.Tensor
    destinations: torch.Tensor
    width: int


_NegativeCache = list[tuple[_NegativeGroup, ...]]


@dataclass(frozen=True)
class _EvaluationResult:
    mrr: float
    queries: int
    scored_pairs: int
    batches: int
    measured_queries: int
    measured_scored_pairs: int
    measured_batches: int
    measured_elapsed_s: float

    @property
    def queries_per_sec(self) -> float:
        return self.measured_queries / self.measured_elapsed_s

    @property
    def scored_pairs_per_sec(self) -> float:
        return self.measured_scored_pairs / self.measured_elapsed_s


def _tgb_fast_path_enabled() -> bool:
    return os.environ.get(OPTIMIZED_SPEED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _make_negative_groups(
    neg_lists: Sequence[Sequence[int]], device: torch.device
) -> tuple[_NegativeGroup, ...]:
    widths = np.fromiter((len(row) for row in neg_lists), dtype=np.int64)
    groups = []
    for width_value in np.unique(widths):
        group_np = np.flatnonzero(widths == width_value)
        groups.append(
            _NegativeGroup(
                indices=torch.as_tensor(group_np, device=device),
                destinations=torch.as_tensor(
                    np.stack([neg_lists[int(i)] for i in group_np]), device=device
                ),
                width=int(width_value),
            )
        )
    return tuple(groups)


class CorrectedLinkPredictor(torch.nn.Module):
    """TGB's link head without the erroneous training-time sigmoid."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_final = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        hidden = self.lin_src(z_src) + self.lin_dst(z_dst)
        return self.lin_final(hidden.relu())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TGB official PyG TGN with official negative candidates and MRR",
        epilog=(
            f"Set {OPTIMIZED_SPEED_ENV}=1 to opt into cached official evaluation "
            "negatives and synchronization reductions. The default is disabled."
        ),
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--memory-dim", type=int, default=100)
    parser.add_argument("--time-dim", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--loss-mode",
        choices=("corrected", "official-exact"),
        default="corrected",
        help=(
            "corrected uses raw logits with BCEWithLogitsLoss; official-exact "
            "retains TGB's sigmoid followed by BCEWithLogitsLoss"
        ),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Benchmark-only train batch cap; 0 processes the full split.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="Benchmark-only validation batch cap; 0 processes the full split.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=0,
        help="Benchmark-only test batch cap; 0 processes the full split.",
    )
    parser.add_argument(
        "--test-throughput-warmup-batches",
        type=int,
        default=0,
        help="Exclude this many initial test batches from throughput timing.",
    )
    parser.add_argument(
        "--train-throughput-warmup-batches",
        type=int,
        default=0,
        help="Exclude this many initial train batches from throughput timing.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def _load_official_modules() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    tgb_root = repo_root / "third_party" / "TGB"
    sys.path.insert(0, str(tgb_root))
    try:
        import tgb.linkproppred.dataset as dataset_module
        from modules.decoder import LinkPredictor
        from modules.emb_module import GraphAttentionEmbedding
        from modules.memory_module import TGNMemory
        from modules.msg_func import IdentityMessage
        from modules.neighbor_loader import LastNeighborLoader
        from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
        from tgb.linkproppred.evaluate import Evaluator
        from tgb.utils.utils import set_random_seed
        from torch_geometric.loader import TemporalDataLoader
        from torch_geometric.nn.models.tgn import LastAggregator
    except ImportError as exc:
        raise RuntimeError(
            "The official TGB TGN runner needs the optional tgb dependency group; "
            "run `uv sync --locked --group tgb`."
        ) from exc

    # LinkPropPredDataset prepends this module global to root even when root is
    # absolute. Emptying it lets the benchmark reuse TGB_DATA_ROOT/raw directly.
    dataset_module.PROJ_DIR = ""
    return {
        "Evaluator": Evaluator,
        "GraphAttentionEmbedding": GraphAttentionEmbedding,
        "IdentityMessage": IdentityMessage,
        "LastAggregator": LastAggregator,
        "LastNeighborLoader": LastNeighborLoader,
        "LinkPredictor": LinkPredictor,
        "PyGLinkPropPredDataset": PyGLinkPropPredDataset,
        "TGNMemory": TGNMemory,
        "TemporalDataLoader": TemporalDataLoader,
        "set_random_seed": set_random_seed,
    }


def _evaluate(
    *,
    loader: Any,
    split: str,
    model: dict[str, Any],
    neighbor_loader: Any,
    neg_sampler: Any,
    evaluator: Any,
    data: Any,
    assoc: torch.Tensor,
    device: torch.device,
    negative_cache: _NegativeCache | None = None,
    max_batches: int = 0,
    throughput_warmup_batches: int = 0,
) -> _EvaluationResult:
    for component in model.values():
        component.eval()
    reciprocal_rank_sum = 0.0
    num_queries = 0
    scored_pairs = 0
    batches = 0
    measured_queries = 0
    measured_scored_pairs = 0
    measured_batches = 0
    measurement_start: float | None = None
    with torch.no_grad():
        for batch_index, pos_batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            if batch_index == throughput_warmup_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                measurement_start = time.perf_counter()
            pos_batch = pos_batch.to(device)
            if negative_cache is not None and batch_index < len(negative_cache):
                negative_groups = negative_cache[batch_index]
            else:
                neg_lists = neg_sampler.query_batch(
                    pos_batch.src,
                    pos_batch.dst,
                    pos_batch.t,
                    split_mode=split,
                )
                negative_groups = _make_negative_groups(neg_lists, device)
                if negative_cache is not None:
                    if batch_index != len(negative_cache):
                        raise RuntimeError("evaluation negative cache is out of order")
                    negative_cache.append(negative_groups)

            for negative_group in negative_groups:
                group = negative_group.indices
                width = negative_group.width
                src_pos = pos_batch.src[group]
                dst_pos = pos_batch.dst[group]
                neg_dst = negative_group.destinations
                candidates = torch.cat((dst_pos[:, None], neg_dst), dim=1)
                repeated_src = src_pos[:, None].expand_as(candidates)
                flat_src = repeated_src.reshape(-1)
                flat_dst = candidates.reshape(-1)

                n_id = torch.cat((flat_src, flat_dst)).unique()
                n_id, edge_index, e_id = neighbor_loader(n_id)
                assoc[n_id] = torch.arange(n_id.numel(), device=device)
                z, last_update = model["memory"](n_id)
                z = model["gnn"](
                    z,
                    last_update,
                    edge_index,
                    data.t[e_id],
                    data.msg[e_id],
                )
                scores = model["link_pred"](
                    z[assoc[flat_src]], z[assoc[flat_dst]]
                ).reshape(group.numel(), width + 1)
                metric = evaluator.eval(
                    {
                        "y_pred_pos": scores[:, 0],
                        "y_pred_neg": scores[:, 1:],
                        "eval_metric": ["mrr"],
                    }
                )["mrr"]
                reciprocal_rank_sum += float(metric) * group.numel()
                num_queries += group.numel()
                group_scored_pairs = group.numel() * (width + 1)
                scored_pairs += group_scored_pairs
                if batch_index >= throughput_warmup_batches:
                    measured_queries += group.numel()
                    measured_scored_pairs += group_scored_pairs

            model["memory"].update_state(
                pos_batch.src, pos_batch.dst, pos_batch.t, pos_batch.msg
            )
            neighbor_loader.insert(pos_batch.src, pos_batch.dst)
            batches += 1
            if batch_index >= throughput_warmup_batches:
                measured_batches += 1
    if num_queries == 0:
        raise RuntimeError(f"{split} evaluation produced no queries")
    if measured_batches == 0 or measurement_start is None:
        raise RuntimeError(
            f"{split} throughput warmup consumed every processed batch: "
            f"warmup={throughput_warmup_batches}, processed={batches}"
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_elapsed_s = time.perf_counter() - measurement_start
    return _EvaluationResult(
        mrr=reciprocal_rank_sum / num_queries,
        queries=num_queries,
        scored_pairs=scored_pairs,
        batches=batches,
        measured_queries=measured_queries,
        measured_scored_pairs=measured_scored_pairs,
        measured_batches=measured_batches,
        measured_elapsed_s=measured_elapsed_s,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch size, and patience must be positive")
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    batch_limits = (
        args.max_train_batches,
        args.max_val_batches,
        args.max_test_batches,
        args.train_throughput_warmup_batches,
        args.test_throughput_warmup_batches,
    )
    if any(value < 0 for value in batch_limits):
        raise ValueError("batch caps and throughput warmup must be non-negative")
    if (
        args.max_train_batches > 0
        and args.train_throughput_warmup_batches >= args.max_train_batches
    ):
        raise ValueError("train throughput warmup must be smaller than max train batches")
    if (
        args.max_test_batches > 0
        and args.test_throughput_warmup_batches >= args.max_test_batches
    ):
        raise ValueError("test throughput warmup must be smaller than max test batches")
    eval_batch_size = args.eval_batch_size or args.batch_size
    fast_path = _tgb_fast_path_enabled()
    modules = _load_official_modules()
    set_random_seed = modules["set_random_seed"]
    set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_root = (args.data_root or tgb_data_root()) / "raw"
    dataset = modules["PyGLinkPropPredDataset"](
        name=args.dataset,
        root=str(raw_root.resolve()),
        download=False,
    )
    if args.prepare_only:
        summary = {
            "system": "tgb_official_tgn",
            "dataset": args.dataset,
            "prepared": True,
            "num_nodes": dataset.num_nodes,
            "num_edges": dataset.num_edges,
        }
        print(
            "TGB_OFFICIAL_TGN_PREPARED " + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        return 0
    dataset.load_val_ns()
    dataset.load_test_ns()
    data = dataset.get_TemporalData().to(device)
    train_data = data[dataset.train_mask]
    val_data = data[dataset.val_mask]
    test_data = data[dataset.test_mask]
    loader_cls = modules["TemporalDataLoader"]
    train_loader = loader_cls(train_data, batch_size=args.batch_size)
    val_loader = loader_cls(val_data, batch_size=eval_batch_size)
    test_loader = loader_cls(test_data, batch_size=eval_batch_size)

    neighbor_loader = modules["LastNeighborLoader"](
        data.num_nodes, size=args.neighbors, device=device
    )
    memory = modules["TGNMemory"](
        data.num_nodes,
        data.msg.size(-1),
        args.memory_dim,
        args.time_dim,
        message_module=modules["IdentityMessage"](
            data.msg.size(-1), args.memory_dim, args.time_dim
        ),
        aggregator_module=modules["LastAggregator"](),
    ).to(device)
    gnn = modules["GraphAttentionEmbedding"](
        in_channels=args.memory_dim,
        out_channels=args.embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)
    predictor_cls = (
        CorrectedLinkPredictor
        if args.loss_mode == "corrected"
        else modules["LinkPredictor"]
    )
    link_pred = predictor_cls(args.embedding_dim).to(device)
    model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}
    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=args.lr,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
    evaluator = modules["Evaluator"](name=args.dataset)
    neg_sampler = dataset.negative_sampler
    val_negative_cache: _NegativeCache | None = [] if fast_path else None
    test_negative_cache: _NegativeCache | None = [] if fast_path else None
    if fast_path:
        negative_dst_low = int(data.dst.min())
        negative_dst_high = int(data.dst.max()) + 1

    rows: list[dict[str, Any]] = []
    best_val_mrr = float("-inf")
    best_row: dict[str, Any] | None = None
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        for component in model.values():
            component.train()
        memory.reset_state()
        neighbor_loader.reset_state()
        loss_sum: float | torch.Tensor
        if fast_path:
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        else:
            loss_sum = 0.0
        train_start = time.perf_counter()
        train_events = 0
        train_batches = 0
        train_measured_events = 0
        train_measured_batches = 0
        train_measurement_start: float | None = None
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                break
            if batch_index == args.train_throughput_warmup_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                train_measurement_start = time.perf_counter()
            batch = batch.to(device)
            optimizer.zero_grad()
            if fast_path:
                low = negative_dst_low
                high = negative_dst_high
            else:
                low = int(data.dst.min())
                high = int(data.dst.max()) + 1
            neg_dst = torch.randint(
                low,
                high,
                (batch.src.numel(),),
                dtype=torch.long,
                device=device,
            )
            n_id = torch.cat((batch.src, batch.dst, neg_dst)).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.numel(), device=device)
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
            pred_pos = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
            pred_neg = link_pred(z[assoc[batch.src]], z[assoc[neg_dst]])
            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)
            loss.backward()
            optimizer.step()
            memory.detach()
            if fast_path:
                loss_sum += loss.detach().to(dtype=torch.float64) * batch.num_events
            else:
                loss_sum += float(loss.detach()) * batch.num_events
            train_events += batch.num_events
            train_batches += 1
            if batch_index >= args.train_throughput_warmup_batches:
                train_measured_events += batch.num_events
                train_measured_batches += 1
        if train_measurement_start is None or train_measured_batches == 0:
            raise RuntimeError(
                "train throughput warmup consumed every processed batch: "
                f"warmup={args.train_throughput_warmup_batches}, "
                f"processed={train_batches}"
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_measured_time = time.perf_counter() - train_measurement_start
        train_time = time.perf_counter() - train_start
        if fast_path:
            loss_sum = float(loss_sum)

        val_start = time.perf_counter()
        val_result = _evaluate(
            loader=val_loader,
            split="val",
            model=model,
            neighbor_loader=neighbor_loader,
            neg_sampler=neg_sampler,
            evaluator=evaluator,
            data=data,
            assoc=assoc,
            device=device,
            negative_cache=val_negative_cache,
            max_batches=args.max_val_batches,
        )
        val_time = time.perf_counter() - val_start
        test_start = time.perf_counter()
        test_result = _evaluate(
            loader=test_loader,
            split="test",
            model=model,
            neighbor_loader=neighbor_loader,
            neg_sampler=neg_sampler,
            evaluator=evaluator,
            data=data,
            assoc=assoc,
            device=device,
            negative_cache=test_negative_cache,
            max_batches=args.max_test_batches,
            throughput_warmup_batches=args.test_throughput_warmup_batches,
        )
        test_time = time.perf_counter() - test_start
        row = {
            "dataset": args.dataset,
            "epoch": epoch,
            "loss": loss_sum / train_events,
            "val_mrr": val_result.mrr,
            "test_mrr": test_result.mrr,
            "train_time_s": train_time,
            "train_events": train_events,
            "train_batches": train_batches,
            "train_throughput_warmup_batches": (
                args.train_throughput_warmup_batches
            ),
            "train_measured_events": train_measured_events,
            "train_measured_batches": train_measured_batches,
            "train_measured_time_s": train_measured_time,
            "train_events_per_sec": train_measured_events / train_measured_time,
            "val_time_s": val_time,
            "test_time_s": test_time,
            "test_queries": test_result.queries,
            "test_scored_pairs": test_result.scored_pairs,
            "test_batches": test_result.batches,
            "test_throughput_warmup_batches": args.test_throughput_warmup_batches,
            "test_measured_queries": test_result.measured_queries,
            "test_measured_scored_pairs": test_result.measured_scored_pairs,
            "test_measured_batches": test_result.measured_batches,
            "test_measured_time_s": test_result.measured_elapsed_s,
            "test_queries_per_sec": test_result.queries_per_sec,
            "test_scored_pairs_per_sec": test_result.scored_pairs_per_sec,
            "epoch_time_s": time.perf_counter() - epoch_start,
        }
        rows.append(row)
        print("TGB_OFFICIAL_TGN " + json.dumps(row, sort_keys=True), flush=True)
        if is_significant_improvement(
            val_result.mrr,
            best_val_mrr,
            args.tolerance,
        ):
            best_val_mrr = val_result.mrr
            best_row = row
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    assert best_row is not None

    summary = {
        "system": "tgb_official_tgn",
        "dataset": args.dataset,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "eval_batch_size": eval_batch_size,
        "loss_mode": args.loss_mode,
        "fast_path": fast_path,
        "patience": args.patience,
        "tolerance": args.tolerance,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_test_batches": args.max_test_batches,
        "train_throughput_warmup_batches": args.train_throughput_warmup_batches,
        "test_throughput_warmup_batches": args.test_throughput_warmup_batches,
        "epochs": rows,
        "best_epoch": best_row["epoch"],
        "best_val_mrr": best_row["val_mrr"],
        "best_test_mrr": best_row["test_mrr"],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("TGB_OFFICIAL_TGN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
