"""Multi-GPU TPNet training with a single causal RP state and TGB MRR.

TPNet's random-projection (RP) module is mutable temporal state, rather than a
normal DDP parameter. Every rank therefore reads the same pre-batch state;
rank zero performs the original global-batch update and broadcasts it before
the next batch. Gradients are synchronized by DDP.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

# TPNet enables deterministic algorithms for RP runs. cuBLAS requires this
# workspace setting before its first CUDA operation in that mode.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from frost.metrics import tgb_reciprocal_ranks

# Resolve this backend relative to its source file, rather than the installed
# ``frost_bench.paths`` package.  That keeps isolated worktrees usable with a
# shared virtual environment whose editable installation may point elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIC_TPNET_ROOT = REPO_ROOT / "third_party" / "TPNet"
TGB_TPNET_ROOT = REPO_ROOT / "third_party" / "TGB_TPNet"
TGB_ROOT = REPO_ROOT / "third_party" / "TGB"


class TPNetLinkModel(nn.Module):
    """Give TPNet's embedding/predictor pair a DDP-visible forward method."""

    def __init__(self, backbone: nn.Module, predictor: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor

    def forward(self, src, dst, times, negative_dst):
        positive_src, positive_dst = (
            self.backbone.compute_src_dst_node_temporal_embeddings(src, dst, times)
        )
        negative_src, negative_dst_embedding = (
            self.backbone.compute_src_dst_node_temporal_embeddings(
                src, negative_dst, times
            )
        )
        positive_logits = self.predictor(src, dst, positive_src, positive_dst).squeeze(
            -1
        )
        negative_logits = self.predictor(
            src, negative_dst, negative_src, negative_dst_embedding
        ).squeeze(-1)
        return positive_logits, negative_logits


def _launcher_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--report-mrr", action="store_true")
    parser.add_argument("--report-ap", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--metrics-json", type=Path, default=None)
    return parser.parse_known_args()


def _option_value(argv: list[str], option: str, default: str) -> str:
    """Read one upstream option without consuming its argument list."""
    prefix = f"{option}="
    for index, argument in enumerate(argv):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == option and index + 1 < len(argv):
            return argv[index + 1]
    return default


def _backend_root(dataset_name: str) -> Path:
    return TGB_TPNET_ROOT if dataset_name.startswith("tgbl-") else CLASSIC_TPNET_ROOT


def _tgb_negative_capacity(dataset_name: str) -> tuple[int, bool]:
    normalized = dataset_name.lower().replace("_", "-")
    if normalized == "tgbl-wiki":
        return 999, True
    if normalized == "tgbl-review":
        return 100, False
    if normalized in {"tgbl-coin", "tgbl-comment", "tgbl-flight"}:
        return 20, False
    raise ValueError(f"Unsupported TGB link-prediction dataset: {dataset_name}")


def _frost_tgb_dataset_name(dataset_name: str) -> str:
    return dataset_name.upper().replace("-", "_")


def _load_tgb_tpnet_data(data_type: Any, dataset_name: str) -> tuple[Any, ...]:
    """Adapt Frost's canonical raw TGB data to TPNet's padded layout."""
    from frost.data.paths import resolve_tgb_raw_dataset_dir
    from frost.data.tgb import load_tgb_features, load_tgb_frame

    frost_dataset_name = _frost_tgb_dataset_name(dataset_name)
    raw_dir = resolve_tgb_raw_dataset_dir(frost_dataset_name)
    frame = load_tgb_frame(raw_dir, frost_dataset_name)
    src = frame["src"].to_numpy(dtype=np.int64, copy=True) + 1
    dst = frame["dst"].to_numpy(dtype=np.int64, copy=True) + 1
    times = frame["time"].to_numpy(dtype=np.float64, copy=True)
    edge_ids = frame["eid"].to_numpy(dtype=np.int64, copy=True)
    split = frame["default_split"].to_numpy(dtype=np.int8, copy=False)
    labels = np.ones(len(frame), dtype=np.float32)

    num_nodes = int(max(src.max(), dst.max()))
    node_features = load_tgb_features(raw_dir, frost_dataset_name, "node")
    if node_features is None:
        node_features = np.zeros((num_nodes, 1), dtype=np.float64)
    else:
        node_features = np.asarray(node_features, dtype=np.float64)
    edge_features = load_tgb_features(raw_dir, frost_dataset_name, "edge")
    if edge_features is None:
        edge_features = np.zeros((len(frame), 1), dtype=np.float64)
    else:
        edge_features = np.asarray(edge_features, dtype=np.float64)
    if node_features.ndim == 1:
        node_features = node_features[:, np.newaxis]
    if edge_features.ndim == 1:
        edge_features = edge_features[:, np.newaxis]
    if node_features.shape[0] != num_nodes:
        raise ValueError(
            "TGB node feature rows do not match the observed node range: "
            f"{node_features.shape[0]} != {num_nodes}"
        )
    if edge_features.shape[0] != len(frame):
        raise ValueError(
            "TGB edge feature rows do not match the event table: "
            f"{edge_features.shape[0]} != {len(frame)}"
        )

    node_features = np.vstack(
        (np.zeros((1, node_features.shape[1]), dtype=np.float64), node_features)
    )
    edge_features = np.vstack(
        (np.zeros((1, edge_features.shape[1]), dtype=np.float64), edge_features)
    )

    def select(mask: np.ndarray) -> Any:
        return data_type(
            src_node_ids=src[mask],
            dst_node_ids=dst[mask],
            node_interact_times=times[mask],
            edge_ids=edge_ids[mask],
            labels=labels[mask],
        )

    full_mask = np.ones(len(frame), dtype=np.bool_)
    return (
        node_features,
        edge_features,
        select(full_mask),
        select(split == 0),
        select(split == 1),
        select(split == 2),
    )


def _evaluate_mrr(
    *,
    model: TPNetLinkModel,
    data: Any,
    data_loader: Any,
    neighbor_sampler: Any,
    negative_sampler: Any,
    random_projections: nn.Module | None,
    split: str,
) -> float:
    """Evaluate TPNet with Frost's official-negative and MRR implementations."""
    model.eval()
    backbone = cast(Any, model.backbone)
    predictor = cast(Any, model.predictor)
    backbone.set_neighbor_sampler(neighbor_sampler)
    negative_sampler.reset()
    reciprocal_rank_sum = 0.0
    query_count = 0

    with torch.no_grad():
        for batch_number, index_tensor in enumerate(data_loader):
            indices = index_tensor.numpy()
            src = data.src_node_ids[indices]
            dst = data.dst_node_ids[indices]
            times = data.node_interact_times[indices]
            edge_ids = data.edge_ids[indices]

            # TPNet reserves node zero for padding, while official TGB query
            # dictionaries use zero-based node IDs.
            negative_sampler.set_batch_context(
                batch_src=src - 1,
                batch_dst=dst - 1,
                batch_start_time=float(times[0]),
                batch_end_time=float(times[-1]),
                batch_ts=times,
            )
            negative_dst_tensor = negative_sampler.sample(
                torch.from_numpy(src - 1), torch.from_numpy(edge_ids)
            )
            negative_valid_mask = negative_sampler.get_last_negative_mask()
            n_neg = negative_sampler.n_neg
            negative_dst = (negative_dst_tensor.numpy() + 1).reshape(-1)
            negative_src = np.repeat(src, n_neg)
            negative_times = np.repeat(times, n_neg)

            positive_src_embedding, positive_dst_embedding = (
                backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src,
                    dst_node_ids=dst,
                    node_interact_times=times,
                )
            )
            negative_src_embedding, negative_dst_embedding = (
                backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=negative_src,
                    dst_node_ids=negative_dst,
                    node_interact_times=negative_times,
                )
            )
            positive_scores = predictor(
                src_node_ids=src,
                dst_node_ids=dst,
                src_node_embeddings=positive_src_embedding,
                dst_node_embeddings=positive_dst_embedding,
            ).squeeze(-1)
            negative_scores = predictor(
                src_node_ids=negative_src,
                dst_node_ids=negative_dst,
                src_node_embeddings=negative_src_embedding,
                dst_node_embeddings=negative_dst_embedding,
            ).squeeze(-1)

            if random_projections is not None:
                cast(Any, random_projections).update(
                    src_node_ids=src,
                    dst_node_ids=dst,
                    node_interact_times=times,
                )

            valid_mask = negative_valid_mask.view(len(src), n_neg).to(
                device=negative_scores.device
            )
            reciprocal_ranks = tgb_reciprocal_ranks(
                positive_scores,
                negative_scores.view(len(src), n_neg),
                valid_mask,
            )
            reciprocal_rank_sum += reciprocal_ranks.sum().item()
            query_count += len(src)
            if (batch_number + 1) % 100 == 0:
                print(
                    f"mrr_eval split={split} batch={batch_number + 1} "
                    f"queries={query_count}",
                    flush=True,
                )

    if not query_count:
        raise ValueError(f"No {split} queries were available for MRR evaluation")
    return reciprocal_rank_sum / query_count


def _evaluate_classic_ap(
    *,
    model: TPNetLinkModel,
    data: Any,
    data_loader: Any,
    neighbor_sampler: Any,
    negative_sampler: Any,
    random_projections: nn.Module | None,
    split: str,
) -> tuple[float, float]:
    """Evaluate classic TPNet with the upstream per-batch AP/AUC protocol."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    backbone = cast(Any, model.backbone)
    predictor = cast(Any, model.predictor)
    backbone.set_neighbor_sampler(neighbor_sampler)
    negative_sampler.reset_random_state()
    average_precisions: list[float] = []
    roc_aucs: list[float] = []

    with torch.no_grad():
        for index_tensor in data_loader:
            indices = index_tensor.numpy()
            src = data.src_node_ids[indices]
            dst = data.dst_node_ids[indices]
            times = data.node_interact_times[indices]
            _, negative_dst = negative_sampler.sample(size=len(src))

            positive_src_embedding, positive_dst_embedding = (
                backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src,
                    dst_node_ids=dst,
                    node_interact_times=times,
                )
            )
            negative_src_embedding, negative_dst_embedding = (
                backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src,
                    dst_node_ids=negative_dst,
                    node_interact_times=times,
                )
            )
            positive_scores = predictor(
                src_node_ids=src,
                dst_node_ids=dst,
                src_node_embeddings=positive_src_embedding,
                dst_node_embeddings=positive_dst_embedding,
            ).squeeze(-1)
            negative_scores = predictor(
                src_node_ids=src,
                dst_node_ids=negative_dst,
                src_node_embeddings=negative_src_embedding,
                dst_node_embeddings=negative_dst_embedding,
            ).squeeze(-1)

            if random_projections is not None:
                cast(Any, random_projections).update(
                    src_node_ids=src,
                    dst_node_ids=dst,
                    node_interact_times=times,
                )

            predictions = torch.cat((positive_scores, negative_scores)).sigmoid()
            labels = torch.cat(
                (torch.ones_like(positive_scores), torch.zeros_like(negative_scores))
            )
            predictions_numpy = predictions.cpu().numpy()
            labels_numpy = labels.cpu().numpy()
            average_precisions.append(
                float(average_precision_score(labels_numpy, predictions_numpy))
            )
            roc_aucs.append(float(roc_auc_score(labels_numpy, predictions_numpy)))

    if not average_precisions:
        raise ValueError(f"No {split} examples were available for AP evaluation")
    return float(np.mean(average_precisions)), float(np.mean(roc_aucs))


def _broadcast_rp_state(random_projections: nn.Module | None) -> None:
    if random_projections is None:
        return
    projections = getattr(random_projections, "random_projections")
    for index in range(len(projections)):
        projection = cast(torch.Tensor, projections[index])
        dist.broadcast(projection.data, src=0)
    now_time = cast(torch.Tensor, getattr(random_projections, "now_time"))
    dist.broadcast(now_time.data, src=0)


def _local_slice(size: int, rank: int, world_size: int) -> slice:
    quotient, remainder = divmod(size, world_size)
    start = rank * quotient + min(rank, remainder)
    return slice(start, start + quotient + (rank < remainder))


def _validate_batch_size(batch_size: int, world_size: int) -> None:
    if batch_size < world_size:
        raise ValueError(
            "--batch_size is global and must provide at least one event per rank: "
            f"got batch_size={batch_size}, world_size={world_size}"
        )


def _patch_python312_loader() -> None:
    """Keep the upstream TPNet loader working on Python 3.12.

    Upstream samples directly from a set, which Python 3.12 rejects. This is
    deliberately local to the external backend, leaving the submodule clean.
    """
    import utils.DataLoader as data_loader

    original_sample = data_loader.random.sample

    def sample(population: object, k: int):
        if isinstance(population, set):
            population = tuple(sorted(population))
        return original_sample(population, k)

    data_loader.random.sample = sample


def main() -> int:
    launch_args, tpnet_argv = _launcher_args()
    if launch_args.checkpoint_dir is not None:
        launch_args.checkpoint_dir = launch_args.checkpoint_dir.resolve()
    if launch_args.metrics_json is not None:
        launch_args.metrics_json = launch_args.metrics_json.resolve()
    dataset_name = _option_value(tpnet_argv, "--dataset_name", "wikipedia")
    is_tgb = dataset_name.startswith("tgbl-")
    tpnet_root = _backend_root(dataset_name)
    if not tpnet_root.is_dir():
        raise FileNotFoundError(
            f"TPNet submodule is unavailable: {tpnet_root}. "
            f"Run git submodule update --init {tpnet_root.relative_to(REPO_ROOT)}."
        )
    if is_tgb and not TGB_ROOT.is_dir():
        raise FileNotFoundError(
            f"TGB submodule is unavailable: {TGB_ROOT}. "
            "Run git submodule update --init third_party/TGB."
        )
    if launch_args.report_mrr and not is_tgb:
        raise ValueError("--report-mrr requires an official TGB dataset (tgbl-*)")
    if launch_args.report_ap and is_tgb:
        raise ValueError("--report-ap requires a classic dataset")
    if launch_args.report_ap and launch_args.report_mrr:
        raise ValueError("--report-ap and --report-mrr are mutually exclusive")
    if launch_args.report_mrr and launch_args.max_train_batches:
        raise ValueError("--report-mrr cannot be combined with --max-train-batches")
    if launch_args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size must be non-negative")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("TPNet DDP training requires CUDA")

    # TPNet's data loader intentionally uses paths relative to its own root.
    os.chdir(tpnet_root)
    module_roots = [str(tpnet_root)]
    if is_tgb:
        # The local TGB submodule provides the benchmark package without
        # requiring a second editable installation in Frost's environment.
        module_roots.append(str(TGB_ROOT))
    sys.path[:0] = module_roots
    sys.argv = [sys.argv[0], *tpnet_argv]

    from models.TPNet import RandomProjectionModule, TPNet
    from models.modules import LinkPredictor_v1
    from utils.DataLoader import Data, get_link_prediction_data, get_idx_data_loader
    from utils.load_configs import get_link_prediction_args
    from utils.utils import (
        NegativeEdgeSampler,
        create_optimizer,
        get_neighbor_sampler,
        set_random_seed,
        set_thread,
    )

    _patch_python312_loader()
    args = get_link_prediction_args(is_evaluation=False)
    if args.model_name != "TPNet":
        raise ValueError("frost-train tpnet only accepts --model_name TPNet")
    if args.batch_size < 1:
        raise ValueError("--batch_size is the global batch size and must be positive")
    if is_tgb and args.train_neg_num != 1:
        raise ValueError("Frost TPNet currently supports --train_neg_num 1")
    if is_tgb and args.train_loss_type != "pointwise":
        raise ValueError("Frost TPNet currently supports --train_loss_type pointwise")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    _validate_batch_size(args.batch_size, world_size)
    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"

    # Every rank creates the same global temporal batches. A rank differentiates
    # only its contiguous slice, while rank 0 retains the full slice for RP.
    set_random_seed(seed=0, deterministic_alg=args.use_random_projection)
    set_thread(3)
    logger = logging.getLogger("tpnet")
    full_data = val_data = test_data = None
    if is_tgb:
        (
            node_features,
            edge_features,
            full_data,
            train_data,
            val_data,
            test_data,
        ) = _load_tgb_tpnet_data(Data, args.dataset_name)
    else:
        (
            node_features,
            edge_features,
            full_data,
            train_data,
            val_data,
            test_data,
            *_unused_splits,
        ) = get_link_prediction_data(
            dataset_name=args.dataset_name,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            logger=logger,
            convert_time=args.use_random_projection,
        )
    train_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )
    train_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    if is_tgb:
        negative_sampler = NegativeEdgeSampler(
            src_node_ids=train_data.src_node_ids,
            dst_node_ids=train_data.dst_node_ids,
            interact_times=train_data.node_interact_times,
            last_observed_time=train_data.node_interact_times[0],
            negative_sample_strategy=args.train_negative_sample_strategy,
            seed=(
                None
                if args.train_negative_sample_strategy in {"random", "new_random"}
                else 0
            ),
        )
    else:
        negative_sampler = NegativeEdgeSampler(
            src_node_ids=train_data.src_node_ids,
            dst_node_ids=train_data.dst_node_ids,
        )

    random_projections = None
    if args.use_random_projection:
        random_projections = RandomProjectionModule(
            node_num=node_features.shape[0],
            edge_num=edge_features.shape[0],
            dim_factor=args.rp_dim_factor,
            num_layer=args.rp_num_layer,
            time_decay_weight=args.rp_time_decay_weight,
            device=args.device,
            use_matrix=args.rp_use_matrix,
            beginning_time=train_data.node_interact_times[0],
            not_scale=args.rp_not_scale,
            enforce_dim=args.enforce_dim,
        )
    backbone_kwargs = {
        "node_raw_features": node_features,
        "edge_raw_features": edge_features,
        "neighbor_sampler": train_sampler,
        "time_feat_dim": args.time_feat_dim,
        "random_projections": None if args.encode_not_rp else random_projections,
        "num_neighbors": args.num_neighbors,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "device": args.device,
    }
    if is_tgb:
        backbone_kwargs.update(
            output_dim=args.output_dim,
            not_embedding=args.not_embedding,
        )
    backbone = TPNet(**backbone_kwargs)
    embedding_dim = args.output_dim if is_tgb else node_features.shape[1]
    predictor = LinkPredictor_v1(
        input_dim1=embedding_dim,
        input_dim2=embedding_dim,
        hidden_dim=embedding_dim,
        output_dim=1,
        random_projections=None if args.decode_not_rp else random_projections,
        not_encode=args.not_encode,
    )
    module = TPNetLinkModel(backbone, predictor).to(args.device)
    model = DDP(module, device_ids=[local_rank], output_device=local_rank)
    _broadcast_rp_state(random_projections)
    optimizer = create_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()

    for epoch in range(args.num_epochs):
        model.train()
        model.module.backbone.set_neighbor_sampler(train_sampler)
        if random_projections is not None:
            if rank == 0:
                random_projections.reset_random_projections()
            _broadcast_rp_state(random_projections)

        for batch_number, index_tensor in enumerate(train_loader):
            global_indices = index_tensor.numpy()
            # All ranks must either participate in DDP or skip together.  A
            # final partial batch smaller than the world size cannot give
            # every rank a local example, so leave it out consistently.
            if len(global_indices) < world_size:
                if rank == 0:
                    print(
                        f"epoch={epoch + 1} batch={batch_number + 1} "
                        f"global_batch={len(global_indices)} skipped=undersized",
                        flush=True,
                    )
                continue
            partition = _local_slice(len(global_indices), rank, world_size)
            local_indices = global_indices[partition]
            if not len(local_indices):
                continue
            src = train_data.src_node_ids[local_indices]
            dst = train_data.dst_node_ids[local_indices]
            times = train_data.node_interact_times[local_indices]
            # Identical RNG state produces the same global negative batch on
            # both ranks; each then takes the corresponding data partition.
            if is_tgb:
                _, global_negative_dst = negative_sampler.sample(
                    size=len(global_indices),
                    batch_src_node_ids=train_data.src_node_ids[global_indices],
                    batch_dst_node_ids=train_data.dst_node_ids[global_indices],
                    current_batch_start_time=train_data.node_interact_times[
                        global_indices[0]
                    ],
                    current_batch_end_time=train_data.node_interact_times[
                        global_indices[-1]
                    ],
                )
            else:
                _, global_negative_dst = negative_sampler.sample(
                    size=len(global_indices)
                )
            negative_dst = global_negative_dst[partition]

            positive_logits, negative_logits = model(src, dst, times, negative_dst)
            logits = torch.cat((positive_logits, negative_logits))
            labels = torch.cat(
                (torch.ones_like(positive_logits), torch.zeros_like(negative_logits))
            )
            # DDP averages rank gradients. This restores BCE(mean) over the
            # original 200-event batch, including an uneven final batch.
            loss = loss_function(logits, labels) * (
                world_size * len(local_indices) / len(global_indices)
            )

            if random_projections is not None:
                if rank == 0:
                    random_projections.update(
                        train_data.src_node_ids[global_indices],
                        train_data.dst_node_ids[global_indices],
                        train_data.node_interact_times[global_indices],
                    )
                _broadcast_rp_state(random_projections)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(
                    f"epoch={epoch + 1} batch={batch_number + 1} "
                    f"global_batch={len(global_indices)} loss={loss.item():.6f}",
                    flush=True,
                )
            if (
                launch_args.max_train_batches
                and batch_number + 1 >= launch_args.max_train_batches
            ):
                break

        if launch_args.checkpoint_dir is not None and rank == 0:
            launch_args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                module.state_dict(),
                launch_args.checkpoint_dir / f"tpnet_epoch_{epoch + 1}.pt",
            )

    dist.barrier()
    if launch_args.report_ap and rank == 0:
        assert full_data is not None
        assert val_data is not None
        assert test_data is not None

        full_sampler = get_neighbor_sampler(
            data=full_data,
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor,
            seed=1,
        )
        eval_batch_size = launch_args.eval_batch_size or args.batch_size
        val_loader = get_idx_data_loader(
            indices_list=list(range(len(val_data.src_node_ids))),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        test_loader = get_idx_data_loader(
            indices_list=list(range(len(test_data.src_node_ids))),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        val_negative_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=0,
        )
        test_negative_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=2,
        )
        validation_ap, validation_auc = _evaluate_classic_ap(
            model=module,
            data=val_data,
            data_loader=val_loader,
            neighbor_sampler=full_sampler,
            negative_sampler=val_negative_sampler,
            random_projections=random_projections,
            split="val",
        )
        test_ap, test_auc = _evaluate_classic_ap(
            model=module,
            data=test_data,
            data_loader=test_loader,
            neighbor_sampler=full_sampler,
            negative_sampler=test_negative_sampler,
            random_projections=random_projections,
            split="test",
        )
        results = {
            "dataset": args.dataset_name,
            "checkpoint": "final_epoch",
            "evaluation_metric": "average_precision",
            "eval_batch_size": eval_batch_size,
            "val_ap": validation_ap,
            "val_auc": validation_auc,
            "test_ap": test_ap,
            "test_auc": test_auc,
        }
        print(
            f"TPNET_AP_SUMMARY {json.dumps(results, sort_keys=True)}",
            flush=True,
        )
        if launch_args.metrics_json is not None:
            launch_args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            launch_args.metrics_json.write_text(
                json.dumps(results, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    if launch_args.report_mrr and rank == 0:
        assert full_data is not None
        assert val_data is not None
        assert test_data is not None

        from frost.neg_sampler import TGBEvalNegSampler

        full_sampler = get_neighbor_sampler(
            data=full_data,
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor,
            seed=1,
        )
        n_eval_negatives, allow_variable_width = _tgb_negative_capacity(
            args.dataset_name
        )
        eval_negative_sampler = TGBEvalNegSampler(
            n_nodes=node_features.shape[0] - 1,
            n_neg=n_eval_negatives,
            dataset=_frost_tgb_dataset_name(args.dataset_name),
            allow_variable_width=allow_variable_width,
        )
        eval_batch_size = launch_args.eval_batch_size
        if not eval_batch_size:
            eval_batch_size = (
                20
                if args.dataset_name in {"tgbl-wiki", "tgbl-review"}
                else args.batch_size
            )
        val_loader = get_idx_data_loader(
            indices_list=list(range(len(val_data.src_node_ids))),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        test_loader = get_idx_data_loader(
            indices_list=list(range(len(test_data.src_node_ids))),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        validation_mrr = _evaluate_mrr(
            model=module,
            data=val_data,
            data_loader=val_loader,
            neighbor_sampler=full_sampler,
            negative_sampler=eval_negative_sampler,
            random_projections=random_projections,
            split="val",
        )
        test_mrr = _evaluate_mrr(
            model=module,
            data=test_data,
            data_loader=test_loader,
            neighbor_sampler=full_sampler,
            negative_sampler=eval_negative_sampler,
            random_projections=random_projections,
            split="test",
        )
        results = {
            "dataset": args.dataset_name,
            "checkpoint": "final_epoch",
            "evaluation_metric": "mrr",
            "val_mrr": validation_mrr,
            "test_mrr": test_mrr,
        }
        print(
            f"TPNET_MRR_SUMMARY {json.dumps(results, sort_keys=True)}",
            flush=True,
        )
        if launch_args.metrics_json is not None:
            launch_args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            launch_args.metrics_json.write_text(
                json.dumps(results, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
