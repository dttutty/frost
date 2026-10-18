from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from frost_bench import tpnet_train


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--dataset_name", "tgbl-wiki"], "tgbl-wiki"),
        (["--dataset_name=tgbl-review"], "tgbl-review"),
        (["--batch_size", "200"], "wikipedia"),
    ],
)
def test_option_value_reads_upstream_arguments(argv: list[str], expected: str) -> None:
    assert tpnet_train._option_value(argv, "--dataset_name", "wikipedia") == expected


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("tgbl-wiki", (999, True)),
        ("TGBL_REVIEW", (100, False)),
        ("tgbl-coin", (20, False)),
        ("tgbl-comment", (20, False)),
        ("tgbl-flight", (20, False)),
    ],
)
def test_tgb_negative_capacity_matches_official_candidates(
    dataset: str, expected: tuple[int, bool]
) -> None:
    assert tpnet_train._tgb_negative_capacity(dataset) == expected


def test_tpnet_dataset_name_maps_to_frost_registry() -> None:
    assert tpnet_train._frost_tgb_dataset_name("tgbl-wiki") == "TGBL_WIKI"


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_local_slices_partition_global_batch(world_size: int) -> None:
    slices = [tpnet_train._local_slice(11, rank, world_size) for rank in range(world_size)]
    selected = [
        index
        for partition in slices
        for index in range(partition.start, partition.stop)
    ]

    assert selected == list(range(11))
    assert max(partition.stop - partition.start for partition in slices) - min(
        partition.stop - partition.start for partition in slices
    ) <= 1


def test_batch_size_must_supply_each_rank() -> None:
    tpnet_train._validate_batch_size(batch_size=4, world_size=4)

    with pytest.raises(ValueError, match="at least one event per rank"):
        tpnet_train._validate_batch_size(batch_size=3, world_size=4)


class _Data:
    def __init__(self, **values: np.ndarray) -> None:
        self.__dict__.update(values)


def test_load_tgb_tpnet_data_adds_padding_and_preserves_splits(
    tmp_path, monkeypatch
) -> None:
    from frost.data import paths, tgb

    frame = pd.DataFrame(
        {
            "eid": [1, 2, 3],
            "src": [0, 1, 0],
            "dst": [1, 0, 1],
            "time": [10, 20, 30],
            "default_split": [0, 1, 2],
        }
    )
    edge_features = np.array([[1.0], [2.0], [3.0]])
    monkeypatch.setattr(paths, "resolve_tgb_raw_dataset_dir", lambda _: tmp_path)
    monkeypatch.setattr(tgb, "load_tgb_frame", lambda *_: frame)
    monkeypatch.setattr(
        tgb,
        "load_tgb_features",
        lambda _root, _dataset, kind: None if kind == "node" else edge_features,
    )

    node_features, padded_edges, full, train, val, test = (
        tpnet_train._load_tgb_tpnet_data(_Data, "tgbl-wiki")
    )

    assert node_features.shape == (3, 1)
    assert np.array_equal(padded_edges[:, 0], np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(full.src_node_ids, np.array([1, 2, 1]))
    assert np.array_equal(train.edge_ids, np.array([1]))
    assert np.array_equal(val.edge_ids, np.array([2]))
    assert np.array_equal(test.edge_ids, np.array([3]))


class _Backbone(nn.Module):
    def set_neighbor_sampler(self, sampler: object) -> None:
        self.sampler = sampler

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del node_interact_times
        return (
            torch.from_numpy(src_node_ids).float().unsqueeze(1),
            torch.from_numpy(dst_node_ids).float().unsqueeze(1),
        )


class _Predictor(nn.Module):
    def forward(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        src_node_embeddings: torch.Tensor,
        dst_node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        del src_node_ids, dst_node_ids, src_node_embeddings
        return dst_node_embeddings


class _NegativeSampler:
    n_neg = 2

    def reset(self) -> None:
        self.context = None

    def set_batch_context(self, **context: object) -> None:
        self.context = context

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        del src_ids, edge_ids
        return torch.tensor([0, 1])

    def get_last_negative_mask(self) -> torch.Tensor:
        return torch.tensor([True, True])


def test_evaluate_mrr_uses_frost_sampler_and_removes_tpnet_padding() -> None:
    data = SimpleNamespace(
        src_node_ids=np.array([2], dtype=np.int64),
        dst_node_ids=np.array([11], dtype=np.int64),
        node_interact_times=np.array([100.0]),
        edge_ids=np.array([1], dtype=np.int64),
    )
    sampler = _NegativeSampler()
    model = tpnet_train.TPNetLinkModel(_Backbone(), _Predictor())

    mrr = tpnet_train._evaluate_mrr(
        model=model,
        data=data,
        data_loader=[torch.tensor([0])],
        neighbor_sampler=object(),
        negative_sampler=sampler,
        random_projections=None,
        split="val",
    )

    assert mrr == 1.0
    assert sampler.context is not None
    assert np.array_equal(sampler.context["batch_src"], np.array([1]))
    assert np.array_equal(sampler.context["batch_dst"], np.array([10]))


class _ClassicNegativeSampler:
    def reset_random_state(self) -> None:
        self.was_reset = True

    def sample(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        assert size == 2
        return np.array([1, 2]), np.array([1, 2])


def test_evaluate_classic_ap_matches_upstream_per_batch_protocol() -> None:
    data = SimpleNamespace(
        src_node_ids=np.array([1, 2], dtype=np.int64),
        dst_node_ids=np.array([3, 4], dtype=np.int64),
        node_interact_times=np.array([100.0, 200.0]),
    )
    sampler = _ClassicNegativeSampler()
    model = tpnet_train.TPNetLinkModel(_Backbone(), _Predictor())

    average_precision, roc_auc = tpnet_train._evaluate_classic_ap(
        model=model,
        data=data,
        data_loader=[torch.tensor([0, 1])],
        neighbor_sampler=object(),
        negative_sampler=sampler,
        random_projections=None,
        split="test",
    )

    assert sampler.was_reset
    assert average_precision == 1.0
    assert roc_auc == 1.0
