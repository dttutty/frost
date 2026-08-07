from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from frost.neg_sampler import TGBEvalNegSampler
from frost.metrics import tgb_reciprocal_ranks


def _write_negative_file(
    path: Path, values: dict[tuple[int, int, int], np.ndarray]
) -> None:
    with path.open("wb") as f:
        pickle.dump(values, f)


def _make_sampler(
    tmp_path: Path,
    *,
    values: dict[tuple[int, int, int], np.ndarray],
    n_neg: int,
) -> TGBEvalNegSampler:
    dataset_dir = tmp_path / "tgbl-review"
    dataset_dir.mkdir()
    for split in ("val", "test"):
        _write_negative_file(dataset_dir / f"tgbl-review_{split}_ns_v2.pkl", values)
    return TGBEvalNegSampler(
        n_nodes=100,
        n_neg=n_neg,
        dataset="tgbl-review",
        data_root=tmp_path,
        seed=123,
    )


def test_tgb_sampler_loads_versioned_files_and_preserves_row_major_order(
    tmp_path: Path,
) -> None:
    sampler = _make_sampler(
        tmp_path,
        values={
            (1, 10, 100): np.array([20, 21]),
            (2, 11, 101): np.array([30, 31]),
        },
        n_neg=2,
    )
    sampler.set_batch_context(
        batch_src=np.array([1, 2]),
        batch_dst=np.array([10, 11]),
        batch_start_time=100,
        batch_end_time=101,
        batch_ts=np.array([100, 101]),
    )

    sampled = sampler.sample(torch.tensor([1, 2]), torch.tensor([1, 2]))

    assert torch.equal(sampled, torch.tensor([20, 21, 30, 31]))


def test_tgb_sampler_honors_explicit_dataset_version(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "tgbl-review-v1"
    dataset_dir.mkdir()
    for split in ("val", "test"):
        _write_negative_file(
            dataset_dir / f"tgbl-review_{split}_ns_v1.pkl",
            {(1, 10, 100): np.array([11, 12])},
        )
        _write_negative_file(
            dataset_dir / f"tgbl-review_{split}_ns_v2.pkl",
            {(1, 10, 100): np.array([21, 22])},
        )
    sampler = TGBEvalNegSampler(
        n_nodes=100,
        n_neg=2,
        dataset="tgbl-review-v1",
        data_root=tmp_path,
    )
    sampler.set_batch_context(
        batch_src=np.array([1]),
        batch_dst=np.array([10]),
        batch_start_time=100,
        batch_end_time=100,
        batch_ts=np.array([100]),
    )

    sampled = sampler.sample(torch.tensor([1]), torch.tensor([1]))

    assert torch.equal(sampled, torch.tensor([11, 12]))


def test_tgbl_wiki_sampler_uses_configured_official_raw_root(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "raw" / "tgbl_wiki"
    dataset_dir.mkdir(parents=True)
    values = {(1, 10, 100): np.array([20, 21])}
    for split in ("val", "test"):
        _write_negative_file(dataset_dir / f"tgbl-wiki_{split}_ns_v2.pkl", values)
    monkeypatch.setenv("TGB_DATA_ROOT", str(tmp_path))

    sampler = TGBEvalNegSampler(
        n_nodes=100,
        n_neg=2,
        dataset="TGBL_WIKI",
    )
    sampler.set_batch_context(
        batch_src=np.array([1]),
        batch_dst=np.array([10]),
        batch_start_time=100,
        batch_end_time=100,
        batch_ts=np.array([100]),
    )

    assert torch.equal(
        sampler.sample(torch.tensor([1]), torch.tensor([1])),
        torch.tensor([20, 21]),
    )


def test_tgb_sampler_rejects_variable_width_candidates(tmp_path: Path) -> None:
    sampler = _make_sampler(
        tmp_path,
        values={(1, 10, 100): np.array([20, 21, 22])},
        n_neg=2,
    )
    sampler.set_batch_context(
        batch_src=np.array([1]),
        batch_dst=np.array([10]),
        batch_start_time=100,
        batch_end_time=100,
        batch_ts=np.array([100]),
    )

    with pytest.raises(ValueError, match="fixed-width"):
        sampler.sample(torch.tensor([1]), torch.tensor([1]))


def test_tgb_sampler_pads_variable_width_and_exposes_mask(tmp_path: Path) -> None:
    sampler = _make_sampler(
        tmp_path,
        values={(1, 10, 100): np.array([20, 21])},
        n_neg=3,
    )
    sampler.allow_variable_width = True
    sampler.set_batch_context(
        batch_src=np.array([1]),
        batch_dst=np.array([10]),
        batch_start_time=100,
        batch_end_time=100,
        batch_ts=np.array([100]),
    )

    sampled = sampler.sample(torch.tensor([1]), torch.tensor([1]))

    assert torch.equal(sampled, torch.tensor([20, 21, -1]))
    assert torch.equal(
        sampler.get_last_negative_mask(), torch.tensor([True, True, False])
    )


def test_tgb_sampler_never_falls_back_to_random(tmp_path: Path) -> None:
    sampler = _make_sampler(
        tmp_path,
        values={(1, 10, 100): np.array([20, 21])},
        n_neg=2,
    )
    sampler.set_batch_context(
        batch_src=np.array([9]),
        batch_dst=np.array([99]),
        batch_start_time=999,
        batch_end_time=999,
        batch_ts=np.array([999]),
    )

    with pytest.raises(KeyError, match="Refusing to use random fallback"):
        sampler.sample(torch.tensor([9]), torch.tensor([1]))


def test_tgb_reciprocal_ranks_use_average_tie_rank() -> None:
    positives = torch.tensor([0.5, 0.5])
    negatives = torch.tensor(
        [
            [0.6, 0.5, 0.4],
            [0.6, 0.4, 0.3],
        ]
    )

    reciprocal_ranks = tgb_reciprocal_ranks(positives, negatives)

    assert torch.allclose(reciprocal_ranks, torch.tensor([0.4, 0.5]))


def test_tgb_reciprocal_ranks_ignore_padded_candidates() -> None:
    positives = torch.tensor([0.5])
    negatives = torch.tensor([[0.6, 0.4, 100.0]])
    valid_mask = torch.tensor([[True, True, False]])

    reciprocal_ranks = tgb_reciprocal_ranks(positives, negatives, valid_mask)

    assert torch.allclose(reciprocal_ranks, torch.tensor([0.5]))
