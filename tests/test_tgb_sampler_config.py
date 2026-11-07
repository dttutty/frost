from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from frost.launch.config import resolve_tgb_negative_sampling
from frost.launcher import _instantiate_negative_samplers


ROOT = Path(__file__).resolve().parents[1]


def _config(dataset: str, *overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                "neg_sampler=rnd_dst_then_tgb",
                f"dataset={dataset}",
                "eval_metric=mrr",
                *overrides,
            ],
        )


@pytest.mark.parametrize(
    "dataset,capacity,variable_width",
    [
        ("TGBL_WIKI", 999, True),
        ("TGBL_REVIEW", 100, False),
        ("TGBL_COIN", 20, False),
        ("TGBL_COMMENT", 20, False),
        ("TGBL_FLIGHT", 20, False),
    ],
)
def test_auto_config_builds_official_sampler_and_masks(
    tmp_path, monkeypatch, dataset, capacity, variable_width
):
    cfg = _config(dataset)
    resolve_tgb_negative_sampling(cfg)
    assert cfg.neg_sampler.train.n_neg == 1
    assert cfg.neg_sampler.eval.n_neg == capacity
    assert cfg.neg_sampler.eval.allow_variable_width is variable_width

    directory = tmp_path / "raw" / dataset.lower()
    directory.mkdir(parents=True)
    monkeypatch.setenv("TGB_DATA_ROOT", str(tmp_path))
    available = capacity - int(variable_width)
    values = {
        (1, 10, 100): np.arange(20, 20 + available),
        (2, 11, 101): np.arange(20, 20 + capacity),
    }
    version = "" if dataset == "TGBL_COMMENT" else "_v2"
    for split in ("val", "test"):
        path = directory / f"{dataset.lower().replace('_', '-')}_{split}_ns{version}.pkl"
        with path.open("wb") as f:
            pickle.dump(values, f)

    train, evaluation = _instantiate_negative_samplers(
        sampler_cfg=cfg.neg_sampler,
        dataframe=pd.DataFrame({"src": [1, 2], "dst": [10, 11]}),
        n_nodes=1200,
        dataset=cfg.dataset,
        train_n_neg=cfg.neg_sampler.train.n_neg,
        eval_n_neg=cfg.neg_sampler.eval.n_neg,
        train_only=False,
    )
    assert train.n_neg == 1
    evaluation.set_batch_context(
        batch_src=np.array([1, 2]),
        batch_dst=np.array([10, 11]),
        batch_start_time=100,
        batch_end_time=101,
        batch_ts=np.array([100, 101]),
    )
    result = evaluation.sample(torch.tensor([1, 2]), torch.tensor([1, 2]))
    mask = evaluation.get_last_negative_mask()
    assert result.shape == (2 * capacity,)
    assert mask.sum().item() == available + capacity
    assert torch.equal(result[:available], torch.arange(20, 20 + available))
    if variable_width:
        assert result[capacity - 1] == -1
        assert not mask[capacity - 1]
    else:
        assert mask.all()


def test_hyphenated_name_is_normalized_for_raw_dataset_loading():
    cfg = _config("tgbl-review")
    resolve_tgb_negative_sampling(cfg)
    resolve_tgb_negative_sampling(cfg)  # run.py and direct launcher entry are safe.
    assert cfg.dataset == "TGBL_REVIEW"
    assert cfg.neg_sampler.eval.n_neg == 100


@pytest.mark.parametrize("dataset", ["WIKI", "LASTFM", "tgbl-unknown", "TGBN_TRADE"])
def test_non_supported_datasets_fail_before_initializing_distributed(
    monkeypatch, dataset
):
    import frost.launcher as launcher

    def forbidden():
        pytest.fail("Invalid TGB configuration reached distributed initialization")

    monkeypatch.setattr(launcher, "initialize_process_group", forbidden)
    with pytest.raises(ValueError, match="expected one of TGBL_WIKI"):
        launcher.launch(_config(dataset))


def test_run_entry_rejects_invalid_dataset_before_data_or_gpu_setup(monkeypatch):
    from frost.data import paths

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid TGB configuration reached dataset setup")

    monkeypatch.setattr(paths, "static_features_present_on_disk", forbidden)
    spec = importlib.util.spec_from_file_location("frost_run_tgb_test", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Unsupported official TGB dataset"):
        module.main.__wrapped__(_config("LASTFM"))


@pytest.mark.parametrize(
    "dataset,override",
    [
        ("TGBL_REVIEW", "neg_sampler.eval.n_neg=20"),
        ("TGBL_WIKI", "neg_sampler.eval.n_neg=100"),
        ("TGBL_WIKI", "neg_sampler.eval.allow_variable_width=false"),
        ("TGBL_COIN", "neg_sampler.eval.allow_variable_width=true"),
    ],
)
def test_conflicting_official_settings_are_rejected(dataset, override):
    with pytest.raises(ValueError, match="Official .* evaluation requires"):
        resolve_tgb_negative_sampling(_config(dataset, override))


def test_matching_explicit_settings_remain_compatible():
    cfg = _config("TGBL_REVIEW", "neg_sampler.eval.n_neg=100")
    resolve_tgb_negative_sampling(cfg)
    assert cfg.neg_sampler.eval.n_neg == 100


@pytest.mark.parametrize("dataset,capacity", [("TGBL_WIKI", 999), ("TGBL_REVIEW", 100)])
def test_old_wiki_name_is_a_compatible_automatic_alias(dataset, capacity):
    cfg = _config(dataset, "neg_sampler=rnd_dst_then_tgb_wiki")
    resolve_tgb_negative_sampling(cfg)
    assert cfg.neg_sampler.eval.n_neg == capacity
    assert cfg.neg_sampler.train.candidate_mode == "dst_partition"


def test_non_tgb_sampler_is_unchanged():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config")
    before = OmegaConf.to_container(cfg)
    resolve_tgb_negative_sampling(cfg)
    assert OmegaConf.to_container(cfg) == before
