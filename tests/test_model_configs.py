from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import open_dict

from frost.launch.factory import build_worker_modules
from frost.launch.plan import compile_execution_plan
from frost.models.recipes import RECIPE_TYPES, instantiate_model_recipe
from frost.launcher import _instantiate_negative_samplers
from frost_bench.cli.train import PRESETS


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


@pytest.mark.parametrize("preset", [None, *sorted(RECIPE_TYPES)])
def test_default_negative_sampling_needs_no_artifact_and_draws_all_nodes(preset) -> None:
    overrides = [] if preset is None else list(PRESETS[preset].hydra_overrides)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    frame = pd.DataFrame({"src": [0] * 128, "dst": [7] * 128})
    train, evaluation = _instantiate_negative_samplers(
        sampler_cfg=cfg.neg_sampler,
        dataframe=frame,
        n_nodes=8,
        dataset="MISSING_DATASET",
        train_n_neg=1,
        eval_n_neg=1,
        train_only=False,
    )

    for sampler in (train, evaluation):
        assert torch.equal(sampler.candidates, torch.arange(8))
        src = torch.zeros(128, dtype=torch.long)
        eid = torch.arange(1, 129)
        first = sampler.sample(src, eid)
        second = sampler.sample(src, eid)
        assert first.shape == (128,)
        assert bool(((first >= 0) & (first < 8)).all())
        assert not torch.equal(first, second)


@pytest.mark.parametrize("name", sorted(RECIPE_TYPES))
def test_model_configs_compose_and_build(name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"model={name}"])
    with open_dict(cfg):
        cfg.ndim = 0
        cfg.edim = 0

    recipe = instantiate_model_recipe(cfg.model)
    plan = compile_execution_plan(recipe)
    d_mem = int(recipe.state["updater"]["d_mem"]) if recipe.state else 0
    _, encoder, head, _ = build_worker_modules(
        cfg,
        recipe=recipe,
        plan=plan,
        d_mail=2 * d_mem,
    )

    assert encoder.__class__.__name__.lower().replace("_", "") in name.replace("_", "")
    assert head is not None


def test_compute_policy_is_recorded_without_a_precision_config_group() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")

    assert cfg.compute.tf32 is False
    assert cfg.compute.amp is False
    assert "precision" not in cfg
    assert cfg.cpu_threads == "auto"
    assert cfg.runtime.single_gpu_resident == "off"


@pytest.mark.parametrize(
    "legacy_override",
    [
        "encoder=tgn",
        "mem_updater=gru",
        "predictor=link",
        "access_analyzer=memory_analyzer",
        "loss_fn=bce",
        "model=dyrep",
    ],
)
def test_removed_component_config_groups_fail(legacy_override: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        with pytest.raises(ConfigCompositionException):
            compose(config_name="config", overrides=[legacy_override])
