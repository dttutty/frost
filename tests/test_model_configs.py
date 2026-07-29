from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import open_dict

from frost.launch.factory import build_worker_modules
from frost.launch.plan import DatasetSchema, compile_execution_plan
from frost.models.recipes import RECIPE_TYPES, instantiate_model_recipe


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


@pytest.mark.parametrize("name", sorted(RECIPE_TYPES))
def test_model_configs_compose_and_build(name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"model={name}"])
    with open_dict(cfg):
        cfg.ndim = 0
        cfg.edim = 0

    recipe = instantiate_model_recipe(cfg.model)
    plan = compile_execution_plan(recipe, DatasetSchema("EXAMPLE", 10, 0, 0))
    d_mem = int(recipe.state["updater"]["d_mem"]) if recipe.state else 0
    _, encoder, head, _, _ = build_worker_modules(
        cfg,
        recipe=recipe,
        plan=plan,
        d_mail=2 * d_mem,
        max_n_neg=1,
    )

    assert encoder.__class__.__name__.lower().replace("_", "") in name.replace(
        "_", ""
    )
    assert head is not None


@pytest.mark.parametrize(
    "legacy_override",
    ["encoder=tgn", "mem_updater=gru", "predictor=link", "access_analyzer=memory_analyzer"],
)
def test_removed_component_config_groups_fail(legacy_override: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        with pytest.raises(ConfigCompositionException):
            compose(config_name="config", overrides=[legacy_override])

