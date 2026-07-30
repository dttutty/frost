from __future__ import annotations

import pytest

from frost.launch.plan import DatasetSchema, compile_execution_plan
from frost.models.contracts import (
    MemoryState,
    NeighborhoodMailbox,
    RepresentationKind,
    RootExpansion,
    ScoringKind,
    Stateless,
)
from frost.models.recipes import RECIPE_TYPES, ModelRecipe


DEFAULT_FANOUT = {
    "tgn": [10],
    "jodie": [],
    "apan": [10],
    "tgat": [20, 20],
    "graph_mixer": [2000],
    "tcl": [20],
    "dygformer": [32],
    "cawn": [20, 20],
}


def _recipe(name: str) -> ModelRecipe:
    return RECIPE_TYPES[name](
        embedding_dim=128,
        neighborhood={"fanout": DEFAULT_FANOUT[name]},
        encoder={},
        state={} if name in {"tgn", "jodie", "apan"} else None,
        head={},
    )


@pytest.mark.parametrize("name", sorted(RECIPE_TYPES))
def test_all_model_recipes_compile(name: str) -> None:
    plan = compile_execution_plan(
        _recipe(name),
        DatasetSchema("EXAMPLE", 10, 4, 3),
    )
    assert plan.model_name == name
    assert plan.sampling.fanout == tuple(DEFAULT_FANOUT[name])


def test_state_and_representation_contracts_are_semantic() -> None:
    apan = compile_execution_plan(
        _recipe("apan"), DatasetSchema("EXAMPLE", 10, 4, 3)
    )
    dygformer = compile_execution_plan(
        _recipe("dygformer"), DatasetSchema("EXAMPLE", 10, 4, 3)
    )
    tgat = compile_execution_plan(
        _recipe("tgat"), DatasetSchema("EXAMPLE", 10, 4, 3)
    )

    assert apan.layout.root_expansion is RootExpansion.POSITIVE_ROOTS
    assert isinstance(apan.dynamic_state, MemoryState)
    assert isinstance(apan.dynamic_state.mailbox, NeighborhoodMailbox)
    assert dygformer.representation is RepresentationKind.PAIR_ALIGNED
    assert dygformer.scoring is ScoringKind.ADDITIVE
    assert isinstance(tgat.dynamic_state, Stateless)
    assert tgat.scoring is ScoringKind.CONCAT


def test_contract_rejects_wrong_hop_count() -> None:
    recipe = RECIPE_TYPES["apan"](
        embedding_dim=128,
        neighborhood={"fanout": [10, 5]},
        encoder={},
        state={},
        head={},
    )
    with pytest.raises(ValueError, match="requires 1..1 hops"):
        compile_execution_plan(recipe, DatasetSchema("EXAMPLE", 10, 4, 3))
