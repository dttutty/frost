"""Compile model declarations into immutable runtime execution plans."""

from __future__ import annotations

from dataclasses import dataclass

from frost.models.contracts import (
    DynamicStateContract,
    NodeSet,
    RepresentationKind,
    RootExpansion,
)
from frost.models.recipes import ModelRecipe


@dataclass(frozen=True)
class SamplingPlan:
    fanout: tuple[int, ...]
    root_expansion: RootExpansion


@dataclass(frozen=True)
class ExecutionPlan:
    model_name: str
    embedding_dim: int
    sampling: SamplingPlan
    node_feature_scope: NodeSet
    dynamic_state: DynamicStateContract
    representation: RepresentationKind

    @property
    def has_dynamic_state(self) -> bool:
        from frost.models.contracts import MemoryState

        return isinstance(self.dynamic_state, MemoryState)


def _normalize_fanout(raw: object) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, int):
        return (raw,)
    if isinstance(raw, list | tuple):
        return tuple(int(value) for value in raw)
    raise TypeError(f"model.neighborhood.fanout must be an integer or list, got {raw!r}")


def compile_execution_plan(
    recipe: ModelRecipe,
) -> ExecutionPlan:
    """Resolve and validate a model recipe without constructing torch modules."""

    fanout = _normalize_fanout(recipe.neighborhood.get("fanout"))
    contract = recipe.contract
    contract.neighborhood.validate(fanout)
    sampling = SamplingPlan(
        fanout=fanout,
        root_expansion=contract.neighborhood.expansion,
    )
    return ExecutionPlan(
        model_name=recipe.name,
        embedding_dim=recipe.embedding_dim,
        sampling=sampling,
        node_feature_scope=contract.node_feature_scope,
        dynamic_state=contract.dynamic_state,
        representation=contract.representation,
    )
