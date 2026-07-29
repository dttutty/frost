"""Compile model declarations into immutable runtime execution plans."""

from __future__ import annotations

from dataclasses import dataclass

from frost.models.contracts import (
    DynamicStateContract,
    FeatureContract,
    RepresentationKind,
    RootExpansion,
    ScoringKind,
)
from frost.models.recipes import ModelRecipe


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    n_nodes: int
    node_feature_dim: int
    edge_feature_dim: int


@dataclass(frozen=True)
class SamplingPlan:
    fanout: tuple[int, ...]
    root_expansion: RootExpansion


@dataclass(frozen=True)
class LayoutPlan:
    fanout: tuple[int, ...]
    root_expansion: RootExpansion


@dataclass(frozen=True)
class ExecutionPlan:
    model_name: str
    embedding_dim: int
    sampling: SamplingPlan
    layout: LayoutPlan
    features: FeatureContract
    dynamic_state: DynamicStateContract
    representation: RepresentationKind
    scoring: ScoringKind

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
    dataset: DatasetSchema,
) -> ExecutionPlan:
    """Resolve and validate a model recipe without constructing torch modules."""

    del dataset  # Dataset-dependent feature binding is added by the launch migration.
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
        layout=LayoutPlan(
            fanout=fanout,
            root_expansion=contract.neighborhood.expansion,
        ),
        features=contract.features,
        dynamic_state=contract.dynamic_state,
        representation=contract.representation,
        scoring=contract.scoring,
    )

