"""Model recipes: component configuration plus a declarative contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from frost.models.contracts import (
    FeatureContract,
    FeatureUse,
    InteractionMailbox,
    MemoryState,
    ModelContract,
    NeighborhoodContract,
    NeighborhoodMailbox,
    NodeSet,
    RepresentationKind,
    RootExpansion,
    ScoringKind,
    Stateless,
    StructuralMailbox,
)


class ModelRecipe:
    """Cheap, CPU-only description shared by host and worker ranks."""

    NAME: ClassVar[str]
    CONTRACT: ClassVar[ModelContract]

    def __init__(
        self,
        *,
        embedding_dim: int,
        neighborhood: Mapping[str, Any],
        encoder: Mapping[str, Any],
        state: Mapping[str, Any] | None,
        head: Mapping[str, Any],
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("model.embedding_dim must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.neighborhood = dict(neighborhood)
        self.encoder = dict(encoder)
        self.state = dict(state) if state is not None else None
        self.head = dict(head)

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def contract(self) -> ModelContract:
        return self.CONTRACT


_OPTIONAL_FEATURES = FeatureContract(
    node=FeatureUse.OPTIONAL,
    edge=FeatureUse.OPTIONAL,
)
_SYNTHETIC_FEATURES = FeatureContract(
    node=FeatureUse.SYNTHESIZE_IF_MISSING,
    edge=FeatureUse.SYNTHESIZE_IF_MISSING,
)
_STANDARD_MEMORY = MemoryState(
    memory_read=NodeSet.ALL_SAMPLED,
    memory_write=NodeSet.POSITIVE_ROOTS,
    mailbox_read=NodeSet.ALL_SAMPLED,
    mailbox=InteractionMailbox(),
)


class TGNRecipe(ModelRecipe):
    NAME = "tgn"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(1, 1),
        features=_OPTIONAL_FEATURES,
        dynamic_state=_STANDARD_MEMORY,
        representation=RepresentationKind.NODE_ALIGNED,
        scoring=ScoringKind.ADDITIVE,
    )


class JODIERecipe(ModelRecipe):
    NAME = "jodie"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(0, 0),
        features=_OPTIONAL_FEATURES,
        dynamic_state=_STANDARD_MEMORY,
        representation=RepresentationKind.NODE_ALIGNED,
        scoring=ScoringKind.ADDITIVE,
    )


class DyRepRecipe(ModelRecipe):
    NAME = "dyrep"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(1, 1),
        features=_OPTIONAL_FEATURES,
        dynamic_state=MemoryState(
            memory_read=NodeSet.ALL_SAMPLED,
            memory_write=NodeSet.POSITIVE_ROOTS,
            mailbox_read=NodeSet.ALL_SAMPLED,
            mailbox=StructuralMailbox(),
        ),
        representation=RepresentationKind.NODE_ALIGNED,
        scoring=ScoringKind.ADDITIVE,
    )


class APANRecipe(ModelRecipe):
    NAME = "apan"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(
            1,
            1,
            expansion=RootExpansion.POSITIVE_ROOTS,
        ),
        features=FeatureContract(
            node=FeatureUse.OPTIONAL,
            edge=FeatureUse.OPTIONAL,
            node_scope=NodeSet.ROOTS,
        ),
        dynamic_state=MemoryState(
            memory_read=NodeSet.ROOTS,
            memory_write=NodeSet.POSITIVE_ROOTS,
            mailbox_read=NodeSet.ROOTS,
            mailbox=NeighborhoodMailbox(),
        ),
        representation=RepresentationKind.NODE_ALIGNED,
        scoring=ScoringKind.ADDITIVE,
    )


class TGATRecipe(ModelRecipe):
    NAME = "tgat"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(1, 8),
        features=_SYNTHETIC_FEATURES,
        dynamic_state=Stateless(),
        representation=RepresentationKind.NODE_ALIGNED,
        scoring=ScoringKind.CONCAT,
    )


class GraphMixerRecipe(ModelRecipe):
    NAME = "graph_mixer"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(1, 1),
        features=_OPTIONAL_FEATURES,
        dynamic_state=Stateless(),
        representation=RepresentationKind.NODE_ALIGNED,
        scoring=ScoringKind.CONCAT,
    )


class TCLRecipe(ModelRecipe):
    NAME = "tcl"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(1, 1),
        features=_OPTIONAL_FEATURES,
        dynamic_state=Stateless(),
        representation=RepresentationKind.PAIR_ALIGNED,
        scoring=ScoringKind.CONCAT,
    )


class DyGFormerRecipe(ModelRecipe):
    NAME = "dygformer"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(1, 1),
        features=_SYNTHETIC_FEATURES,
        dynamic_state=Stateless(),
        representation=RepresentationKind.PAIR_ALIGNED,
        scoring=ScoringKind.ADDITIVE,
    )


class CAWNRecipe(ModelRecipe):
    NAME = "cawn"
    CONTRACT = ModelContract(
        neighborhood=NeighborhoodContract(1, 8),
        features=_OPTIONAL_FEATURES,
        dynamic_state=Stateless(),
        representation=RepresentationKind.PAIR_ALIGNED,
        scoring=ScoringKind.CONCAT,
    )


RECIPE_TYPES = {
    recipe.NAME: recipe
    for recipe in (
        TGNRecipe,
        JODIERecipe,
        DyRepRecipe,
        APANRecipe,
        TGATRecipe,
        GraphMixerRecipe,
        TCLRecipe,
        DyGFormerRecipe,
        CAWNRecipe,
    )
}

