"""Declarative model semantics consumed by the runtime plan compiler.

The types in this module deliberately contain no runtime objects.  A model
recipe describes what data and state transitions its computation requires;
the launch layer decides how those requirements are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class RootExpansion(StrEnum):
    """Which roots receive sampled neighborhoods."""

    ALL_ROOTS = "all_roots"
    POSITIVE_ROOTS = "positive_roots"


class NodeSet(StrEnum):
    """Semantic node selections understood by access and state planners."""

    NONE = "none"
    ROOTS = "roots"
    POSITIVE_ROOTS = "positive_roots"
    ALL_SAMPLED = "all_sampled"
    POSITIVE_NEIGHBORS = "positive_neighbors"


class RepresentationKind(StrEnum):
    NODE_ALIGNED = "node_aligned"
    PAIR_ALIGNED = "pair_aligned"


@dataclass(frozen=True)
class NeighborhoodContract:
    min_hops: int
    max_hops: int
    expansion: RootExpansion = RootExpansion.ALL_ROOTS

    def __post_init__(self) -> None:
        if self.min_hops < 0 or self.max_hops < self.min_hops:
            raise ValueError(
                "Neighborhood hop bounds must satisfy 0 <= min_hops <= max_hops."
            )

    def validate(self, fanout: tuple[int, ...]) -> None:
        if not self.min_hops <= len(fanout) <= self.max_hops:
            raise ValueError(
                f"Model requires {self.min_hops}..{self.max_hops} hops, "
                f"but fanout={list(fanout)} defines {len(fanout)}."
            )
        if any(value <= 0 for value in fanout):
            raise ValueError(f"Fanout values must be positive, got {list(fanout)}.")
        if self.expansion is RootExpansion.POSITIVE_ROOTS and len(fanout) > 1:
            raise ValueError("Positive-root-only expansion currently supports one hop.")


@dataclass(frozen=True)
class InteractionMailbox:
    pass


@dataclass(frozen=True)
class NeighborhoodMailbox:
    recipients: NodeSet = NodeSet.POSITIVE_NEIGHBORS


MailboxContract: TypeAlias = InteractionMailbox | NeighborhoodMailbox


@dataclass(frozen=True)
class Stateless:
    pass


@dataclass(frozen=True)
class MemoryState:
    memory_read: NodeSet
    memory_write: NodeSet
    mailbox_read: NodeSet
    mailbox: MailboxContract


DynamicStateContract: TypeAlias = Stateless | MemoryState


@dataclass(frozen=True)
class ModelContract:
    neighborhood: NeighborhoodContract
    node_feature_scope: NodeSet
    dynamic_state: DynamicStateContract
    representation: RepresentationKind
