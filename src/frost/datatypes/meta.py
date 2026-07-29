"""Batch metadata carrier passed from the host to worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import torch

from frost.datatypes.graph import FlatGraph, HierGraph
from frost.datatypes.tasks import DirectWritebackRequest

from .plan import PlanSet


@dataclass(kw_only=True)
class Meta:
    """All information a worker needs to execute one training or evaluation step.

    Scattered from the host to each worker via ``scatter_object_list``.  The
    ``graph`` field will be populated by the H2D thread after the
    host scatter.

    Attributes:
        cur_epoch: Zero-based epoch index.
        cur_macro_of_split: Zero-based index of this macro-batch within the
            current data split (train / val / test).
        n_macro_of_split: Total number of macro-batches in this split.
        cur_macro_of_epoch: Zero-based macro-batch index within the full epoch
            (across all splits).
        split: Data split this batch belongs to.
        n_src: Number of source nodes in this batch (before ``2 + n_neg``
            expansion).
        batch_idx: Global batch counter used to synchronise P2P memory writes.
        flat_graph: Flat sampled subgraph received from the host; ``None`` for
            the sentinel end-of-epoch signal.
        buf_len: Number of valid rows in the ``flat_graph`` RingBuf slot.
        graph: Hierarchical ``HierGraph`` populated in-place before encoder forward.
        plan_set: Memory / mailbox / nfeat access plans; set by the Planner.
        labels: Binary link-prediction labels for this batch, or ``None``.
        fanout: Fanout list forwarded from the config, or ``None``.
        n_neg: Number of negative samples expected for this batch.
        neg_valid_mask: Optional boolean mask over flattened negative scores.
            Used when an evaluation sampler pads variable-width candidate sets.
    """

    cur_epoch: int
    cur_macro_of_split: int
    n_macro_of_split: int
    cur_macro_of_epoch: int
    split: Literal["train", "val", "test"]
    n_src: int
    mini_batch_idx: int
    flat_graph: FlatGraph | None
    buf_len: int
    n_neg: int
    graph: HierGraph | None = None
    plan_set: PlanSet | None = None
    labels: torch.Tensor | None = None
    neg_valid_mask: torch.Tensor | None = None
    fanout: list[int] | None = None
    flat_nid_cpu: torch.Tensor | None = None  # CPU layout for neighborhood mail
    direct_writeback_request: DirectWritebackRequest | None = None

    @cached_property
    def macro_position_of_split(self) -> str | None:
        """Return the positional label of this batch within its data split.

        Returns:
            ``"first"`` for the first batch, ``"last"`` for the final batch,
            or ``None`` for any batch in between.
        """
        if self.cur_macro_of_split + 1 == self.n_macro_of_split:
            return "last"
        elif self.cur_macro_of_split == 0:
            return "first"
        else:
            return None

    @cached_property
    def is_last_macro_of_epoch(self) -> bool:
        """Return ``True`` iff this is the final batch of the test split."""
        return self.macro_position_of_split == "last" and self.split == "test"
