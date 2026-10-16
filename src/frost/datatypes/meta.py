"""Batch metadata carrier passed from the host to worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.tasks import DirectWritebackRequest

from .plan import PlanSet


@dataclass(kw_only=True)
class Meta:
    """All information a worker needs to execute one training or evaluation step.

    Tensor-encoded by the host and sent to one worker through the control plane. The
    Attributes:
        split: Data split this batch belongs to.
        n_src: Number of source nodes in this batch (before ``2 + n_neg``
            expansion).
        mini_batch_idx: Global per-worker batch counter, unique across workers.
        macro_batch_idx: Global macro-batch counter shared by every worker in a
            bundle, or ``-1`` for metadata without distributed state tracking.
        epoch_idx: Zero-based epoch index for dynamic-state reset boundaries.
        flat_graph: Flat sampled subgraph received from the host; ``None`` for
            the sentinel end-of-epoch signal.
        buf_len: Number of valid rows in the ``flat_graph`` RingBuf slot.
        plan_set: Memory / mailbox / nfeat access plans; set by the Planner.
        labels: Binary link-prediction labels for this batch, or ``None``.
        n_neg: Number of negative samples expected for this batch.
        is_last_macro_of_split: Whether this batch closes its data split.
        is_last_macro_of_epoch: Whether this batch closes the epoch, including
            the final train batch in train-only runs.
        neg_valid_mask: Optional boolean mask over flattened negative scores.
            Used when an evaluation sampler pads variable-width candidate sets.
    """

    split: Literal["train", "val", "test"]
    n_src: int
    mini_batch_idx: int
    flat_graph: FlatGraph | None
    buf_len: int
    n_neg: int
    macro_batch_idx: int = -1
    epoch_idx: int = 0
    is_last_macro_of_split: bool = False
    is_last_macro_of_epoch: bool = False
    plan_set: PlanSet | None = None
    labels: torch.Tensor | None = None
    neg_valid_mask: torch.Tensor | None = None
    flat_nid_cpu: torch.Tensor | None = None  # CPU layout for neighborhood mail
    direct_writeback_request: DirectWritebackRequest | None = None
