"""Tests for the worker batch metadata carrier."""

import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.meta import Meta


def _meta(**overrides) -> Meta:
    values = {
        "split": "train",
        "n_src": 8,
        "mini_batch_idx": 0,
        "flat_graph": None,
        "buf_len": 24,
        "n_neg": 1,
    }
    values.update(overrides)
    return Meta(**values)


def test_meta_carries_batch_execution_inputs() -> None:
    graph = FlatGraph(
        nid=torch.tensor([1, 2, 3]),
        eid=torch.tensor([10, 11, 12]),
        ts=torch.tensor([100, 101, 102]),
    )
    labels = torch.tensor([1.0, 0.0])

    meta = _meta(flat_graph=graph, labels=labels)

    assert meta.flat_graph is graph
    assert meta.labels is labels
    assert meta.plan_set is None
    assert meta.is_last_macro_of_epoch is False
    assert meta.is_last_macro_of_split is False
    assert meta.macro_batch_idx == -1
    assert meta.epoch_idx == 0


def test_last_macro_of_epoch_is_explicit_host_input() -> None:
    meta = _meta(split="test", is_last_macro_of_epoch=True)

    assert meta.is_last_macro_of_epoch is True
