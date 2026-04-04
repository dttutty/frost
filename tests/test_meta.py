"""Tests for Meta batch metadata datatype."""

import torch

from frost.datatypes.graph import FlatGraph, HierGraph
from frost.datatypes.meta import Meta
from frost.datatypes.plan import PlanSet


class TestMetaBasic:
    def test_basic_creation(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=5,
            n_macro_of_split=100,
            cur_macro_of_epoch=5,
            split="train",
            n_src=32,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=96,
            n_neg=1,
        )
        assert meta.cur_epoch == 0
        assert meta.cur_macro_of_split == 5
        assert meta.n_macro_of_split == 100
        assert meta.split == "train"
        assert meta.n_src == 32
        assert meta.n_neg == 1

    def test_default_fields(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=0,
            n_macro_of_split=10,
            cur_macro_of_epoch=0,
            split="train",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.graph is None
        assert meta.plan_set is None
        assert meta.labels is None
        assert meta.fanout is None

    def test_with_flat_graph(self):
        fg = FlatGraph(
            nid=torch.tensor([1, 2, 3]),
            eid=torch.tensor([10, 11, 12]),
            ts=torch.tensor([100, 101, 102]),
        )
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=0,
            n_macro_of_split=10,
            cur_macro_of_epoch=0,
            split="train",
            n_src=1,
            mini_batch_idx=0,
            flat_graph=fg,
            buf_len=3,
            n_neg=1,
        )
        assert meta.flat_graph is not None
        assert meta.flat_graph.nid.tolist() == [1, 2, 3]

    def test_with_labels(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=0,
            n_macro_of_split=10,
            cur_macro_of_epoch=0,
            split="train",
            n_src=4,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=12,
            n_neg=1,
            labels=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        assert meta.labels is not None
        assert meta.labels.tolist() == [1.0, 0.0, 1.0, 0.0]


class TestMetaMacroPosition:
    def test_first_batch(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=0,
            n_macro_of_split=10,
            cur_macro_of_epoch=0,
            split="train",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.macro_position_of_split == "first"

    def test_last_batch(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=9,
            n_macro_of_split=10,
            cur_macro_of_epoch=9,
            split="train",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.macro_position_of_split == "last"

    def test_middle_batch(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=5,
            n_macro_of_split=10,
            cur_macro_of_epoch=5,
            split="train",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.macro_position_of_split is None

    def test_single_batch_is_first_and_last(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=0,
            n_macro_of_split=1,
            cur_macro_of_epoch=0,
            split="test",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.macro_position_of_split == "last"


class TestMetaIsLastMacroOfEpoch:
    def test_last_batch_of_test_is_last_macro(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=9,
            n_macro_of_split=10,
            cur_macro_of_epoch=29,
            split="test",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.is_last_macro_of_epoch is True

    def test_last_batch_of_train_is_not_last_macro(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=9,
            n_macro_of_split=10,
            cur_macro_of_epoch=9,
            split="train",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.is_last_macro_of_epoch is False

    def test_first_batch_of_test_is_not_last_macro(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=0,
            n_macro_of_split=10,
            cur_macro_of_epoch=20,
            split="test",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.is_last_macro_of_epoch is False

    def test_val_split_not_last_macro(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=4,
            n_macro_of_split=5,
            cur_macro_of_epoch=14,
            split="val",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        assert meta.is_last_macro_of_epoch is False


class TestMetaCachedProperty:
    def test_cached_property_reuse(self):
        meta = Meta(
            cur_epoch=0,
            cur_macro_of_split=0,
            n_macro_of_split=10,
            cur_macro_of_epoch=0,
            split="train",
            n_src=8,
            mini_batch_idx=0,
            flat_graph=None,
            buf_len=24,
            n_neg=1,
        )
        pos1 = meta.macro_position_of_split
        pos2 = meta.macro_position_of_split
        assert pos1 is pos2
