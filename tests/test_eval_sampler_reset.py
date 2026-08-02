from queue import Queue

import pandas as pd
import pytest
import torch

from frost.neg_sampler.random import RandomNegLinkSampler
from frost.runtime.host._engine import Host


class _FakeEvalSampler:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_random_neg_sampler_reset_replays_the_same_sequence():
    sampler = RandomNegLinkSampler(
        n_nodes=128,
        n_neg=2,
        dataset="DUMMY",
        candidate_mode="all",
        seed=7,
    )
    src = torch.arange(32, dtype=torch.long)
    eid = torch.arange(1, 33, dtype=torch.long)

    first = sampler.sample(src, eid)
    second = sampler.sample(src, eid)
    sampler.reset()
    replay = sampler.sample(src, eid)

    assert not torch.equal(first, second)
    assert torch.equal(first, replay)


def test_handle_a_split_resets_eval_sampler_but_not_train_sampler():
    host = Host.__new__(Host)
    host.batch_size = 1
    host.n_gpu = 1
    host.show_tqdm = False
    host.mini_batch_idx = 0
    host.curr_epoch = 0
    host.plan_router = None
    host.scatter_q = Queue()
    host.neg_sampler_train = object()
    host.neg_sampler_eval = _FakeEvalSampler()
    host._add_to_td_list = lambda *args, **kwargs: ("meta", "layout")

    val_df = pd.DataFrame(
        {"src": [0], "dst": [1], "time": [1.0], "eid": [1], "default_split": [1]}
    )
    host._handle_a_split(val_df, "val", 0)
    assert host.neg_sampler_eval.reset_calls == 1
    assert host.scatter_q.get_nowait() == ["meta"]

    train_df = pd.DataFrame(
        {"src": [0], "dst": [1], "time": [1.0], "eid": [1], "default_split": [0]}
    )
    host._handle_a_split(train_df, "train", 0)
    assert host.neg_sampler_eval.reset_calls == 1


def test_split_specific_debug_batch_cap_overrides_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = Host.__new__(Host)
    host.batch_size = 1
    host.eval_batch_size = 1
    host.n_gpu = 1
    host.show_tqdm = False
    host.mini_batch_idx = 0
    host.curr_epoch = 0
    host.plan_router = None
    host.scatter_q = Queue()
    host.neg_sampler_train = object()
    host.neg_sampler_eval = _FakeEvalSampler()
    calls: list[int] = []

    def add_to_td_list(edges, *args, **kwargs):
        calls.append(len(edges))
        return "meta", "layout"

    host._add_to_td_list = add_to_td_list
    monkeypatch.setenv("FROST_DEBUG_MAX_BATCHES_PER_SPLIT", "2")
    monkeypatch.setenv("FROST_DEBUG_MAX_BATCHES_TEST", "1")
    test_df = pd.DataFrame(
        {
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "time": [1.0, 2.0, 3.0],
            "eid": [1, 2, 3],
            "default_split": [2, 2, 2],
        }
    )

    host._handle_a_split(test_df, "test", 0)

    assert calls == [1]
