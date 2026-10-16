"""Host macro IDs must represent the same state version on every worker."""

from queue import Queue

import pandas as pd
import pytest

from frost.datatypes.meta import Meta
from frost.runtime.host._engine import Host


def _host(*, train_only: bool = False) -> Host:
    host = Host.__new__(Host)
    host.batch_size = 2
    host.eval_batch_size = 2
    host.n_gpu = 4
    host.show_tqdm = False
    host.mini_batch_idx = 0
    host.macro_batch_idx = 0
    host.curr_epoch = 0
    host.train_only = train_only
    host.plan_router = None
    host.scatter_q = Queue()
    host.share_neg_sampler_across_splits = True

    def add_to_td_list(edges, split, mini_batch_idx, gpu_id, **metadata):
        return Meta(
            split=split,
            n_src=len(edges),
            mini_batch_idx=mini_batch_idx,
            flat_graph=None,
            buf_len=0,
            n_neg=1,
            **metadata,
        ), None

    host._add_to_td_list = add_to_td_list
    return host


def _edges(count: int) -> pd.DataFrame:
    return pd.DataFrame({"src": range(count)})


def _drain(host: Host) -> list[list[Meta]]:
    bundles = []
    while not host.scatter_q.empty():
        item = host.scatter_q.get_nowait()
        if isinstance(item, list):
            bundles.append(item)
    return bundles


def test_macro_ids_are_shared_and_monotonic_across_splits_and_epochs() -> None:
    host = _host()

    for epoch in range(2):
        for split in ("train", "val", "test"):
            host._handle_a_split(_edges(13), split, epoch)

    bundles = _drain(host)
    assert len(bundles) == 12
    for macro_idx, bundle in enumerate(bundles):
        assert len(bundle) == host.n_gpu
        assert {meta.macro_batch_idx for meta in bundle} == {macro_idx}
        assert {meta.epoch_idx for meta in bundle} == {macro_idx // 6}
        assert {meta.is_last_macro_of_split for meta in bundle} == {
            macro_idx % 2 == 1
        }
        assert {meta.is_last_macro_of_epoch for meta in bundle} == {
            macro_idx % 6 == 5
        }
    assert [meta.mini_batch_idx for bundle in bundles for meta in bundle] == list(
        range(48)
    )


@pytest.mark.parametrize("train_only", [False, True])
def test_only_train_only_runs_mark_last_train_macro_as_epoch_end(
    train_only: bool,
) -> None:
    host = _host(train_only=train_only)

    host._handle_a_split(_edges(13), "train", 0)

    first, last = _drain(host)
    assert all(not meta.is_last_macro_of_epoch for meta in first)
    assert all(meta.is_last_macro_of_epoch is train_only for meta in last)
    assert all(meta.is_last_macro_of_split for meta in last)


def test_debug_batch_cap_marks_actual_last_macro(monkeypatch) -> None:
    monkeypatch.setenv("FROST_DEBUG_MAX_BATCHES_PER_SPLIT", "1")
    host = _host(train_only=True)

    host._handle_a_split(_edges(24), "train", 0)

    (bundle,) = _drain(host)
    assert all(meta.is_last_macro_of_split for meta in bundle)
    assert all(meta.is_last_macro_of_epoch for meta in bundle)
