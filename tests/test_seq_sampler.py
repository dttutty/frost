import frost.seq_sampler as seq_sampler_mod
import numpy as np
import pytest

from frost.seq_sampler import SeqSampler, TemporalGraph


def _make_int64_graph():
    return TemporalGraph(
        indptr=np.array([0, 4], dtype=np.int64),
        indices=np.array([10, 11, 12, 13], dtype=np.int64),
        eid=np.array([100, 101, 102, 103], dtype=np.int64),
        ts=np.array([1, 2, 3, 4], dtype=np.int64),
    )


def test_recent_strategy_left_pads_when_history_is_short():
    sampler = SeqSampler(_make_int64_graph(), strategy="recent", seed=0)

    out_nid, out_eid, out_ts = sampler.get_historical_neighbors([0], [3], 4)

    assert out_nid.tolist() == [[-1, -1, 10, 11]]
    assert out_eid.tolist() == [[-1, -1, 100, 101]]
    assert out_ts.tolist() == [[0, 0, 1, 2]]


def test_recent_strategy_can_mimic_tgl_off_by_one_for_debug(monkeypatch):
    monkeypatch.setenv("FROST_DEBUG_TGL_RECENT_OFF_BY_ONE", "1")
    sampler = SeqSampler(_make_int64_graph(), strategy="recent", seed=0)

    out_nid, out_eid, out_ts = sampler.get_historical_neighbors([0], [3], 4)

    assert out_nid.tolist() == [[-1, -1, -1, 11]]
    assert out_eid.tolist() == [[-1, -1, -1, 101]]
    assert out_ts.tolist() == [[0, 0, 0, 2]]


def test_recent_strategy_can_mimic_tgl_batch_shared_window(monkeypatch):
    monkeypatch.setenv("FROST_DEBUG_TGL_BATCH_SHARED_RECENT", "1")
    sampler = SeqSampler(_make_int64_graph(), strategy="recent", seed=0)

    out_nid, out_eid, out_ts = sampler.get_historical_neighbors([0, 0], [3, 5], 4)

    assert out_nid.tolist() == [[-1, -1, -1, 11], [-1, 11, 12, 13]]
    assert out_eid.tolist() == [[-1, -1, -1, 101], [-1, 101, 102, 103]]
    assert out_ts.tolist() == [[0, 0, 0, 2], [0, 2, 3, 4]]


@pytest.mark.parametrize("strategy", ["recent", "uniform", "time_interval_aware"])
def test_direct_output_matches_returning_api(strategy):
    sampler = SeqSampler(
        _make_int64_graph(),
        strategy=strategy,
        seed=7,
        time_scaling_factor=1.0e-6,
    )
    nid = np.array([0, 0, -1], dtype=np.int64)
    ts = np.array([3, 5, 5], dtype=np.int64)
    expected = sampler.get_historical_neighbors(nid, ts, 3)
    actual = (
        np.empty((3, 3), dtype=np.int64),
        np.empty((3, 3), dtype=np.int64),
        np.empty((3, 3), dtype=np.int64),
    )

    sampler.get_historical_neighbors_into(nid, ts, *actual)

    for expected_array, actual_array in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_uniform_strategy_can_mimic_tgl_low_degree_recent_branch(monkeypatch):
    monkeypatch.setenv("FROST_DEBUG_TGL_UNIFORM_LOW_DEGREE_RECENT", "1")
    sampler = SeqSampler(_make_int64_graph(), strategy="uniform", seed=0)

    out_nid, out_eid, out_ts = sampler.get_historical_neighbors([0], [4], 4)

    # TGL enters its recent branch when eligible <= fanout and its loop omits
    # the oldest eligible edge.
    assert out_nid.tolist() == [[-1, -1, 11, 12]]
    assert out_eid.tolist() == [[-1, -1, 101, 102]]
    assert out_ts.tolist() == [[0, 0, 2, 3]]


def test_uniform_tgl_low_degree_branch_returns_empty_for_one_edge(monkeypatch):
    monkeypatch.setenv("FROST_DEBUG_TGL_UNIFORM_LOW_DEGREE_RECENT", "1")
    sampler = SeqSampler(_make_int64_graph(), strategy="uniform", seed=0)

    out_nid, out_eid, out_ts = sampler.get_historical_neighbors([0], [2], 4)

    assert out_nid.tolist() == [[-1, -1, -1, -1]]
    assert out_eid.tolist() == [[-1, -1, -1, -1]]
    assert out_ts.tolist() == [[0, 0, 0, 0]]


def test_time_interval_aware_outputs_sorted_sampled_times_without_padding():
    sampler = SeqSampler(
        _make_int64_graph(),
        strategy="time_interval_aware",
        seed=7,
        time_scaling_factor=1.0e-6,
    )

    batch_size = 32
    out_nid, out_eid, out_ts = sampler.get_historical_neighbors(
        [0] * batch_size,
        [5] * batch_size,
        3,
    )

    assert out_nid.shape == (batch_size, 3)
    assert np.all(out_nid != -1)
    assert np.all(np.diff(out_ts, axis=1) >= 0)
    assert set(np.unique(out_nid)).issubset({10, 11, 12, 13})
    assert set(np.unique(out_eid)).issubset({100, 101, 102, 103})


def test_time_interval_aware_sampling_changes_empirical_distribution():
    batch_size = 4000
    query_nid = [0] * batch_size
    query_ts = [5] * batch_size

    uniform_sampler = SeqSampler(_make_int64_graph(), strategy="uniform", seed=11)
    tia_sampler = SeqSampler(
        _make_int64_graph(),
        strategy="time_interval_aware",
        seed=11,
        time_scaling_factor=1.0e-6,
    )

    uniform_nid, _, _ = uniform_sampler.get_historical_neighbors(query_nid, query_ts, 1)
    tia_nid, _, _ = tia_sampler.get_historical_neighbors(query_nid, query_ts, 1)

    uniform_counts = np.bincount(uniform_nid.reshape(-1) - 10, minlength=4)
    tia_counts = np.bincount(tia_nid.reshape(-1) - 10, minlength=4)

    # DyGLib's CAWN weighting is not uniform even when sampling with replacement.
    assert tia_counts.tolist() != uniform_counts.tolist()
    assert tia_counts[0] > tia_counts[1] > tia_counts[2] > tia_counts[3]


def test_legacy_native_extension_falls_back_when_scaling_is_zero():
    calls: list[tuple[object, ...]] = []

    def legacy_factory(*args):
        calls.append(args)
        if len(args) == 4:
            raise TypeError("__init__(): incompatible function arguments")
        return args

    graph = object()
    result = seq_sampler_mod._instantiate_seq_sampler(
        legacy_factory,
        graph,
        strategy="recent",
        seed=0,
        time_scaling_factor=0.0,
    )

    assert calls == [
        (graph, "recent", 0, 0.0),
        (graph, "recent", 0),
    ]
    assert result == (graph, "recent", 0)


def test_legacy_native_extension_raises_actionable_error_for_nonzero_scaling():
    def legacy_factory(*args):
        raise TypeError("__init__(): incompatible function arguments")

    with pytest.raises(RuntimeError, match="uv pip install -e \\."):
        seq_sampler_mod._instantiate_seq_sampler(
            legacy_factory,
            object(),
            strategy="time_interval_aware",
            seed=7,
            time_scaling_factor=1.0e-6,
        )
