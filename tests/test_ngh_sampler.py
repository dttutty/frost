"""Tests for zero-hop and unified neighborhood sampling."""

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from frost.models.contracts import RootExpansion
from frost.sampling.ngh_sampler import (
    NeighborhoodSampler,
    NoSampler,
    _copy_native_array_into,
    dist_safe_tensor,
)


@dataclass
class MockSlot:
    NID: torch.Tensor
    TS: torch.Tensor
    EID: torch.Tensor


class MockRingBuf:
    def __init__(self, total_size: int = 1000) -> None:
        self.slot = MockSlot(
            NID=torch.zeros(total_size, dtype=torch.int64),
            TS=torch.zeros(total_size, dtype=torch.int64),
            EID=torch.zeros(total_size, dtype=torch.int64),
        )

    def acquire_slot(self, gpu_id: int) -> dict[str, torch.Tensor]:
        del gpu_id
        return {"NID": self.slot.NID, "TS": self.slot.TS, "EID": self.slot.EID}

    def commit_slot(self, gpu_id: int) -> None:
        del gpu_id


@pytest.fixture(autouse=True)
def _restore_tgl_sampler_parity_env():
    names = (
        "FROST_DEBUG_TGL_BATCH_SHARED_RECENT",
        "FROST_DEBUG_TGL_UNIFORM_LOW_DEGREE_RECENT",
    )
    old = {name: os.environ.pop(name, None) for name in names}
    yield
    for name in names:
        os.environ.pop(name, None)
        if old[name] is not None:
            os.environ[name] = old[name]


def _graph() -> MagicMock:
    graph = MagicMock()
    graph.indices = np.array([1, 2, 3], dtype=np.int64)
    graph.ts = np.array([100, 200, 300], dtype=np.int64)
    return graph


def test_no_sampler_writes_roots_using_batch_negative_width() -> None:
    ring_buf = MockRingBuf(total_size=100)
    sampler = NoSampler(
        ring_buf=ring_buf,  # type: ignore[arg-type]
        root_expansion=RootExpansion.ALL_ROOTS,
    )
    root_nid = torch.arange(8)
    root_ts = torch.tensor([10, 20])

    sampled, length = sampler.sample(
        root_nid=root_nid,
        root_ts=root_ts,
        sample_nid=torch.empty(0),
        sample_ts=torch.empty(0),
        pos_eid=torch.tensor([100, 200]),
        gpu_id=0,
    )

    assert sampler.fanout == ()
    assert length == 8
    assert sampled.tolist() == root_nid.tolist()
    assert ring_buf.slot.TS[:length].tolist() == [10, 20, 10, 20, 10, 10, 20, 20]
    assert ring_buf.slot.EID[:length].tolist() == [100, 200, -1, -1, -1, -1, -1, -1]


@pytest.mark.parametrize(
    ("value", "expected_shape"),
    [(None, (0,)), (np.array([], dtype=np.int64), (0,)), (np.array([[1, 2]]), (1, 2))],
)
def test_dist_safe_tensor_detaches_native_arrays(value, expected_shape) -> None:
    result = dist_safe_tensor(value, torch.int64)

    assert result.shape == expected_shape
    assert result.dtype == torch.int64


def test_copy_native_array_into_detaches_source() -> None:
    source = np.array([[1, 2], [3, 4]], dtype=np.int64)
    destination = torch.zeros((2, 2), dtype=torch.int64)

    _copy_native_array_into(destination, source)
    source.fill(99)

    assert destination.tolist() == [[1, 2], [3, 4]]


def test_copy_native_array_into_rejects_shape_mismatch() -> None:
    destination = torch.zeros((2, 2), dtype=torch.int64)

    with pytest.raises(ValueError, match="does not match destination shape"):
        _copy_native_array_into(destination, np.zeros((4,), dtype=np.int64))


def test_neighborhood_sampler_accepts_one_or_more_hops() -> None:
    ring_buf = MockRingBuf()
    with patch("frost.sampling.ngh_sampler.SeqSampler") as sampler_type:
        one_hop = NeighborhoodSampler(
            graph=_graph(),
            fanout=[10],
            ring_buf=ring_buf,  # type: ignore[arg-type]
            root_expansion=RootExpansion.POSITIVE_ROOTS,
            strategy="recent",
            seed=42,
        )
        two_hop = NeighborhoodSampler(
            graph=_graph(),
            fanout=[10, 5],
            ring_buf=ring_buf,  # type: ignore[arg-type]
            root_expansion=RootExpansion.ALL_ROOTS,
            strategy="uniform",
            seed=0,
        )

    assert one_hop.fanout == (10,)
    assert two_hop.fanout == (10, 5)
    assert sampler_type.call_count == 2


def test_multi_hop_rejects_positive_root_only_expansion() -> None:
    with pytest.raises(ValueError, match="supports one hop"):
        NeighborhoodSampler(
            graph=_graph(),
            fanout=[10, 5],
            ring_buf=MockRingBuf(),  # type: ignore[arg-type]
            root_expansion=RootExpansion.POSITIVE_ROOTS,
            strategy="uniform",
            seed=0,
        )


def test_tgl_parity_enables_native_sampler_flags() -> None:
    with patch("frost.sampling.ngh_sampler.SeqSampler"):
        NeighborhoodSampler(
            graph=_graph(),
            fanout=[10],
            ring_buf=MockRingBuf(),  # type: ignore[arg-type]
            root_expansion=RootExpansion.ALL_ROOTS,
            strategy="recent",
            seed=0,
            tgl_sampler_parity=True,
        )

    assert os.environ["FROST_DEBUG_TGL_BATCH_SHARED_RECENT"] == "1"
    assert os.environ["FROST_DEBUG_TGL_UNIFORM_LOW_DEGREE_RECENT"] == "1"


def test_multi_hop_clears_reused_rows_below_padding_parents() -> None:
    ring_buf = MockRingBuf(total_size=100)
    ring_buf.slot.NID.fill_(777)
    ring_buf.slot.TS.fill_(777)
    ring_buf.slot.EID.fill_(777)
    first_hop = (
        np.array([[10, -1], [20, 21], [-1, -1]], dtype=np.int64),
        np.array([[1, -1], [2, 3], [-1, -1]], dtype=np.int64),
        np.array([[9, 0], [8, 7], [0, 0]], dtype=np.int64),
    )
    second_hop = (
        np.array([[30, 31], [40, -1], [50, 51]], dtype=np.int64),
        np.array([[4, 5], [6, -1], [7, 8]], dtype=np.int64),
        np.array([[6, 5], [4, 0], [3, 2]], dtype=np.int64),
    )

    with patch("frost.sampling.ngh_sampler.SeqSampler") as sampler_type:
        sampler_type.return_value.get_historical_neighbors_into = None
        sampler_type.return_value.get_historical_neighbors.side_effect = [
            first_hop,
            second_hop,
        ]
        sampler = NeighborhoodSampler(
            graph=_graph(),
            fanout=[2, 2],
            ring_buf=ring_buf,  # type: ignore[arg-type]
            root_expansion=RootExpansion.ALL_ROOTS,
            strategy="uniform",
            seed=0,
        )
        _, length = sampler.sample(
            root_nid=torch.tensor([1, 2, 3]),
            root_ts=torch.tensor([10]),
            sample_nid=torch.tensor([1, 2, 3]),
            sample_ts=torch.tensor([10, 10, 10]),
            pos_eid=torch.tensor([9]),
            gpu_id=0,
        )

    assert length == 21
    second_call_nodes = sampler_type.return_value.get_historical_neighbors.call_args_list[1].args[0]
    assert second_call_nodes.tolist() == [10, 20, 21]
    second_tier = ring_buf.slot.NID[9:21].view(6, 2)
    assert second_tier[1].tolist() == [-1, -1]
    assert second_tier[4:].tolist() == [[-1, -1], [-1, -1]]
