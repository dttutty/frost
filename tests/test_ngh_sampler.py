"""Tests for neighborhood samplers."""

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from frost.sampling.ngh_sampler import NoSampler, dist_safe_tensor


@dataclass
class MockSlot:
    NID: torch.Tensor
    TS: torch.Tensor
    EID: torch.Tensor


class MockRingBuf:
    def __init__(self, total_size: int = 1000, ndtype: torch.dtype = torch.int64):
        self.slot = MockSlot(
            NID=torch.zeros(total_size, dtype=ndtype),
            TS=torch.zeros(total_size, dtype=torch.int64),
            EID=torch.zeros(total_size, dtype=ndtype),
        )
        self.acquired = False

    def acquire_slot(self, gpu_id: int) -> dict[str, torch.Tensor]:
        self.acquired = True
        return {
            "NID": self.slot.NID,
            "TS": self.slot.TS,
            "EID": self.slot.EID,
        }

    def commit_slot(self, gpu_id: int) -> None:
        self.acquired = False


@pytest.fixture(autouse=True)
def _restore_tgl_sampler_parity_env():
    old = os.environ.pop("FROST_DEBUG_TGL_BATCH_SHARED_RECENT", None)
    try:
        yield
    finally:
        os.environ.pop("FROST_DEBUG_TGL_BATCH_SHARED_RECENT", None)
        if old is not None:
            os.environ["FROST_DEBUG_TGL_BATCH_SHARED_RECENT"] = old


class TestNoSampler:
    def test_fanout_always_none(self):
        ring_buf = MockRingBuf()
        sampler = NoSampler(
            graph=MagicMock(),
            ring_buf=ring_buf,  # type: ignore[arg-type]
            n_neg=1,
            fanout=[10, 5],
            sample_4_neg=True,
        )
        assert sampler.fanout is None

    def test_sample_writes_seeds_only(self):
        ring_buf = MockRingBuf(total_size=100)
        sampler = NoSampler(
            graph=MagicMock(),
            ring_buf=ring_buf,  # type: ignore[arg-type]
            n_neg=1,
            fanout=None,
            sample_4_neg=False,
        )

        n_src = 3
        n_neg = 1
        root_nid = torch.arange(n_src * (2 + n_neg))
        root_ts = torch.arange(n_src)
        sample_nid = torch.tensor([])
        sample_ts = torch.tensor([])
        pos_eid = torch.arange(n_src)

        valid_nid_raw, length = sampler.sample(
            root_nid=root_nid,
            root_ts=root_ts,
            sample_nid=sample_nid,
            sample_ts=sample_ts,
            pos_eid=pos_eid,
            gpu_id=0,
        )

        assert length == n_src * (2 + n_neg)
        assert valid_nid_raw.tolist() == root_nid.tolist()

    def test_sample_uses_batch_n_neg_instead_of_configured_maximum(self):
        ring_buf = MockRingBuf(total_size=100)
        sampler = NoSampler(
            graph=MagicMock(),
            ring_buf=ring_buf,  # type: ignore[arg-type]
            n_neg=20,
            fanout=None,
            sample_4_neg=False,
        )
        root_nid = torch.arange(6)  # two positives, one negative each
        root_ts = torch.tensor([10, 20])

        valid_nid_raw, length = sampler.sample(
            root_nid=root_nid,
            root_ts=root_ts,
            sample_nid=torch.tensor([]),
            sample_ts=torch.tensor([]),
            pos_eid=torch.tensor([1, 2]),
            gpu_id=0,
        )

        assert length == 6
        assert ring_buf.slot.TS[:length].tolist() == [10, 20, 10, 20, 10, 20]
        assert valid_nid_raw.tolist() == root_nid.tolist()

    def test_sample_expands_timestamps_for_negatives(self):
        ring_buf = MockRingBuf(total_size=100)
        sampler = NoSampler(
            graph=MagicMock(),
            ring_buf=ring_buf,  # type: ignore[arg-type]
            n_neg=2,
            fanout=None,
            sample_4_neg=False,
        )

        n_src = 2
        n_neg = 2
        root_nid = torch.arange(n_src * (2 + n_neg))
        root_ts = torch.tensor([10, 20], dtype=torch.int64)
        sample_nid = torch.tensor([])
        sample_ts = torch.tensor([])
        pos_eid = torch.arange(n_src)

        valid_nid_raw, length = sampler.sample(
            root_nid=root_nid,
            root_ts=root_ts,
            sample_nid=sample_nid,
            sample_ts=sample_ts,
            pos_eid=pos_eid,
            gpu_id=0,
        )

        assert length == n_src * (2 + n_neg)
        assert ring_buf.slot.TS[:length].tolist() == [10, 20, 10, 20, 10, 10, 20, 20]
        assert valid_nid_raw.tolist() == root_nid.tolist()

    def test_sample_sets_eid_for_seeds(self):
        ring_buf = MockRingBuf(total_size=100)
        sampler = NoSampler(
            graph=MagicMock(),
            ring_buf=ring_buf,  # type: ignore[arg-type]
            n_neg=1,
            fanout=None,
            sample_4_neg=False,
        )

        n_src = 2
        root_nid = torch.arange(n_src * 3)
        root_ts = torch.arange(n_src)
        sample_nid = torch.tensor([])
        sample_ts = torch.tensor([])
        pos_eid = torch.tensor([100, 200])

        valid_nid_raw, length = sampler.sample(
            root_nid=root_nid,
            root_ts=root_ts,
            sample_nid=sample_nid,
            sample_ts=sample_ts,
            pos_eid=pos_eid,
            gpu_id=0,
        )

        assert ring_buf.slot.EID[:n_src].tolist() == [100, 200]
        assert (ring_buf.slot.EID[n_src : n_src * 3] == -1).all()


class TestDistSafeTensor:
    def test_returns_empty_for_none(self):
        result = dist_safe_tensor(None, torch.int64)
        assert result.shape == (0,)
        assert result.dtype == torch.int64

    def test_returns_empty_for_empty_array(self):
        arr = np.array([], dtype=np.int64)
        result = dist_safe_tensor(arr, torch.int64)
        assert result.shape == (0,)

    def test_copies_non_empty_array(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = dist_safe_tensor(arr, torch.int64)
        assert result.tolist() == [1, 2, 3]
        assert result.dtype == torch.int64

    def test_preserves_shape(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        result = dist_safe_tensor(arr, torch.int64)
        assert result.shape == (2, 2)
        assert result.tolist() == [[1, 2], [3, 4]]


class TestNghSamplerInit:
    def test_fanout_must_be_single_hop(self):
        ring_buf = MockRingBuf()
        with pytest.raises(AssertionError, match="Only single-hop"):
            from frost.sampling.ngh_sampler import NghSampler

            NghSampler(
                graph=MagicMock(),
                fanout=[10, 5],
                ring_buf=ring_buf,  # type: ignore[arg-type]
                sample_4_neg=True,
                n_neg=1,
                strategy="uniform",
                seed=0,
            )

    def test_initializes_seq_sampler(self):
        ring_buf = MockRingBuf()
        mock_graph = MagicMock()
        mock_graph.indices = np.array([1, 2, 3], dtype=np.int64)
        mock_graph.ts = np.array([100, 200, 300], dtype=np.int64)

        with patch("frost.sampling.ngh_sampler.SeqSampler") as MockSeqSampler:
            mock_sampler = MagicMock()
            MockSeqSampler.return_value = mock_sampler

            from frost.sampling.ngh_sampler import NghSampler

            sampler = NghSampler(
                graph=mock_graph,
                fanout=[10],
                ring_buf=ring_buf,  # type: ignore[arg-type]
                sample_4_neg=True,
                n_neg=1,
                strategy="recent",
                seed=42,
            )

            assert sampler.n_ngh == 10
            MockSeqSampler.assert_called_once_with(
                graph=mock_graph,
                strategy="recent",
                seed=42,
                time_scaling_factor=0.0,
            )

    def test_initializes_seq_sampler_with_tgl_sampler_parity(self, monkeypatch):
        ring_buf = MockRingBuf()
        mock_graph = MagicMock()
        mock_graph.indices = np.array([1, 2, 3], dtype=np.int64)
        mock_graph.ts = np.array([100, 200, 300], dtype=np.int64)
        monkeypatch.delenv("FROST_DEBUG_TGL_BATCH_SHARED_RECENT", raising=False)

        with patch("frost.sampling.ngh_sampler.SeqSampler") as MockSeqSampler:
            from frost.sampling.ngh_sampler import NghSampler

            sampler = NghSampler(
                graph=mock_graph,
                fanout=[10],
                ring_buf=ring_buf,  # type: ignore[arg-type]
                sample_4_neg=True,
                n_neg=1,
                strategy="recent",
                seed=42,
                tgl_sampler_parity=True,
            )

            assert sampler.tgl_sampler_parity is True
            assert os.environ["FROST_DEBUG_TGL_BATCH_SHARED_RECENT"] == "1"
            MockSeqSampler.assert_called_once_with(
                graph=mock_graph,
                strategy="recent",
                seed=42,
                time_scaling_factor=0.0,
            )


class TestMultiHopNghSamplerInit:
    def test_stores_fanout_and_params(self):
        ring_buf = MockRingBuf()
        mock_graph = MagicMock()
        mock_graph.indices = np.array([1, 2], dtype=np.int64)
        mock_graph.ts = np.array([100, 200], dtype=np.int64)

        with patch("frost.sampling.ngh_sampler.SeqSampler"):
            from frost.sampling.ngh_sampler import MultiHopNghSampler

            sampler = MultiHopNghSampler(
                graph=mock_graph,
                fanout=[10, 5],
                ring_buf=ring_buf,  # type: ignore[arg-type]
                n_neg=1,
                strategy="uniform",
                seed=0,
                sample_4_neg=True,
            )

            assert sampler.fanout == [10, 5]
            assert sampler.n_neg == 1
            assert sampler.sample_4_neg is True
            assert sampler.padding_node_id == -1

    def test_stores_tgl_sampler_parity(self):
        ring_buf = MockRingBuf()
        mock_graph = MagicMock()
        mock_graph.indices = np.array([1, 2], dtype=np.int64)
        mock_graph.ts = np.array([100, 200], dtype=np.int64)

        with patch("frost.sampling.ngh_sampler.SeqSampler"):
            from frost.sampling.ngh_sampler import MultiHopNghSampler

            sampler = MultiHopNghSampler(
                graph=mock_graph,
                fanout=[10, 5],
                ring_buf=ring_buf,  # type: ignore[arg-type]
                n_neg=1,
                strategy="recent",
                seed=0,
                sample_4_neg=True,
                tgl_sampler_parity=True,
            )

            assert sampler.tgl_sampler_parity is True

    def test_sample_clears_reused_rows_below_padding_parents(self):
        ring_buf = MockRingBuf(total_size=100)
        ring_buf.slot.NID.fill_(777)
        ring_buf.slot.TS.fill_(777)
        ring_buf.slot.EID.fill_(777)
        mock_graph = MagicMock()
        mock_graph.indices = np.array([1, 2], dtype=np.int64)
        mock_graph.ts = np.array([100, 200], dtype=np.int64)

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

        with patch("frost.sampling.ngh_sampler.SeqSampler") as seq_sampler_cls:
            seq_sampler_cls.return_value.get_historical_neighbors.side_effect = [
                first_hop,
                second_hop,
            ]
            from frost.sampling.ngh_sampler import MultiHopNghSampler

            sampler = MultiHopNghSampler(
                graph=mock_graph,
                fanout=[2, 2],
                ring_buf=ring_buf,  # type: ignore[arg-type]
                n_neg=1,
                strategy="uniform",
                seed=0,
                sample_4_neg=True,
            )
            _, length = sampler.sample(
                root_nid=torch.tensor([1, 2, 3]),
                root_ts=torch.tensor([10]),
                sample_nid=torch.tensor([1, 2, 3]),
                sample_ts=torch.tensor([10, 10, 10]),
                pos_eid=torch.tensor([9]),
                gpu_id=0,
            )

        assert length == 3 + 6 + 12
        second_call_nodes = (
            seq_sampler_cls.return_value.get_historical_neighbors.call_args_list[1]
            .args[0]
            .tolist()
        )
        assert second_call_nodes == [10, 20, 21]

        second_tier = ring_buf.slot.NID[9:21].view(6, 2)
        second_tier_eid = ring_buf.slot.EID[9:21].view(6, 2)
        second_tier_ts = ring_buf.slot.TS[9:21].view(6, 2)
        assert second_tier[1].tolist() == [-1, -1]
        assert second_tier[4:].tolist() == [[-1, -1], [-1, -1]]
        assert (second_tier_eid[1] == -1).all()
        assert (second_tier_eid[4:] == -1).all()
        assert (second_tier_ts[1] == 0).all()
        assert (second_tier_ts[4:] == 0).all()
