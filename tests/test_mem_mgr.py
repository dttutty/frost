"""Tests for MemBoxStore and MemBoxStoreManager."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

from frost.shm.mem_mgr import MemBoxStore


@dataclass
class MockMail:
    nid: torch.Tensor
    data: torch.Tensor
    ts: torch.Tensor


class TestMemBoxStore:
    def test_reset_zeros_all_tensors(self):
        store = MemBoxStore(
            mem=torch.ones(10, 16),
            mem_ts=torch.ones(10),
            box=torch.ones(10, 4, 32),
            box_ts=torch.ones(10, 4),
            box_pos=torch.ones(10, dtype=torch.long),
        )

        store.reset()

        assert (store.mem == 0).all()
        assert (store.mem_ts == 0).all()
        assert (store.box == 0).all()
        assert (store.box_ts == 0).all()
        assert (store.box_pos == 0).all()

    def test_write_mail_single_entry(self):
        n_nodes = 5
        boxsize = 3
        d_mail = 8

        store = MemBoxStore(
            mem=torch.zeros(n_nodes, 16),
            mem_ts=torch.zeros(n_nodes),
            box=torch.zeros(n_nodes, boxsize, d_mail),
            box_ts=torch.zeros(n_nodes, boxsize),
            box_pos=torch.zeros(n_nodes, dtype=torch.long),
        )

        mail = MockMail(
            nid=torch.tensor([0]),
            data=torch.ones(1, d_mail) * 7.0,
            ts=torch.tensor([100.0]),
        )

        store.write_mail(mail)

        assert store.box[0, 0].tolist() == [7.0] * d_mail
        assert store.box_ts[0, 0].item() == 100.0
        assert store.box_pos[0].item() == 1

    def test_write_mail_multiple_entries(self):
        n_nodes = 5
        boxsize = 4
        d_mail = 8

        store = MemBoxStore(
            mem=torch.zeros(n_nodes, 16),
            mem_ts=torch.zeros(n_nodes),
            box=torch.zeros(n_nodes, boxsize, d_mail),
            box_ts=torch.zeros(n_nodes, boxsize),
            box_pos=torch.zeros(n_nodes, dtype=torch.long),
        )

        mail = MockMail(
            nid=torch.tensor([1, 3]),
            data=torch.tensor([[1.0] * d_mail, [2.0] * d_mail]),
            ts=torch.tensor([10.0, 20.0]),
        )

        store.write_mail(mail)

        assert store.box_pos[1].item() == 1
        assert store.box_pos[3].item() == 1
        assert store.box[1, 0, 0].item() == 1.0
        assert store.box[3, 0, 0].item() == 2.0
        assert store.box_ts[1, 0].item() == 10.0
        assert store.box_ts[3, 0].item() == 20.0

    def test_write_mail_wraps_around(self):
        n_nodes = 2
        boxsize = 2
        d_mail = 4

        store = MemBoxStore(
            mem=torch.zeros(n_nodes, 16),
            mem_ts=torch.zeros(n_nodes),
            box=torch.zeros(n_nodes, boxsize, d_mail),
            box_ts=torch.zeros(n_nodes, boxsize),
            box_pos=torch.zeros(n_nodes, dtype=torch.long),
        )

        mail1 = MockMail(
            nid=torch.tensor([0]),
            data=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            ts=torch.tensor([100.0]),
        )
        store.write_mail(mail1)

        mail2 = MockMail(
            nid=torch.tensor([0]),
            data=torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
            ts=torch.tensor([200.0]),
        )
        store.write_mail(mail2)

        mail3 = MockMail(
            nid=torch.tensor([0]),
            data=torch.tensor([[9.0, 10.0, 11.0, 12.0]]),
            ts=torch.tensor([300.0]),
        )
        store.write_mail(mail3)

        assert store.box_pos[0].item() == 1
        assert store.box[0, 0].tolist() == [9.0, 10.0, 11.0, 12.0]
        assert store.box[0, 1].tolist() == [5.0, 6.0, 7.0, 8.0]

    def test_write_mail_moves_to_cpu(self):
        n_nodes = 3
        boxsize = 2
        d_mail = 4

        store = MemBoxStore(
            mem=torch.zeros(n_nodes, 16),
            mem_ts=torch.zeros(n_nodes),
            box=torch.zeros(n_nodes, boxsize, d_mail),
            box_ts=torch.zeros(n_nodes, boxsize),
            box_pos=torch.zeros(n_nodes, dtype=torch.long),
        )

        mail = MockMail(
            nid=torch.tensor([0], dtype=torch.int32),
            data=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64),
            ts=torch.tensor([100.0], dtype=torch.float64),
        )

        store.write_mail(mail)

        assert store.box[0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


class TestMemBoxStoreManagerInit:
    def test_initializes_with_correct_dimensions(self):
        mock_env = MagicMock()
        mock_env.is_host = True

        from frost.shm.mem_mgr import MemBoxStoreManager

        mgr = MemBoxStoreManager(
            env=mock_env,
            run_id="test_run",
            n_nodes=100,
            boxsize=10,
            d_mem=64,
            edim=16,
            mdtype=torch.float32,
            tdtype=torch.float32,
        )

        assert mgr.n_nodes == 100
        assert mgr.boxsize == 10
        assert mgr.d_mem == 64
        assert mgr.d_mail == 2 * 64 + 16
        assert mgr.mdtype == torch.float32
        assert mgr.state is None

    def test_d_mail_computation(self):
        mock_env = MagicMock()
        mock_env.is_host = True

        from frost.shm.mem_mgr import MemBoxStoreManager

        mgr = MemBoxStoreManager(
            env=mock_env,
            run_id="test",
            n_nodes=10,
            boxsize=5,
            d_mem=32,
            edim=8,
            mdtype=torch.float32,
            tdtype=torch.float32,
        )

        assert mgr.d_mail == 2 * 32 + 8

    def test_worker_prefaults_all_dynamic_state_mappings(self, monkeypatch):
        from frost.shm import mem_mgr

        mock_env = MagicMock()
        mock_env.is_host = False
        calls = []

        def fake_get(name, shape, dtype, *, populate_read=False):
            calls.append((name, tuple(shape), dtype, populate_read))
            return torch.empty(shape, dtype=dtype)

        monkeypatch.setattr(mem_mgr, "gsm", fake_get)
        monkeypatch.setattr(mem_mgr.dist, "barrier", lambda: None)

        mgr = mem_mgr.MemBoxStoreManager(
            env=mock_env,
            run_id="prefault",
            n_nodes=10,
            boxsize=1,
            d_mem=4,
            edim=0,
            mdtype=torch.float32,
            tdtype=torch.float32,
        )
        state = mgr.prepare()

        assert state.mem.shape == (10, 4)
        assert [call[0] for call in calls] == [
            "prefault_mem",
            "prefault_mem_ts",
            "prefault_box",
            "prefault_box_ts",
            "prefault_box_pos",
        ]
        assert all(call[3] for call in calls)
