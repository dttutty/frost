from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from frost.shm import dataset_mgr as dm


class TestDatasetManager:
    def test_ignore_nfeat_on_disk_skips_node_feature_loading(
        self, tmp_path: Path, monkeypatch
    ):
        dataset_dir = tmp_path / "MOCK"
        dataset_dir.mkdir()
        np.save(dataset_dir / "node_features.npy", np.ones((4, 3), dtype=np.float32))

        monkeypatch.setattr(dm, "DATA_DIR", tmp_path)

        mgr = dm.DatasetManager(
            env=None,  # type: ignore[arg-type]
            run_id="test",
            dataset="MOCK",
            index_dtype=np.int64,
            time_dtype=np.int64,
            feature_dtype=torch.float32,
            ignore_nfeat_on_disk=True,
        )

        feat, dim, dtype = mgr._load_feature_to_shm(
            dataset="MOCK",
            feat_type="node",
            shm_name="nfeat",
            expected_rows=4,
        )

        assert feat is None
        assert dim == 0
        assert dtype is None


def _write_graph_fixture(dataset_dir: Path, *, add_reverse: bool) -> None:
    frame = pd.DataFrame(
        {
            "eid": [1, 2],
            "src": [0, 1],
            "dst": [1, 2],
            "time": [10, 20],
            "default_split": [0, 0],
        }
    )
    frame.to_csv(dataset_dir / "edges.csv", index=False)
    if add_reverse:
        indptr = np.array([0, 1, 3, 4], dtype=np.int64)
        indices = np.array([1, 0, 2, 1], dtype=np.int64)
        eid = np.array([1, 1, 2, 2], dtype=np.int64)
        ts = np.array([10, 10, 20, 20], dtype=np.int64)
        filename = "full_graph_with_reverse_edges.npz"
    else:
        indptr = np.array([0, 1, 2, 2], dtype=np.int64)
        indices = np.array([1, 2], dtype=np.int64)
        eid = np.array([1, 2], dtype=np.int64)
        ts = np.array([10, 20], dtype=np.int64)
        filename = "full_graph_no_reverse_edges.npz"
    np.savez(dataset_dir / filename, indptr=indptr, indices=indices, eid=eid, ts=ts)


def test_load_graph_accepts_directed_gdelt(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "GDELT"
    dataset_dir.mkdir()
    _write_graph_fixture(dataset_dir, add_reverse=False)
    sentinel = object()
    monkeypatch.setattr(dm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dm, "TemporalGraph", lambda **_: sentinel)

    graph, _, n_nodes, n_edges = dm.load_graph("GDELT", np.int64, np.int64)

    assert graph is sentinel
    assert n_nodes == 3
    assert n_edges == 2


def test_load_graph_rejects_bidirectional_mag(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "MAG"
    dataset_dir.mkdir()
    _write_graph_fixture(dataset_dir, add_reverse=True)
    monkeypatch.setattr(dm, "DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="requires a directed temporal graph"):
        dm.load_graph("MAG", np.int64, np.int64)
