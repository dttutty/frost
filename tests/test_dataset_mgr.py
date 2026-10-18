from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from frost.data.tgb import (
    load_tgb_features,
    load_tgb_frame,
    load_tgbl_wiki_edge_features,
    load_tgbl_wiki_frame,
)
from frost.shm import dataset_mgr as dm


class TestDatasetManager:
    def test_ignore_nfeat_on_disk_skips_node_feature_loading(
        self, tmp_path: Path, monkeypatch
    ):
        dataset_dir = tmp_path / "MOCK"
        dataset_dir.mkdir()
        np.save(dataset_dir / "node_features.npy", np.ones((4, 3), dtype=np.float32))

        monkeypatch.setenv("FROST_DATA_ROOT", str(tmp_path))

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
    monkeypatch.setenv("FROST_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(dm, "TemporalGraph", lambda **_: sentinel)

    graph, _, n_nodes, n_edges = dm.load_graph("GDELT", np.int64, np.int64)

    assert graph is sentinel
    assert n_nodes == 3
    assert n_edges == 2


def test_load_graph_rejects_bidirectional_mag(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "MAG"
    dataset_dir.mkdir()
    _write_graph_fixture(dataset_dir, add_reverse=True)
    monkeypatch.setenv("FROST_DATA_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="requires a directed temporal graph"):
        dm.load_graph("MAG", np.int64, np.int64)


def test_load_graph_normalizes_compressed_upstream_edges(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "MAG"
    dataset_dir.mkdir()
    frame = pd.DataFrame(
        {
            "src": [0, 1],
            "dst": [1, 2],
            "time": [10, 20],
            "int_roll": [0, 1],
            "ext_roll": [0, 2],
        }
    )
    frame.to_csv(dataset_dir / "edges.csv.zst", index=False)
    np.savez(
        dataset_dir / "full_graph_no_reverse_edges.npz",
        indptr=np.array([0, 1, 2, 2]),
        indices=np.array([1, 2]),
        eid=np.array([1, 2]),
        ts=np.array([10, 20]),
    )
    monkeypatch.setenv("FROST_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(dm, "TemporalGraph", lambda **_: object())

    _, loaded, _, _ = dm.load_graph("MAG", np.int64, np.int64)

    assert loaded.columns.tolist() == ["eid", "src", "dst", "time", "default_split"]
    assert loaded["eid"].tolist() == [1, 2]
    assert loaded["default_split"].tolist() == [0, 2]


def test_pt_features_keep_compact_storage_dtype(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "edge_features.pt"
    torch.save(torch.ones((3, 4), dtype=torch.bool), path)
    monkeypatch.setattr(
        dm,
        "csm",
        lambda _name, shape, dtype, **_kwargs: torch.empty(shape, dtype=dtype),
    )
    manager = dm.DatasetManager(
        env=None,  # type: ignore[arg-type]
        run_id="test",
        dataset="MOCK",
        index_dtype=np.int64,
        time_dtype=np.int64,
        feature_dtype=torch.float32,
    )

    feature, _, dtype = manager._load_pt_feature_to_shm(
        path, "edge", "efeat", expected_rows=2
    )

    assert dtype == torch.bool
    assert feature.dtype == torch.bool


def test_pt_node_features_stay_file_backed(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "node_features.pt"
    expected = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    torch.save(expected, path)
    monkeypatch.setattr(
        dm,
        "csm",
        lambda *_args, **_kwargs: pytest.fail(
            "file-backed node features must not allocate shared memory"
        ),
    )
    manager = dm.DatasetManager(
        env=None,  # type: ignore[arg-type]
        run_id="test",
        dataset="MOCK",
        index_dtype=np.int64,
        time_dtype=np.int64,
        feature_dtype=torch.float32,
    )

    feature, dim, dtype = manager._load_pt_feature_to_shm(
        path, "node", "nfeat", expected_rows=3
    )
    attached = manager._attach_feature(
        "nfeat",
        [3, 4],
        torch.float16,
        backing_path=str(path),
    )

    assert manager._file_backed_feature_paths == {"node": str(path.resolve())}
    assert dim == 4
    assert dtype == torch.float16
    assert attached is not None
    assert torch.equal(feature, expected)
    assert torch.equal(attached, expected)


def test_node_features_may_include_isolated_nodes(tmp_path: Path) -> None:
    path = tmp_path / "node_features.pt"

    dm._validate_feature_rows(
        feat_type="node",
        path=path,
        actual_rows=4,
        n_items=3,
    )
    with pytest.raises(ValueError, match="expected 3"):
        dm._validate_feature_rows(
            feat_type="node",
            path=path,
            actual_rows=2,
            n_items=3,
        )


def test_unpadded_edge_features_gain_reserved_zero_row(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "edge_features.pt"
    torch.save(torch.ones((2, 3), dtype=torch.bool), path)
    monkeypatch.setattr(
        dm,
        "csm",
        lambda _name, shape, dtype, **_kwargs: torch.empty(shape, dtype=dtype),
    )
    manager = dm.DatasetManager(
        env=None,  # type: ignore[arg-type]
        run_id="test",
        dataset="MOCK",
        index_dtype=np.int64,
        time_dtype=np.int64,
        feature_dtype=torch.float32,
    )

    feature, _, _ = manager._load_pt_feature_to_shm(
        path, "edge", "efeat", expected_rows=2
    )

    assert feature.shape == (3, 3)
    assert not feature[0].any()
    assert feature[1:].all()


def _write_tgbl_wiki_raw_fixture(tgb_root: Path) -> Path:
    raw_dir = tgb_root / "raw" / "tgbl_wiki"
    raw_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "u": [0, 1, 0, 2],
            "i": [3, 4, 4, 3],
            "ts": [10, 20, 30, 40],
            "idx": [0, 1, 2, 3],
            "w": [1.0, 1.0, 1.0, 1.0],
        }
    )
    frame.to_pickle(raw_dir / "ml_tgbl-wiki.pkl")
    with (raw_dir / "ml_tgbl-wiki_edge.pkl").open("wb") as handle:
        pickle.dump(np.arange(8, dtype=np.float64).reshape(4, 2), handle)
    return raw_dir


def test_generic_tgb_loader_reuses_official_processed_ids_and_features(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "tgbl_comment"
    raw_dir.mkdir(parents=True)
    processed = pd.DataFrame(
        {
            "u": [4.0, 1.0, 4.0, 2.0],
            "i": [2.0, 4.0, 3.0, 3.0],
            "ts": [10.0, 20.0, 30.0, 40.0],
            "idx": [1.0, 2.0, 3.0, 4.0],
            "w": [5.0, 6.0, 7.0, 8.0],
        }
    )
    processed.to_pickle(raw_dir / "ml_tgbl-comment.pkl")
    with (raw_dir / "ml_tgbl-comment_edge.pkl").open("wb") as handle:
        pickle.dump(np.arange(8, dtype=np.float64).reshape(4, 2), handle)

    frame = load_tgb_frame(raw_dir, "TGBL_COMMENT")
    features = load_tgb_features(raw_dir, "TGBL_COMMENT", "edge")

    assert frame["src"].tolist() == [4, 1, 4, 2]
    assert frame["dst"].tolist() == [2, 4, 3, 3]
    assert frame["default_split"].tolist() == [0, 0, 0, 2]
    assert features is not None
    assert features.shape == (4, 3)
    assert np.array_equal(features[:, -1], processed["w"].to_numpy())


def test_load_graph_builds_tgbl_wiki_from_official_raw_files(
    tmp_path: Path, monkeypatch
) -> None:
    _write_tgbl_wiki_raw_fixture(tmp_path)
    captured: dict[str, np.ndarray] = {}

    def fake_temporal_graph(**arrays):
        captured.update(arrays)
        return object()

    monkeypatch.setenv("TGB_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(dm, "TemporalGraph", fake_temporal_graph)

    _, frame, n_nodes, n_edges = dm.load_graph("TGBL_WIKI", np.int32, np.int64)

    assert n_nodes == 5
    assert n_edges == 4
    assert frame["src"].tolist() == [0, 1, 0, 2]
    assert frame["default_split"].tolist() == [0, 0, 0, 2]
    assert captured["indices"].dtype == np.int32
    assert len(captured["indices"]) == 2 * n_edges


def test_tgbl_wiki_edge_features_are_built_without_node_features(
    tmp_path: Path, monkeypatch
) -> None:
    _write_tgbl_wiki_raw_fixture(tmp_path)
    monkeypatch.setenv("TGB_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(
        dm,
        "csm",
        lambda _name, shape, dtype, **_kwargs: torch.empty(shape, dtype=dtype),
    )
    manager = dm.DatasetManager(
        env=None,  # type: ignore[arg-type]
        run_id="test",
        dataset="TGBL_WIKI",
        index_dtype=np.int64,
        time_dtype=np.int64,
        feature_dtype=torch.float32,
    )

    edge_features, edge_dim, _ = manager._load_feature_to_shm(
        "TGBL_WIKI", "edge", "efeat", expected_rows=4
    )
    node_features, node_dim, _ = manager._load_feature_to_shm(
        "TGBL_WIKI", "node", "nfeat", expected_rows=5
    )

    assert edge_dim == 2
    assert edge_features is not None
    assert torch.equal(edge_features[0], torch.zeros(2))
    assert torch.equal(
        edge_features[1:], torch.arange(8, dtype=torch.float32).reshape(4, 2)
    )
    assert node_dim == 0
    assert node_features is None


def test_tgbl_wiki_runtime_can_read_freshly_extracted_official_csv(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "tgbl_wiki"
    raw_dir.mkdir(parents=True)
    (raw_dir / "tgbl-wiki_edgelist_v2.csv").write_text(
        "user_id,item_id,timestamp,state_label,features\n"
        "0,0,10.0,0,1.5,2.5\n"
        "2,1,20.0,0,3.5,4.5\n",
        encoding="utf-8",
    )

    frame = load_tgbl_wiki_frame(raw_dir)
    features = load_tgbl_wiki_edge_features(raw_dir)

    assert frame["src"].tolist() == [0, 2]
    assert frame["dst"].tolist() == [3, 4]
    assert frame["time"].tolist() == [10, 20]
    assert np.array_equal(features, np.array([[1.5, 2.5], [3.5, 4.5]]))


def test_tgbl_wiki_frame_falls_back_when_pickle_is_incompatible(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw" / "tgbl_wiki"
    raw_dir.mkdir(parents=True)
    (raw_dir / "ml_tgbl-wiki.pkl").touch()
    (raw_dir / "tgbl-wiki_edgelist_v2.csv").write_text(
        "user_id,item_id,timestamp,state_label,features\n"
        "0,0,10.0,0,1.5\n"
        "1,1,20.0,0,2.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pd,
        "read_pickle",
        lambda _path: (_ for _ in ()).throw(NotImplementedError()),
    )

    frame = load_tgbl_wiki_frame(raw_dir)

    assert frame["src"].tolist() == [0, 1]
    assert frame["dst"].tolist() == [2, 3]
