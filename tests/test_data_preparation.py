from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from frost.data.dgb import import_dgb_snapshot
from frost.data.preparation import (
    assign_temporal_splits,
    build_temporal_csr,
    convert_tensor_to_numpy,
    validate_dataset,
    write_temporal_csr,
)


def _canonical_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eid": [2, 1],
            "src": [1, 1],
            "dst": [3, 2],
            "time": [20, 10],
            "default_split": [0, 0],
        }
    )


def _published_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eid": [1, 2, 3, 4, 5, 6],
            "src": [1, 1, 1, 1, 1, 1],
            "dst": [2, 3, 2, 3, 2, 3],
            "time": [10, 20, 30, 40, 50, 60],
            "default_split": [0, 0, 1, 1, 2, 2],
        }
    )


def test_build_temporal_csr_adds_reverse_edges_and_sorts_adjacencies() -> None:
    graph = build_temporal_csr(_canonical_edges(), add_reverse=True)

    np.testing.assert_array_equal(graph["indptr"], [0, 0, 2, 3, 4])
    np.testing.assert_array_equal(graph["indices"], [2, 3, 1, 1])
    np.testing.assert_array_equal(graph["ts"], [10, 20, 10, 20])
    np.testing.assert_array_equal(graph["eid"], [1, 2, 1, 2])


def test_build_temporal_csr_accepts_legacy_columns_without_reverse() -> None:
    edges = pd.DataFrame({"u": [2], "i": [4], "ts": [9], "idx": [7]})

    graph = build_temporal_csr(edges, add_reverse=False)

    np.testing.assert_array_equal(graph["indptr"], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(graph["indices"], [4])
    np.testing.assert_array_equal(graph["ts"], [9])
    np.testing.assert_array_equal(graph["eid"], [7])


def test_convert_tensor_to_numpy_uses_explicit_paths(tmp_path: Path) -> None:
    source = tmp_path / "features.pt"
    output = tmp_path / "features.npy"
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save(tensor, source)

    summary = convert_tensor_to_numpy(source, output)

    assert summary["shape"] == [2, 3]
    assert summary["dtype"] == "float32"
    np.testing.assert_array_equal(np.load(output), tensor.numpy())


def test_assign_temporal_splits_keeps_tied_timestamps_together() -> None:
    edges = pd.DataFrame(
        {
            "eid": np.arange(1, 11),
            "src": np.zeros(10, dtype=np.int64),
            "dst": np.ones(10, dtype=np.int64),
            "time": [0, 0, 1, 1, 2, 2, 2, 3, 4, 5],
        }
    )

    split = assign_temporal_splits(edges, train_ratio=0.5, val_ratio=0.2)

    for timestamp in edges["time"].unique():
        assert np.unique(split[edges["time"].eq(timestamp)]).size == 1
    assert set(split.tolist()) == {0, 1, 2}


def test_validate_dataset_checks_graph_and_optional_row_contracts(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "EXAMPLE"
    dataset_dir.mkdir()
    edges_path = dataset_dir / "edges.csv"
    edges = _published_edges()
    edges.to_csv(edges_path, index=False)
    write_temporal_csr(
        edges_path,
        dataset_dir / "full_graph_with_reverse_edges.npz",
        add_reverse=True,
    )
    np.save(dataset_dir / "edge_features.npy", np.zeros((7, 2), dtype=np.float32))
    summary = validate_dataset(dataset_dir)

    assert summary["dataset"] == "EXAMPLE"
    assert summary["num_nodes"] == 4
    assert summary["num_edges"] == 6
    assert summary["stored_adjacencies"] == 12
    assert summary["graph_direction"] == "bidirectional"
    assert summary["graph_artifact"] == "full_graph_with_reverse_edges.npz"
    assert summary["optional_artifacts"]["edge_features.npy"] == [7, 2]


def test_validate_dataset_accepts_tgl_directed_policy_for_gdelt(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "GDELT"
    dataset_dir.mkdir()
    edges_path = dataset_dir / "edges.csv"
    _published_edges().to_csv(edges_path, index=False)
    write_temporal_csr(
        edges_path,
        dataset_dir / "full_graph_no_reverse_edges.npz",
        add_reverse=False,
    )

    summary = validate_dataset(dataset_dir)

    assert summary["graph_direction"] == "directed"
    assert summary["stored_adjacencies"] == 6


def test_validate_dataset_rejects_reverse_edges_for_gdelt(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "GDELT"
    dataset_dir.mkdir()
    edges_path = dataset_dir / "edges.csv"
    _published_edges().to_csv(edges_path, index=False)
    write_temporal_csr(
        edges_path,
        dataset_dir / "full_graph_with_reverse_edges.npz",
        add_reverse=True,
    )

    with pytest.raises(ValueError, match="requires a directed temporal graph"):
        validate_dataset(dataset_dir)


def test_validate_dataset_detects_graph_drift(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "EXAMPLE"
    dataset_dir.mkdir()
    edges_path = dataset_dir / "edges.csv"
    edges = _published_edges()
    edges.to_csv(edges_path, index=False)
    graph = build_temporal_csr(edges, add_reverse=True)
    graph["indices"][0] = 3
    np.savez(dataset_dir / "full_graph_with_reverse_edges.npz", **graph)

    with pytest.raises(ValueError, match="indices"):
        validate_dataset(dataset_dir)


def test_write_temporal_csr_refuses_to_overwrite(tmp_path: Path) -> None:
    edges_path = tmp_path / "edges.csv"
    output_path = tmp_path / "graph.npz"
    _canonical_edges().to_csv(edges_path, index=False)
    write_temporal_csr(edges_path, output_path, add_reverse=True)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_temporal_csr(edges_path, output_path, add_reverse=True)


def test_import_dgb_snapshot_materializes_reproducible_release(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    output_dir = tmp_path / "MOOC"
    source_dir.mkdir()
    pd.DataFrame(
        {
            "u": [1, 2] * 5,
            "i": [10, 11] * 5,
            "ts": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "label": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            "idx": np.arange(1, 11),
        }
    ).to_csv(source_dir / "ml_mooc.csv", index=False)
    np.save(
        source_dir / "ml_mooc.npy",
        np.arange(44, dtype=np.float64).reshape(11, 4),
    )

    manifest = import_dgb_snapshot(
        source_dir,
        output_dir,
        dataset="MOOC",
        graph_direction="bidirectional",
    )

    assert manifest["split"]["strategy"] == "timestamp"
    assert manifest["negative_sampling"]["candidate_set"] == "observed_dst_nodes"
    assert not (output_dir / "node_features.npy").exists()
    assert not (output_dir / "state_labels.csv").exists()
    negatives = pd.read_csv(output_dir / "neg_pre_sampling_1_result.csv")
    assert set(negatives["neg_dst"]).issubset({10, 11})
    validate_dataset(output_dir)
