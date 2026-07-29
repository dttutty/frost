from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from frost.data.preparation import (
    build_temporal_csr,
    convert_tensor_to_numpy,
    generate_node_roles,
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


def test_generate_node_roles_infers_bipartite_partitions(tmp_path: Path) -> None:
    edges_path = tmp_path / "edges.csv"
    output_path = tmp_path / "node_role.npy"
    _canonical_edges().to_csv(edges_path, index=False)

    summary = generate_node_roles(edges_path, output_path)

    roles = np.load(output_path)
    assert summary["num_nodes"] == 4
    assert roles.shape == (4, 1)
    np.testing.assert_array_equal(roles[:, 0], [False, False, True, True])


def test_generate_node_roles_rejects_overlapping_partitions(tmp_path: Path) -> None:
    edges = _canonical_edges()
    edges.loc[1, "dst"] = 1
    edges_path = tmp_path / "edges.csv"
    edges.to_csv(edges_path, index=False)

    with pytest.raises(ValueError, match="disjoint"):
        generate_node_roles(edges_path, tmp_path / "node_role.npy")


def test_convert_tensor_to_numpy_uses_explicit_paths(tmp_path: Path) -> None:
    source = tmp_path / "features.pt"
    output = tmp_path / "features.npy"
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save(tensor, source)

    summary = convert_tensor_to_numpy(source, output)

    assert summary["shape"] == [2, 3]
    assert summary["dtype"] == "float32"
    np.testing.assert_array_equal(np.load(output), tensor.numpy())


def test_validate_dataset_checks_graph_and_optional_row_contracts(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "EXAMPLE"
    dataset_dir.mkdir()
    edges_path = dataset_dir / "edges.csv"
    _canonical_edges().to_csv(edges_path, index=False)
    write_temporal_csr(
        edges_path,
        dataset_dir / "full_graph_with_reverse_edges.npz",
        add_reverse=True,
    )
    np.save(dataset_dir / "edge_features.npy", np.zeros((3, 2), dtype=np.float32))
    np.save(dataset_dir / "node_role.npy", np.zeros((4, 1), dtype=np.bool_))
    pd.DataFrame({"label": [False, True]}).to_csv(
        dataset_dir / "ban_labels.csv", index=False
    )

    summary = validate_dataset(dataset_dir)

    assert summary["dataset"] == "EXAMPLE"
    assert summary["num_nodes"] == 4
    assert summary["num_edges"] == 2
    assert summary["stored_adjacencies"] == 4
    assert summary["graph_direction"] == "bidirectional"
    assert summary["graph_artifact"] == "full_graph_with_reverse_edges.npz"
    assert summary["optional_artifacts"]["edge_features.npy"] == [3, 2]


def test_validate_dataset_accepts_tgl_directed_policy_for_gdelt(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "GDELT"
    dataset_dir.mkdir()
    edges_path = dataset_dir / "edges.csv"
    _canonical_edges().to_csv(edges_path, index=False)
    write_temporal_csr(
        edges_path,
        dataset_dir / "full_graph_no_reverse_edges.npz",
        add_reverse=False,
    )

    summary = validate_dataset(dataset_dir)

    assert summary["graph_direction"] == "directed"
    assert summary["stored_adjacencies"] == 2


def test_validate_dataset_rejects_reverse_edges_for_gdelt(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "GDELT"
    dataset_dir.mkdir()
    edges_path = dataset_dir / "edges.csv"
    _canonical_edges().to_csv(edges_path, index=False)
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
    edges = _canonical_edges()
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
