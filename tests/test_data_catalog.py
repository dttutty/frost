from __future__ import annotations

import json
from pathlib import Path

import pytest

from frost.data.catalog import (
    load_dataset_catalog,
    render_dataset_catalog_markdown,
    write_dataset_catalog,
)


def _write_manifest(root: Path) -> None:
    dataset = root / "EXAMPLE"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "EXAMPLE",
                "split": {"counts": [70, 16, 14]},
                "graph": {"sampling_adjacency": "bidirectional"},
                "artifacts": {
                    "edge_features.npy": {"shape": [101, 4], "dtype": "float32"}
                },
            }
        ),
        encoding="utf-8",
    )


def test_catalog_is_derived_from_manifest_split_counts(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    entries = load_dataset_catalog(tmp_path)
    assert entries == [
        {
            "dataset": "EXAMPLE",
            "num_edges": 100,
            "train_edges": 70,
            "validation_edges": 16,
            "test_edges": 14,
            "max_eval_macro_batch_size": 14,
            "sampling_adjacency": "bidirectional",
            "optional_artifacts": {
                "edge_features.npy": {"shape": [101, 4], "dtype": "float32"}
            },
        }
    ]
    markdown = render_dataset_catalog_markdown(entries)
    assert "| EXAMPLE | 100 | 70 | 16 | 14 | 14 | bidirectional |" in markdown


def test_write_catalog_refuses_implicit_overwrite(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    output = tmp_path / "CATALOG.md"
    result = write_dataset_catalog(tmp_path, output)
    assert result == {"datasets": 1, "output": str(output.resolve())}
    assert output.read_text(encoding="utf-8").startswith("# Dataset Catalog")
    with pytest.raises(FileExistsError):
        write_dataset_catalog(tmp_path, output)
