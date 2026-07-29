"""Generate dataset-release summaries from per-dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset_catalog(root: Path) -> list[dict[str, Any]]:
    """Return release facts derived exclusively from ``*/manifest.json`` files."""
    root = root.expanduser().resolve()
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = manifest.get("split", {}).get("counts")
        if not isinstance(counts, list) or len(counts) != 3:
            raise ValueError(f"{manifest_path} must define three split counts")
        split_counts = [int(value) for value in counts]
        if any(value <= 0 for value in split_counts):
            raise ValueError(f"{manifest_path} split counts must be positive")

        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError(f"{manifest_path} artifacts must be a mapping")
        entries.append(
            {
                "dataset": str(manifest["dataset"]),
                "num_edges": sum(split_counts),
                "train_edges": split_counts[0],
                "validation_edges": split_counts[1],
                "test_edges": split_counts[2],
                "max_eval_macro_batch_size": min(split_counts[1:]),
                "sampling_adjacency": str(manifest["graph"]["sampling_adjacency"]),
                "optional_artifacts": {
                    name: {
                        "shape": value["shape"],
                        "dtype": value["dtype"],
                    }
                    for name, value in sorted(artifacts.items())
                },
            }
        )
    if not entries:
        raise FileNotFoundError(f"No dataset manifests found under {root}")
    return entries


def render_dataset_catalog_markdown(entries: list[dict[str, Any]]) -> str:
    """Render manifest-derived facts as a stable Markdown table."""
    lines = [
        "# Dataset Catalog",
        "",
        "This file is generated from each dataset's `manifest.json` by FROST.",
        "Do not edit the table by hand.",
        "",
        "| Dataset | Edges | Train | Validation | Test | Max eval macro-batch | Sampling adjacency | Optional artifacts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for entry in entries:
        artifact_parts = []
        for name, metadata in entry["optional_artifacts"].items():
            shape = "x".join(str(value) for value in metadata["shape"])
            artifact_parts.append(f"`{name}` ({shape}, {metadata['dtype']})")
        artifacts = "<br>".join(artifact_parts) or "—"
        lines.append(
            "| {dataset} | {num_edges:,} | {train_edges:,} | "
            "{validation_edges:,} | {test_edges:,} | "
            "{max_eval_macro_batch_size:,} | {sampling_adjacency} | {artifacts} |".format(
                **entry, artifacts=artifacts
            )
        )
    return "\n".join(lines) + "\n"


def write_dataset_catalog(root: Path, output: Path, *, force: bool = False) -> dict[str, Any]:
    """Write a manifest-derived Markdown catalog atomically."""
    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    entries = load_dataset_catalog(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(render_dataset_catalog_markdown(entries), encoding="utf-8")
    temporary.replace(output)
    return {"datasets": len(entries), "output": str(output)}
